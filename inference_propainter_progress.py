# -*- coding: utf-8 -*-
import argparse
import json
import os

import cv2
import imageio
import numpy as np
import torch
from tqdm import tqdm

from core.utils import to_tensors
from inference_propainter import (
    extrapolation,
    get_ref_index,
    imwrite,
    pretrain_model_url,
    read_frame_from_videos,
    read_mask,
    resize_frames,
)
from model.misc import get_device
from model.modules.flow_comp_raft import RAFT_bi
from model.propainter import InpaintGenerator
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from utils.download_util import load_file_from_url


STAGE_RANGES: dict[str, tuple[float, float]] = {
    "prepare": (0.0, 8.0),
    "raft": (8.0, 36.0),
    "flow_complete": (36.0, 56.0),
    "img_prop": (56.0, 76.0),
    "transformer": (76.0, 96.0),
    "save": (96.0, 100.0),
}


def emit_progress(
    stage: str,
    *,
    current: int,
    total: int,
    message: str,
) -> None:
    start, end = STAGE_RANGES[stage]
    safe_total = max(1, total)
    ratio = min(max(current / safe_total, 0.0), 1.0)
    overall = start + (end - start) * ratio
    payload = {
        "stage": stage,
        "current": current,
        "total": total,
        "stage_progress": round(ratio * 100.0, 2),
        "overall_progress": round(overall, 2),
        "message": message,
    }
    print(f"PROGRESS_JSON: {json.dumps(payload, ensure_ascii=True)}", flush=True)


if __name__ == "__main__":
    device = get_device()

    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video", type=str, default="inputs/object_removal/bmx-trees")
    parser.add_argument("-m", "--mask", type=str, default="inputs/object_removal/bmx-trees_mask")
    parser.add_argument("-o", "--output", type=str, default="results")
    parser.add_argument("--resize_ratio", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=-1)
    parser.add_argument("--width", type=int, default=-1)
    parser.add_argument("--mask_dilation", type=int, default=4)
    parser.add_argument("--ref_stride", type=int, default=10)
    parser.add_argument("--neighbor_length", type=int, default=10)
    parser.add_argument("--subvideo_length", type=int, default=80)
    parser.add_argument("--raft_iter", type=int, default=20)
    parser.add_argument("--mode", default="video_inpainting", choices=["video_inpainting", "video_outpainting"])
    parser.add_argument("--scale_h", type=float, default=1.0)
    parser.add_argument("--scale_w", type=float, default=1.2)
    parser.add_argument("--save_fps", type=int, default=24)
    parser.add_argument("--save_frames", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument(
        "--blend_feather",
        type=int,
        default=0,
        help="Gaussian blur kernel size for soft mask blending (0 disables feathering).",
    )
    args = parser.parse_args()

    use_half = bool(args.fp16)
    if device == torch.device("cpu"):
        use_half = False

    emit_progress("prepare", current=0, total=1, message="Reading input video and mask")
    frames, fps, size, video_name = read_frame_from_videos(args.video)
    if args.width != -1 and args.height != -1:
        size = (args.width, args.height)
    if args.resize_ratio != 1.0:
        size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))
    frames, size, out_size = resize_frames(frames, size)
    fps = args.save_fps if fps is None else fps
    save_root = os.path.join(args.output, video_name)
    os.makedirs(save_root, exist_ok=True)

    if args.mode == "video_inpainting":
        frames_len = len(frames)
        flow_masks, masks_dilated = read_mask(
            args.mask,
            frames_len,
            size,
            flow_mask_dilates=args.mask_dilation,
            mask_dilates=args.mask_dilation,
        )
        w, h = size
    elif args.mode == "video_outpainting":
        if args.scale_h is None or args.scale_w is None:
            raise AssertionError("Please provide a outpainting scale (s_h, s_w).")
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError

    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]), 2).repeat(3, axis=2) / 255.0
        img = np.array(frames[i])
        green = np.zeros([h, w, 3])
        green[:, :, 1] = 255
        alpha = 0.6
        fuse_img = (1 - alpha) * img + alpha * green
        fuse_img = mask_ * fuse_img + (1 - mask_) * img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))
    emit_progress("prepare", current=1, total=1, message="Input prepared")

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, "raft-things.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )
    fix_raft = RAFT_bi(ckpt_path, device)

    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, "recurrent_flow_completion.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters():
        p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    ckpt_path = load_file_from_url(
        url=os.path.join(pretrain_model_url, "ProPainter.pth"),
        model_dir="weights",
        progress=True,
        file_name=None,
    )
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    video_length = frames.size(1)
    print(f"\nProcessing: {video_name} [{video_length} frames]...")
    with torch.no_grad():
        if frames.size(-1) <= 640:
            short_clip_len = 12
        elif frames.size(-1) <= 720:
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            raft_steps = len(range(0, video_length, short_clip_len))
            for step_idx, f in enumerate(range(0, video_length, short_clip_len), start=1):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:, f:end_f], iters=args.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:, f - 1 : end_f], iters=args.raft_iter)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                emit_progress("raft", current=step_idx, total=raft_steps, message="RAFT flow estimation")
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=args.raft_iter)
            torch.cuda.empty_cache()
            emit_progress("raft", current=1, total=1, message="RAFT flow estimation")

        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        flow_length = gt_flows_bi[0].size(1)
        if flow_length > args.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            flow_steps = len(range(0, flow_length, args.subvideo_length))
            for step_idx, f in enumerate(range(0, flow_length, args.subvideo_length), start=1):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    flow_masks[:, s_f : e_f + 1],
                )
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]),
                    pred_flows_bi_sub,
                    flow_masks[:, s_f : e_f + 1],
                )
                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s : e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s : e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
                emit_progress("flow_complete", current=step_idx, total=flow_steps, message="Flow completion")
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()
            emit_progress("flow_complete", current=1, total=1, message="Flow completion")

        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, args.subvideo_length)
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            img_steps = len(range(0, video_length, subvideo_length_img_prop))
            for step_idx, f in enumerate(range(0, video_length, subvideo_length_img_prop), start=1):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)

                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f : e_f - 1], pred_flows_bi[1][:, s_f : e_f - 1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(
                    masked_frames[:, s_f:e_f],
                    pred_flows_bi_sub,
                    masks_dilated[:, s_f:e_f],
                    "nearest",
                )
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + prop_imgs_sub.view(
                    b, t, 3, h, w
                ) * masks_dilated[:, s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)

                updated_frames.append(updated_frames_sub[:, pad_len_s : e_f - s_f - pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s : e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
                emit_progress("img_prop", current=step_idx, total=img_steps, message="Image propagation")

            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, "nearest")
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            emit_progress("img_prop", current=1, total=1, message="Image propagation")

    ori_frames = frames_inp
    comp_frames = [None] * video_length
    neighbor_stride = args.neighbor_length // 2
    if video_length > args.subvideo_length:
        ref_num = args.subvideo_length // args.ref_stride
    else:
        ref_num = -1

    transformer_steps = len(range(0, video_length, neighbor_stride))
    for step_idx, f in enumerate(tqdm(range(0, video_length, neighbor_stride)), start=1):
        neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
            pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :],
            pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :],
        )

        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            soft_masks = binary_masks.astype(np.float32)
            feather = max(0, int(args.blend_feather))
            if feather > 0:
                if feather % 2 == 0:
                    feather += 1
                for m_idx in range(len(neighbor_ids)):
                    # Blur mask edges to reduce hard seams between inpainted and original content.
                    soft_masks[m_idx, :, :, 0] = cv2.GaussianBlur(soft_masks[m_idx, :, :, 0], (feather, feather), 0)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                alpha = np.clip(soft_masks[i], 0.0, 1.0)
                img = np.array(pred_img[i], dtype=np.float32) * alpha + ori_frames[idx].astype(np.float32) * (1.0 - alpha)
                img = np.clip(img, 0, 255).astype(np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        torch.cuda.empty_cache()
        emit_progress("transformer", current=step_idx, total=transformer_steps, message="Transformer inpainting")

    if args.save_frames:
        frame_total = max(1, video_length)
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, "frames", str(idx).zfill(4) + ".png")
            imwrite(f, img_save_root)
            emit_progress("save", current=idx + 1, total=frame_total, message="Saving output frames")

    masked_frame_for_save = [cv2.resize(f, out_size) for f in masked_frame_for_save]
    comp_frames = [cv2.resize(f, out_size) for f in comp_frames]
    imageio.mimwrite(os.path.join(save_root, "masked_in.mp4"), masked_frame_for_save, fps=fps, quality=7, macro_block_size=1)
    imageio.mimwrite(os.path.join(save_root, "inpaint_out.mp4"), comp_frames, fps=fps, quality=7, macro_block_size=1)
    emit_progress("save", current=1, total=1, message="Video outputs saved")

    print(f"\nAll results are saved in {save_root}")
    torch.cuda.empty_cache()
