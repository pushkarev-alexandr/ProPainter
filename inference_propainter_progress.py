# -*- coding: utf-8 -*-
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import imageio
import numpy as np
import scipy.ndimage
import torch
from PIL import Image
from tqdm import tqdm

from core.utils import to_tensors
from inference_propainter import (
    extrapolation,
    get_ref_index,
    imwrite,
    pretrain_model_url,
    read_frame_from_videos,
    resize_frames,
)
from model.misc import get_device
from model.modules.flow_comp_raft import RAFT_bi
from model.propainter import InpaintGenerator
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from utils.download_util import load_file_from_url
from utils.propainter_crop_utils import fit_crop_size_to_budget, smooth_center_trajectory


OBJECT_CROP_PADDING_RATIO = 0.20
OBJECT_BOXES_FILENAME = "object_boxes.json"

STAGE_RANGES: dict[str, tuple[float, float]] = {
    "prepare": (0.0, 8.0),
    "raft": (8.0, 36.0),
    "flow_complete": (36.0, 56.0),
    "img_prop": (56.0, 76.0),
    "transformer": (76.0, 96.0),
    "save": (96.0, 100.0),
}


@dataclass(slots=True)
class LoadedModels:
    fix_raft: RAFT_bi
    fix_flow_complete: RecurrentFlowCompleteNet
    model: InpaintGenerator


@dataclass(slots=True)
class CropWindow:
    x: int
    y: int
    width: int
    height: int


@dataclass(slots=True)
class ObjectTrack:
    label: int
    boxes_by_frame: dict[int, list[float]]


def emit_progress(
    stage: str,
    *,
    current: int,
    total: int,
    message: str,
    overall_start: float | None = None,
    overall_end: float | None = None,
) -> None:
    start, end = STAGE_RANGES[stage]
    safe_total = max(1, total)
    ratio = min(max(current / safe_total, 0.0), 1.0)
    overall = start + (end - start) * ratio
    if overall_start is not None and overall_end is not None:
        overall = overall_start + (overall_end - overall_start) * (overall / 100.0)
    payload = {
        "stage": stage,
        "current": current,
        "total": total,
        "stage_progress": round(ratio * 100.0, 2),
        "overall_progress": round(overall, 2),
        "message": message,
    }
    print(f"PROGRESS_JSON: {json.dumps(payload, ensure_ascii=True)}", flush=True)


def binary_mask(mask: np.ndarray, th: float = 0.1) -> np.ndarray:
    return np.where(mask > th, 1, 0)


def read_mask_images(mask_path: str | Path, length: int) -> list[Image.Image]:
    path = Path(mask_path)
    if path.suffix.lower() in {".jpg", ".jpeg", ".png"}:
        masks = [Image.open(path)]
    else:
        masks = [Image.open(item) for item in sorted(path.iterdir()) if item.suffix.lower() == ".png"]
    if len(masks) == 1:
        masks = masks * length
    if len(masks) != length:
        raise ValueError(f"Mask frame count ({len(masks)}) does not match video frame count ({length})")
    return masks


def prepare_mask_images(
    masks_img: list[Image.Image],
    *,
    size: tuple[int, int],
    flow_mask_dilates: int,
    mask_dilates: int,
) -> tuple[list[Image.Image], list[Image.Image]]:
    flow_masks: list[Image.Image] = []
    masks_dilated: list[Image.Image] = []
    for mask_img in masks_img:
        mask_img = mask_img.resize(size, Image.NEAREST)
        mask = np.array(mask_img.convert("L"))
        if flow_mask_dilates > 0:
            flow_mask = scipy.ndimage.binary_dilation(mask, iterations=flow_mask_dilates).astype(np.uint8)
        else:
            flow_mask = binary_mask(mask).astype(np.uint8)
        if mask_dilates > 0:
            dilated = scipy.ndimage.binary_dilation(mask, iterations=mask_dilates).astype(np.uint8)
        else:
            dilated = binary_mask(mask).astype(np.uint8)
        flow_masks.append(Image.fromarray(flow_mask * 255))
        masks_dilated.append(Image.fromarray(dilated * 255))
    return flow_masks, masks_dilated


def load_models(device: torch.device) -> LoadedModels:
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
    return LoadedModels(fix_raft=fix_raft, fix_flow_complete=fix_flow_complete, model=model)


def run_propainter_inpaint(
    *,
    frames_pil: list[Image.Image],
    flow_masks_pil: list[Image.Image],
    masks_dilated_pil: list[Image.Image],
    args: argparse.Namespace,
    device: torch.device,
    models: LoadedModels,
    use_half: bool,
    progress_start: float | None = None,
    progress_end: float | None = None,
    progress_message_suffix: str = "",
) -> list[np.ndarray]:
    video_length = len(frames_pil)
    w, h = frames_pil[0].size
    frames_inp = [np.array(frame).astype(np.uint8) for frame in frames_pil]
    frames = to_tensors()(frames_pil).unsqueeze(0) * 2 - 1
    flow_masks = to_tensors()(flow_masks_pil).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated_pil).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

    def progress(stage: str, current: int, total: int, message: str) -> None:
        emit_progress(
            stage,
            current=current,
            total=total,
            message=f"{message}{progress_message_suffix}",
            overall_start=progress_start,
            overall_end=progress_end,
        )

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
                    flows_f, flows_b = models.fix_raft(frames[:, f:end_f], iters=args.raft_iter)
                else:
                    flows_f, flows_b = models.fix_raft(frames[:, f - 1 : end_f], iters=args.raft_iter)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
                progress("raft", step_idx, raft_steps, "RAFT flow estimation")
            gt_flows_bi = (torch.cat(gt_flows_f_list, dim=1), torch.cat(gt_flows_b_list, dim=1))
        else:
            gt_flows_bi = models.fix_raft(frames, iters=args.raft_iter)
            torch.cuda.empty_cache()
            progress("raft", 1, 1, "RAFT flow estimation")

        fix_flow_complete = models.fix_flow_complete
        model = models.model
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
                progress("flow_complete", step_idx, flow_steps, "Flow completion")
            pred_flows_bi = (torch.cat(pred_flows_f, dim=1), torch.cat(pred_flows_b, dim=1))
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()
            progress("flow_complete", 1, 1, "Flow completion")

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
                progress("img_prop", step_idx, img_steps, "Image propagation")
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated, "nearest")
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()
            progress("img_prop", 1, 1, "Image propagation")

    comp_frames: list[np.ndarray | None] = [None] * video_length
    neighbor_stride = args.neighbor_length // 2
    ref_num = args.subvideo_length // args.ref_stride if video_length > args.subvideo_length else -1
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
            pred_img = models.model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
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
                    soft_masks[m_idx, :, :, 0] = cv2.GaussianBlur(soft_masks[m_idx, :, :, 0], (feather, feather), 0)
            for i, idx in enumerate(neighbor_ids):
                alpha = np.clip(soft_masks[i], 0.0, 1.0)
                img = np.array(pred_img[i], dtype=np.float32) * alpha + frames_inp[idx].astype(np.float32) * (1.0 - alpha)
                img = np.clip(img, 0, 255).astype(np.uint8)
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = (comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5).astype(
                        np.uint8
                    )
        torch.cuda.empty_cache()
        progress("transformer", step_idx, transformer_steps, "Transformer inpainting")

    return [frame if frame is not None else frames_inp[idx] for idx, frame in enumerate(comp_frames)]


def load_label_masks(mask_path: str | Path, length: int, size: tuple[int, int]) -> list[np.ndarray]:
    masks = read_mask_images(mask_path, length)
    return [np.array(mask.resize(size, Image.NEAREST).convert("L"), dtype=np.uint8) for mask in masks]


def load_object_tracks(object_boxes_path: str | Path) -> list[ObjectTrack]:
    payload = json.loads(Path(object_boxes_path).read_text(encoding="utf-8"))
    tracks: dict[int, dict[int, list[float]]] = {}
    for frame in payload.get("frames", []):
        frame_index = int(frame.get("frame_index", 0))
        for obj in frame.get("objects", []):
            label = int(obj.get("label", 0))
            box = obj.get("box_xywh")
            if label <= 0 or not isinstance(box, list) or len(box) != 4:
                continue
            tracks.setdefault(label, {})[frame_index] = [float(value) for value in box]
    return [ObjectTrack(label=label, boxes_by_frame=boxes) for label, boxes in sorted(tracks.items())]


def normalized_box_to_pixels(box_xywh: list[float], frame_width: int, frame_height: int) -> tuple[float, float, float, float]:
    x, y, w, h = box_xywh
    x1 = x * frame_width
    y1 = y * frame_height
    return x1, y1, w * frame_width, h * frame_height


def build_crop_windows_union(
    *,
    track: ObjectTrack,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    padding_ratio: float = OBJECT_CROP_PADDING_RATIO,
) -> list[CropWindow]:
    """One fixed crop rectangle for all frames: union of per-frame expanded boxes.

    ``fit_crop_size_to_budget`` scales the union so its long side is ``OBJECT_CROP_TARGET_LONG_SIDE``
    (512 by default). If the budget step shrinks the window below the pixel union, the window stays
    centered on the union and clamped to the frame; some object pixels may fall outside the crop.
    """
    ux1 = math.inf
    uy1 = math.inf
    ux2 = -math.inf
    uy2 = -math.inf
    for _frame_index, box in track.boxes_by_frame.items():
        x, y, width, height = normalized_box_to_pixels(box, frame_width, frame_height)
        if width <= 0 or height <= 0:
            continue
        expanded_width = width * (1.0 + padding_ratio * 2.0)
        expanded_height = height * (1.0 + padding_ratio * 2.0)
        cx = x + width / 2.0
        cy = y + height / 2.0
        half_w = expanded_width / 2.0
        half_h = expanded_height / 2.0
        rx1 = cx - half_w
        ry1 = cy - half_h
        rx2 = cx + half_w
        ry2 = cy + half_h
        ux1 = min(ux1, rx1)
        uy1 = min(uy1, ry1)
        ux2 = max(ux2, rx2)
        uy2 = max(uy2, ry2)

    if ux1 == math.inf:
        raise ValueError(f"No valid boxes found for object label {track.label}")

    union_w = ux2 - ux1
    union_h = uy2 - uy1
    crop_width, crop_height = fit_crop_size_to_budget(
        width=union_w,
        height=union_h,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    cx = (ux1 + ux2) / 2.0
    cy = (uy1 + uy2) / 2.0
    win_x = int(round(cx - crop_width / 2.0))
    win_y = int(round(cy - crop_height / 2.0))
    win_x = max(0, min(win_x, frame_width - crop_width))
    win_y = max(0, min(win_y, frame_height - crop_height))
    window = CropWindow(x=win_x, y=win_y, width=crop_width, height=crop_height)
    return [window] * frame_count


def build_crop_windows_tracking(
    *,
    track: ObjectTrack,
    frame_count: int,
    frame_width: int,
    frame_height: int,
    padding_ratio: float = OBJECT_CROP_PADDING_RATIO,
    center_smooth_sigma: float = 0.0,
) -> list[CropWindow]:
    visible: dict[int, tuple[float, float, float, float]] = {}
    max_width = 0.0
    max_height = 0.0
    for frame_index, box in track.boxes_by_frame.items():
        x, y, width, height = normalized_box_to_pixels(box, frame_width, frame_height)
        if width <= 0 or height <= 0:
            continue
        expanded_width = width * (1.0 + padding_ratio * 2.0)
        expanded_height = height * (1.0 + padding_ratio * 2.0)
        visible[frame_index] = (x + width / 2.0, y + height / 2.0, expanded_width, expanded_height)
        max_width = max(max_width, expanded_width)
        max_height = max(max_height, expanded_height)

    if not visible:
        raise ValueError(f"No valid boxes found for object label {track.label}")

    crop_width, crop_height = fit_crop_size_to_budget(
        width=max_width,
        height=max_height,
        frame_width=frame_width,
        frame_height=frame_height,
    )
    visible_indices = sorted(visible)
    raw_centers: list[tuple[float, float]] = []
    for frame_index in range(frame_count):
        if frame_index in visible:
            cx, cy = visible[frame_index][0], visible[frame_index][1]
        else:
            nearest = min(visible_indices, key=lambda idx: abs(idx - frame_index))
            cx, cy = visible[nearest][0], visible[nearest][1]
        raw_centers.append((cx, cy))

    centers = smooth_center_trajectory(raw_centers, sigma=center_smooth_sigma)
    windows: list[CropWindow] = []
    for cx, cy in centers:
        x = int(round(cx - crop_width / 2.0))
        y = int(round(cy - crop_height / 2.0))
        x = max(0, min(x, frame_width - crop_width))
        y = max(0, min(y, frame_height - crop_height))
        windows.append(CropWindow(x=x, y=y, width=crop_width, height=crop_height))
    return windows


def crop_with_padding(
    image: np.ndarray,
    window: CropWindow,
    *,
    border_type: int,
    value: int | tuple[int, int, int] = 0,
) -> np.ndarray:
    img_h, img_w = image.shape[:2]
    x1, y1 = window.x, window.y
    x2, y2 = window.x + window.width, window.y + window.height
    src_x1, src_y1 = max(0, x1), max(0, y1)
    src_x2, src_y2 = min(img_w, x2), min(img_h, y2)
    crop = image[src_y1:src_y2, src_x1:src_x2]
    if crop.size == 0:
        shape = (window.height, window.width) if image.ndim == 2 else (window.height, window.width, image.shape[2])
        return np.zeros(shape, dtype=image.dtype)
    pad_left = src_x1 - x1
    pad_top = src_y1 - y1
    pad_right = x2 - src_x2
    pad_bottom = y2 - src_y2
    return cv2.copyMakeBorder(crop, pad_top, pad_bottom, pad_left, pad_right, border_type, value=value)


def paste_object_crop(
    *,
    composite_frames: list[np.ndarray],
    crop_frames: list[np.ndarray],
    crop_masks: list[np.ndarray],
    windows: list[CropWindow],
    blend_feather: int,
) -> None:
    feather = max(0, int(blend_feather))
    if feather > 0 and feather % 2 == 0:
        feather += 1
    for idx, window in enumerate(windows):
        frame = composite_frames[idx]
        frame_h, frame_w = frame.shape[:2]
        x1, y1 = window.x, window.y
        x2, y2 = window.x + window.width, window.y + window.height
        dst_x1, dst_y1 = max(0, x1), max(0, y1)
        dst_x2, dst_y2 = min(frame_w, x2), min(frame_h, y2)
        if dst_x1 >= dst_x2 or dst_y1 >= dst_y2:
            continue
        crop_x1, crop_y1 = dst_x1 - x1, dst_y1 - y1
        crop_x2, crop_y2 = crop_x1 + (dst_x2 - dst_x1), crop_y1 + (dst_y2 - dst_y1)
        mask = crop_masks[idx][crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32) / 255.0
        if not np.any(mask):
            continue
        if feather > 0:
            mask = cv2.GaussianBlur(mask, (feather, feather), 0)
        alpha = np.clip(mask[:, :, None], 0.0, 1.0)
        source = crop_frames[idx][crop_y1:crop_y2, crop_x1:crop_x2].astype(np.float32)
        target = frame[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
        frame[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(source * alpha + target * (1.0 - alpha), 0, 255).astype(np.uint8)


def save_video_outputs(
    *,
    save_root: str | Path,
    comp_frames: list[np.ndarray],
    out_size: tuple[int, int],
) -> None:
    save_root = Path(save_root)
    frame_total = max(1, len(comp_frames))
    for idx, frame in enumerate(comp_frames):
        output = cv2.resize(frame, out_size, interpolation=cv2.INTER_CUBIC)
        output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        imwrite(output, str(save_root / "frames" / f"{idx:04d}.png"))
        emit_progress("save", current=idx + 1, total=frame_total, message="Saving output frames")


def run_full_frame_mode(
    *,
    args: argparse.Namespace,
    device: torch.device,
    models: LoadedModels,
    use_half: bool,
    frames: list[Image.Image],
    fps: float,
    size: tuple[int, int],
    out_size: tuple[int, int],
    save_root: Path,
) -> None:
    if args.mode == "video_inpainting":
        masks_img = read_mask_images(args.mask, len(frames))
        flow_masks, masks_dilated = prepare_mask_images(
            masks_img,
            size=size,
            flow_mask_dilates=args.mask_dilation,
            mask_dilates=args.mask_dilation,
        )
    elif args.mode == "video_outpainting":
        if args.scale_h is None or args.scale_w is None:
            raise AssertionError("Please provide a outpainting scale (s_h, s_w).")
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
    else:
        raise NotImplementedError

    emit_progress("prepare", current=1, total=1, message="Input prepared")
    comp_frames = run_propainter_inpaint(
        frames_pil=frames,
        flow_masks_pil=flow_masks,
        masks_dilated_pil=masks_dilated,
        args=args,
        device=device,
        models=models,
        use_half=use_half,
    )
    save_video_outputs(save_root=save_root, comp_frames=comp_frames, out_size=out_size)


def run_object_crop_mode(
    *,
    args: argparse.Namespace,
    device: torch.device,
    models: LoadedModels,
    use_half: bool,
    frames: list[Image.Image],
    fps: float,
    out_size: tuple[int, int],
    save_root: Path,
) -> None:
    frame_arrays = [np.array(frame).astype(np.uint8) for frame in frames]
    frame_height, frame_width = frame_arrays[0].shape[:2]
    label_masks = load_label_masks(args.mask, len(frames), (frame_width, frame_height))
    tracks = load_object_tracks(args.object_boxes)
    if not tracks:
        raise ValueError(f"No object tracks found in {args.object_boxes}")

    composite_frames = [frame.copy() for frame in frame_arrays]
    emit_progress("prepare", current=1, total=1, message=f"Prepared {len(tracks)} object tracks")
    work_start, work_end = STAGE_RANGES["raft"][0], STAGE_RANGES["transformer"][1]
    object_span = (work_end - work_start) / max(1, len(tracks))

    debug_root = save_root / "object_debug_crops"
    if args.save_object_debug_crops:
        debug_root.mkdir(parents=True, exist_ok=True)

    for object_index, track in enumerate(tracks):
        if args.object_crop_strategy == "union":
            windows = build_crop_windows_union(
                track=track,
                frame_count=len(frames),
                frame_width=frame_width,
                frame_height=frame_height,
            )
        else:
            windows = build_crop_windows_tracking(
                track=track,
                frame_count=len(frames),
                frame_width=frame_width,
                frame_height=frame_height,
                center_smooth_sigma=args.crop_center_smooth_sigma,
            )
        crop_frames_np: list[np.ndarray] = []
        crop_masks_np: list[np.ndarray] = []
        for frame_idx, window in enumerate(windows):
            object_mask = np.where(label_masks[frame_idx] == track.label, 255, 0).astype(np.uint8)
            crop_frames_np.append(
                crop_with_padding(frame_arrays[frame_idx], window, border_type=cv2.BORDER_REPLICATE)
            )
            crop_masks_np.append(
                crop_with_padding(object_mask, window, border_type=cv2.BORDER_CONSTANT, value=0)
            )

        if args.save_object_debug_crops:
            imageio.mimwrite(
                str(debug_root / f"object_{track.label:03d}_crop.mp4"),
                crop_frames_np,
                fps=fps,
                quality=7,
                macro_block_size=1,
            )

        crop_frames_pil = [Image.fromarray(frame) for frame in crop_frames_np]
        crop_masks_pil = [Image.fromarray(mask) for mask in crop_masks_np]
        flow_masks, masks_dilated_pil = prepare_mask_images(
            crop_masks_pil,
            size=crop_frames_pil[0].size,
            flow_mask_dilates=args.mask_dilation,
            mask_dilates=args.mask_dilation,
        )
        crop_masks_dilated_np = [
            np.array(m.convert("L"), dtype=np.uint8) for m in masks_dilated_pil
        ]
        progress_start = work_start + object_span * object_index
        progress_end = progress_start + object_span
        crop_result = run_propainter_inpaint(
            frames_pil=crop_frames_pil,
            flow_masks_pil=flow_masks,
            masks_dilated_pil=masks_dilated_pil,
            args=args,
            device=device,
            models=models,
            use_half=use_half,
            progress_start=progress_start,
            progress_end=progress_end,
            progress_message_suffix=f" (object {track.label})",
        )
        paste_object_crop(
            composite_frames=composite_frames,
            crop_frames=crop_result,
            crop_masks=crop_masks_dilated_np,
            windows=windows,
            blend_feather=args.blend_feather,
        )

    save_video_outputs(
        save_root=save_root,
        comp_frames=composite_frames,
        out_size=out_size,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--video", type=str, default="inputs/object_removal/bmx-trees")
    parser.add_argument("-m", "--mask", type=str, default="inputs/object_removal/bmx-trees_mask")
    parser.add_argument("-o", "--output", type=str, default="results")
    parser.add_argument("--object_boxes", type=str, default=None)
    parser.add_argument("--save_object_debug_crops", action="store_true")
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
        "--object_crop_strategy",
        type=str,
        default="union",
        choices=["union", "tracking"],
        help="union: fixed crop from union of all boxes per track; tracking: moving window + optional center smoothing.",
    )
    parser.add_argument(
        "--crop_center_smooth_sigma",
        type=float,
        default=0.0,
        help="Gaussian smoothing sigma in frames for object crop centers when --object_crop_strategy tracking (0 disables).",
    )
    parser.add_argument(
        "--blend_feather",
        type=int,
        default=0,
        help="Gaussian blur kernel size for soft mask blending (0 disables feathering).",
    )
    return parser


def main() -> None:
    device = get_device()
    args = build_parser().parse_args()
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
    save_root = Path(args.output) / video_name
    save_root.mkdir(parents=True, exist_ok=True)

    models = load_models(device)
    print(f"\nProcessing: {video_name} [{len(frames)} frames]...")
    if args.object_boxes:
        run_object_crop_mode(
            args=args,
            device=device,
            models=models,
            use_half=use_half,
            frames=frames,
            fps=fps,
            out_size=out_size,
            save_root=save_root,
        )
    else:
        run_full_frame_mode(
            args=args,
            device=device,
            models=models,
            use_half=use_half,
            frames=frames,
            fps=fps,
            size=size,
            out_size=out_size,
            save_root=save_root,
        )
    print(f"\nAll results are saved in {save_root}")
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
