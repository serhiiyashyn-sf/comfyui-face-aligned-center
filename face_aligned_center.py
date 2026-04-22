"""
Face-Aligned Center — ComfyUI node for batch-aware character centering.

Designed for multi-angle character sheets (typically 8 angles).

Pipeline:
    Pass 1: per image, auto-detect the background color from the border,
            build a foreground mask, analyze the silhouette (trimmed body
            extent + head region).
    Pass 2: one batch scale = target_body_h / median(body_h) across the
            batch, where target_body_h is derived from face_fill via a
            chibi-typical face/body ratio (0.21). Every image uses the
            same scale so all angles come out at the same physical size.
    Pass 3: translate so the silhouette head-cx lands at the user-specified
            canvas target. Padding is filled with the image's own detected
            BG color so white/green/black/etc. backgrounds stay seamless.

Why batch-aware: within one character sheet, every image is the same
character at the same camera distance (just rotated). Using the batch
median body_h smooths out per-image silhouette noise, so all 8 angles end
up at the same output size.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


_BG_TOLERANCE = 10              # max-channel abs diff; matches 'min(RGB) < 245' on white BG
_BG_CORNER_FRAC = 0.03          # fraction of image edge sampled for BG color detection
_ROW_DENSITY_TRIM = 0.25        # trim top/bottom rows thinner than 25% of the densest row
_HEAD_TOP_FRACTION = 0.30       # top 30% of the trimmed body is treated as the head region
_ASSUMED_FACE_TO_BODY = 0.21    # chibi-typical face height / body height ratio


def _sample_bg_color(img_np: np.ndarray) -> np.ndarray:
    """Return the median color along the image border (int32 RGB triple).

    Samples a full outer frame (top + bottom + left + right strips) rather
    than just 4 corners, so the median stays stable even when a character
    touches or extends close to an edge.
    """
    h, w = img_np.shape[:2]
    pad = max(2, int(min(h, w) * _BG_CORNER_FRAC))
    border = np.concatenate([
        img_np[:pad, :].reshape(-1, 3),
        img_np[-pad:, :].reshape(-1, 3),
        img_np[pad:-pad, :pad].reshape(-1, 3),
        img_np[pad:-pad, -pad:].reshape(-1, 3),
    ]).astype(np.int32)
    return np.median(border, axis=0).astype(np.int32)


def _detect_fg(img_np: np.ndarray, bg: np.ndarray) -> np.ndarray:
    """A pixel is foreground iff its MAX per-channel absolute difference from
    ``bg`` exceeds _BG_TOLERANCE. Max-channel (not sum) matches the semantics
    of the classic 'min(RGB) < 245' threshold on white BG, and generalizes
    cleanly to any uniform background color.
    """
    diff = np.abs(img_np.astype(np.int32) - bg).max(axis=2)
    return diff > _BG_TOLERANCE


def _resize(chw: torch.Tensor, size_hw: tuple[int, int], mode: str) -> torch.Tensor:
    return F.interpolate(
        chw.unsqueeze(0), size=size_hw, mode=mode, align_corners=False
    ).squeeze(0)


def _silhouette(fg: np.ndarray) -> dict | None:
    ys, _ = np.where(fg)
    if ys.size == 0:
        return None
    row_counts = fg.sum(axis=1).astype(np.float32)
    max_rc = float(row_counts.max())
    if max_rc <= 0:
        return None
    thresh = _ROW_DENSITY_TRIM * max_rc
    y_lo = int(ys.min())
    y_hi = int(ys.max())
    while y_lo < y_hi and row_counts[y_lo] < thresh:
        y_lo += 1
    while y_hi > y_lo and row_counts[y_hi] < thresh:
        y_hi -= 1
    body_h = y_hi - y_lo + 1
    head_y_max = y_lo + max(1, int(round(body_h * _HEAD_TOP_FRACTION)))
    head_y_max = min(head_y_max, y_hi)

    head_rows = fg[y_lo : head_y_max + 1]
    head_cx = _largest_run_cx(head_rows)
    if head_cx is None:
        xs_all = np.where(fg[y_lo : y_hi + 1].any(axis=0))[0]
        head_cx = (
            (float(xs_all.min()) + float(xs_all.max())) / 2.0
            if xs_all.size > 0
            else float(fg.shape[1]) / 2.0
        )

    return {
        "body_y_min": y_lo,
        "body_y_max": y_hi,
        "head_y_min": y_lo,
        "head_y_max": head_y_max,
        "head_cx": head_cx,
    }


def _largest_run_cx(rows: np.ndarray) -> float | None:
    """Median of per-row longest-contiguous-run midpoints. Robust to sideways
    props (sceptres, swords) that create a separate narrow run beside the main
    body run.
    """
    if rows.size == 0:
        return None
    centers: list[float] = []
    for row in rows:
        if not row.any():
            continue
        idx = np.flatnonzero(np.diff(np.concatenate([[0], row.astype(np.int8), [0]])))
        starts = idx[0::2]
        ends = idx[1::2] - 1
        if starts.size == 0:
            continue
        lengths = ends - starts + 1
        k = int(np.argmax(lengths))
        centers.append((int(starts[k]) + int(ends[k])) / 2.0)
    if not centers:
        return None
    return float(np.median(centers))


class FaceAlignedCenterNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "canvas_size": ("INT", {
                    "default": 928, "min": 64, "max": 8192, "step": 8,
                    "tooltip": "Square output size (pixels).",
                }),
                "face_fill": ("FLOAT", {
                    "default": 0.12, "min": 0.02, "max": 0.9, "step": 0.01,
                    "tooltip": "Zoom knob. Target face height as a fraction of the canvas. LOWER = zoom out (smaller character, more empty space). HIGHER = zoom in. Same value across every call = same face size for every character sheet.",
                }),
                "face_y": ("FLOAT", {
                    "default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01,
                    "tooltip": "Vertical position of the face center on the canvas (0=top, 1=bottom). 0.30 puts the face in the upper-third.",
                }),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional foreground mask; overrides BG-color auto detection."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "STRING")
    RETURN_NAMES = ("images", "masks", "info")
    FUNCTION = "process"
    CATEGORY = "FaceAlignedCenter"

    @staticmethod
    def _place(
        img: torch.Tensor,
        fg: np.ndarray,
        canvas: int,
        scale: float,
        src_cx: float,
        src_cy: float,
        target_cx: float,
        target_cy: float,
        bg_color_u8: np.ndarray,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        H, W = fg.shape
        bg_tensor = torch.from_numpy(
            bg_color_u8.astype(np.float32) / 255.0
        ).to(dtype=img.dtype, device=img.device)
        out = torch.empty((canvas, canvas, 3), dtype=img.dtype, device=img.device)
        out[:, :, :] = bg_tensor
        empty = torch.zeros((canvas, canvas), dtype=img.dtype, device=img.device)

        new_H = max(1, int(round(H * scale)))
        new_W = max(1, int(round(W * scale)))

        img_chw = img.permute(2, 0, 1).contiguous()
        resized = _resize(img_chw, (new_H, new_W), "bicubic").permute(1, 2, 0).clamp(0.0, 1.0)
        mask_chw = torch.from_numpy(fg.astype(np.float32)).to(img.device).unsqueeze(0)
        resized_mask = _resize(mask_chw, (new_H, new_W), "bilinear").squeeze(0).clamp(0.0, 1.0)

        n_cx = src_cx * scale
        n_cy = src_cy * scale
        ox = int(round(target_cx - n_cx))
        oy = int(round(target_cy - n_cy))

        sx0 = max(0, -ox)
        sy0 = max(0, -oy)
        sx1 = min(new_W, canvas - ox)
        sy1 = min(new_H, canvas - oy)
        if sx1 <= sx0 or sy1 <= sy0:
            return out, empty

        dx0, dy0 = max(0, ox), max(0, oy)
        w = sx1 - sx0
        h = sy1 - sy0
        out[dy0 : dy0 + h, dx0 : dx0 + w, :] = resized[sy0:sy1, sx0:sx1, :]
        empty[dy0 : dy0 + h, dx0 : dx0 + w] = resized_mask[sy0:sy1, sx0:sx1]
        return out, empty

    def process(
        self,
        images: torch.Tensor,
        canvas_size: int,
        face_fill: float,
        face_y: float,
        mask: torch.Tensor | None = None,
    ):
        N = images.shape[0]
        target_cx = canvas_size / 2.0
        target_cy = canvas_size * face_y

        # Pass 1: per-image FG detection + silhouette
        infos: list[dict[str, Any]] = []
        for i in range(N):
            img_full = images[i]
            img_t = img_full[..., :3].contiguous()
            img_np = (img_t.detach().cpu().numpy() * 255).astype(np.uint8)
            bg_u8 = _sample_bg_color(img_np)
            if mask is not None and mask.shape[0] > i:
                fg = mask[i].detach().cpu().numpy() > 0.5
            elif img_full.shape[-1] == 4:
                # RGBA input — trust alpha only if it actually carries
                # transparency. Many ComfyUI pipelines pass 4-channel tensors
                # with alpha=1.0 everywhere, in which case the color corner-
                # detection path is the right one.
                alpha = img_full[..., 3].detach().cpu().numpy()
                if alpha.min() < 0.5:
                    fg = alpha > 0.5
                else:
                    fg = _detect_fg(img_np, bg_u8)
            else:
                fg = _detect_fg(img_np, bg_u8)
            sil = _silhouette(fg)
            infos.append({"img": img_t, "fg": fg, "sil": sil, "bg": bg_u8})

        # Pass 2: one batch scale derived from median body height.
        # face_fill targets face height on canvas. Infer body target via
        # the assumed face/body ratio, then scale each image by the batch
        # median body_h so every angle ends up at the same size.
        body_hs = [
            i_["sil"]["body_y_max"] - i_["sil"]["body_y_min"] + 1
            for i_ in infos
            if i_["sil"] is not None
        ]
        if body_hs:
            target_body_h = min(canvas_size * 0.95, canvas_size * face_fill / _ASSUMED_FACE_TO_BODY)
            batch_scale = target_body_h / float(np.median(body_hs))
        else:
            batch_scale = 1.0
        print(f"[FaceAlignedCenter] Batch silhouette scale = {batch_scale:.3f}")

        # Pass 3: placement
        imgs_out: list[torch.Tensor] = []
        masks_out: list[torch.Tensor] = []
        info_lines: list[str] = [f"Batch scale = {batch_scale:.3f} (target body_h {canvas_size * face_fill / _ASSUMED_FACE_TO_BODY:.0f}px)"]

        for i, info in enumerate(infos):
            img_t = info["img"]
            fg = info["fg"]
            sil = info["sil"]
            bg_u8 = info["bg"]

            if sil is None:
                bg_tensor = torch.from_numpy(bg_u8.astype(np.float32) / 255.0).to(img_t.dtype)
                solid = torch.empty((canvas_size, canvas_size, 3), dtype=img_t.dtype)
                solid[:, :, :] = bg_tensor
                imgs_out.append(solid)
                masks_out.append(torch.zeros((canvas_size, canvas_size), dtype=img_t.dtype))
                info_lines.append(f"[{i}] empty silhouette — skipped")
                continue

            src_cx = sil["head_cx"]
            src_cy = (sil["head_y_min"] + sil["head_y_max"]) / 2.0

            out_img, out_mask = self._place(
                img_t, fg, canvas_size, batch_scale,
                src_cx=src_cx, src_cy=src_cy,
                target_cx=target_cx, target_cy=target_cy,
                bg_color_u8=bg_u8,
            )
            imgs_out.append(out_img)
            masks_out.append(out_mask)
            info_lines.append(f"[{i}] head_cx={src_cx:.0f} head_cy={src_cy:.0f}")

        return (
            torch.stack(imgs_out, dim=0),
            torch.stack(masks_out, dim=0),
            "\n".join(info_lines),
        )


NODE_CLASS_MAPPINGS = {"FaceAlignedCenter": FaceAlignedCenterNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceAlignedCenter": "Face-Aligned Center"}
