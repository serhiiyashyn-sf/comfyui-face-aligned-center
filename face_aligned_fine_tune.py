"""
Face-Aligned Fine-Tune — zoom + nudge with in-node preview.

Preview-style node: displays the adjusted output in the node body and
passes it through to the IMAGE output. Numeric widgets hold the state
(zoom / dx / dy); a custom JS widget hides them behind action buttons
that auto-queue on click. A clean antialiased grid overlay (crosshair,
diagonals, safe-box) can be toggled as a view-only alignment aid.
"""

from __future__ import annotations

import hashlib
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw

import folder_paths


_BG_CORNER_FRAC = 0.03
_LAST_INPUT_HASH: dict[str, str] = {}  # per-node input-hash memo


def _sample_bg(img: torch.Tensor) -> torch.Tensor:
    H, W = img.shape[:2]
    pad = max(2, int(min(H, W) * _BG_CORNER_FRAC))
    corners = torch.cat([
        img[:pad, :pad].reshape(-1, 3),
        img[:pad, -pad:].reshape(-1, 3),
        img[-pad:, :pad].reshape(-1, 3),
        img[-pad:, -pad:].reshape(-1, 3),
    ], dim=0)
    return corners.median(dim=0).values


def _resize(chw: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        chw.unsqueeze(0), size=size_hw, mode="bicubic", align_corners=False
    ).squeeze(0).clamp(0.0, 1.0)


def _apply_zoom(img: torch.Tensor, zoom: float, bg: torch.Tensor) -> torch.Tensor:
    if zoom == 1.0:
        return img
    H, W = img.shape[:2]
    new_H = max(1, int(round(H * zoom)))
    new_W = max(1, int(round(W * zoom)))

    chw = img.permute(2, 0, 1).contiguous()
    resized = _resize(chw, (new_H, new_W)).permute(1, 2, 0)

    if zoom >= 1.0:
        y0 = (new_H - H) // 2
        x0 = (new_W - W) // 2
        return resized[y0 : y0 + H, x0 : x0 + W].contiguous()

    out = torch.empty_like(img)
    out[:, :, :] = bg
    y0 = (H - new_H) // 2
    x0 = (W - new_W) // 2
    out[y0 : y0 + new_H, x0 : x0 + new_W] = resized
    return out


def _apply_translate(img: torch.Tensor, dx: int, dy: int, bg: torch.Tensor) -> torch.Tensor:
    if dx == 0 and dy == 0:
        return img
    H, W = img.shape[:2]
    out = torch.empty_like(img)
    out[:, :, :] = bg
    sy0 = max(0, -dy)
    sy1 = min(H, H - dy)
    sx0 = max(0, -dx)
    sx1 = min(W, W - dx)
    dy0 = max(0, dy)
    dx0 = max(0, dx)
    h = sy1 - sy0
    w = sx1 - sx0
    if h > 0 and w > 0:
        out[dy0 : dy0 + h, dx0 : dx0 + w] = img[sy0 : sy0 + h, sx0 : sx0 + w]
    return out


def _make_grid_overlay(size: int, safe_frac: float) -> Image.Image:
    """Antialiased RGBA overlay: crosshair + diagonals + inner safe-box."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    dark = (20, 20, 20, 170)
    blue = (77, 200, 255, 255)
    line_w = max(1, size // 700)
    box_w = max(2, size // 350)

    cx = size // 2
    cy = size // 2
    draw.line([(0, cy), (size - 1, cy)], fill=dark, width=line_w)
    draw.line([(cx, 0), (cx, size - 1)], fill=dark, width=line_w)
    draw.line([(0, 0), (size - 1, size - 1)], fill=dark, width=line_w)
    draw.line([(size - 1, 0), (0, size - 1)], fill=dark, width=line_w)

    half = int(size * safe_frac / 2)
    draw.rectangle(
        [cx - half, cy - half, cx + half, cy + half],
        outline=blue,
        width=box_w,
    )
    return img


def _composite_grid(img: torch.Tensor, safe_frac: float) -> torch.Tensor:
    H, W = img.shape[:2]
    size = min(H, W)
    overlay = _make_grid_overlay(size, safe_frac)
    if (H, W) != (size, size):
        overlay = overlay.resize((W, H), Image.LANCZOS)
    ov = np.asarray(overlay).astype(np.float32) / 255.0  # H,W,4
    alpha = torch.from_numpy(ov[..., 3:4]).to(img.dtype).to(img.device)
    rgb = torch.from_numpy(ov[..., :3]).to(img.dtype).to(img.device)
    return img * (1 - alpha) + rgb * alpha


class FaceAlignedFineTuneNode:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "zoom": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 5.0, "step": 0.02,
                }),
                "dx": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 10,
                }),
                "dy": ("INT", {
                    "default": 0, "min": -4096, "max": 4096, "step": 10,
                }),
            },
            "hidden": {"node_id": "UNIQUE_ID"},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    OUTPUT_NODE = True
    FUNCTION = "process"
    CATEGORY = "FaceAlignedCenter"

    def process(self, images, zoom, dx, dy, node_id):
        # Detect a fresh input by hashing the first image. On change,
        # stale adjustments are dropped and the JS widget is told via
        # `fa_new` to reset its state too.
        h = hashlib.md5(
            images[0].detach().cpu().numpy().tobytes()
        ).hexdigest()
        is_new = _LAST_INPUT_HASH.get(str(node_id)) != h
        _LAST_INPUT_HASH[str(node_id)] = h

        if is_new:
            zoom, dx, dy = 1.0, 0, 0

        out_imgs: list[torch.Tensor] = []
        for i in range(images.shape[0]):
            img = images[i]
            bg = _sample_bg(img)
            img = _apply_zoom(img, zoom, bg)
            img = _apply_translate(img, dx, dy, bg)
            out_imgs.append(img)

        result = torch.stack(out_imgs, dim=0)

        # Save raw inputs for the JS canvas. Include node_id in the
        # filename so multiple Fine-Tune nodes in the same workflow
        # don't collide on the same path (previously all nodes wrote
        # to raw_0000_<pid>.png and stomped each other, making every
        # node display the same image).
        subfolder = "face_aligned_fine_tune"
        preview_dir = os.path.join(folder_paths.get_temp_directory(), subfolder)
        os.makedirs(preview_dir, exist_ok=True)

        raw_previews: list[dict] = []
        for i in range(images.shape[0]):
            arr = (images[i].detach().cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr)
            filename = f"raw_{node_id}_{i:04d}_{os.getpid()}.png"
            pil.save(os.path.join(preview_dir, filename), compress_level=1)
            raw_previews.append({
                "filename": filename,
                "subfolder": subfolder,
                "type": "temp",
            })

        return {
            "ui": {"fa_raw": raw_previews, "fa_new": [is_new]},
            "result": (result,),
        }


NODE_CLASS_MAPPINGS = {"FaceAlignedFineTune": FaceAlignedFineTuneNode}
NODE_DISPLAY_NAME_MAPPINGS = {"FaceAlignedFineTune": "Face-Aligned Fine-Tune"}
