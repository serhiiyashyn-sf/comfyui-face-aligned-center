"""
Anime Face Detect & Crop — detect an anime/chibi face using
lbpcascade_animeface (OpenCV Haar cascade) and return a cropped
face image plus a mask over the original image dimensions.

The cascade file is tiny (~2MB) and CPU-only. It is downloaded once
to the ComfyUI models directory on first use.
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import folder_paths


_MODEL_URL = (
    "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/"
    "master/lbpcascade_animeface.xml"
)
_MODEL_FILENAME = "lbpcascade_animeface.xml"


def _model_path() -> str:
    target_dir = os.path.join(folder_paths.models_dir, "face_detection")
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, _MODEL_FILENAME)
    if not os.path.exists(path):
        print(f"[AnimeFaceDetect] downloading {_MODEL_URL} -> {path}")
        urllib.request.urlretrieve(_MODEL_URL, path)
    return path


def _detect_face(img_u8: np.ndarray, cascade: cv2.CascadeClassifier):
    """Return (x, y, w, h) of the largest detection, or None.

    Tries progressively looser params — lbpcascade is strict by default
    and misses a lot of chibi frames. Each pass halts as soon as it
    finds anything.
    """
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    passes = [
        dict(scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)),
        dict(scaleFactor=1.05, minNeighbors=3, minSize=(24, 24)),
        dict(scaleFactor=1.03, minNeighbors=2, minSize=(16, 16)),
    ]
    for params in passes:
        faces = cascade.detectMultiScale(gray, **params)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x), int(y), int(w), int(h)
    return None


def _resize(chw: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        chw.unsqueeze(0), size=size_hw, mode="bicubic", align_corners=False
    ).squeeze(0).clamp(0.0, 1.0)


class AnimeFaceDetectNode:
    _cascade: cv2.CascadeClassifier | None = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "crop_size": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 8,
                    "tooltip": "Output square crop size (pixels).",
                }),
                "padding": ("FLOAT", {
                    "default": 0.3, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Extra margin around the detected bbox (fraction of bbox size).",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("cropped_images", "face_masks")
    FUNCTION = "process"
    CATEGORY = "FaceAlignedCenter"

    @classmethod
    def _get_cascade(cls) -> cv2.CascadeClassifier:
        if cls._cascade is None:
            cls._cascade = cv2.CascadeClassifier(_model_path())
        return cls._cascade

    def process(self, images: torch.Tensor, crop_size: int, padding: float):
        cascade = self._get_cascade()
        cropped_out: list[torch.Tensor] = []
        masks_out: list[torch.Tensor] = []

        for i in range(images.shape[0]):
            img_t = images[i, ..., :3]
            H, W = img_t.shape[:2]
            img_u8 = (img_t.detach().cpu().numpy() * 255).astype(np.uint8)

            mask = torch.zeros((H, W), dtype=img_t.dtype, device=img_t.device)
            bbox = _detect_face(img_u8, cascade)

            if bbox is None:
                # Fallback: return a center square crop of the original so
                # the user still sees something (not a black frame). Mask
                # stays empty so downstream can tell detection failed.
                side = min(H, W)
                cy, cx = H // 2, W // 2
                half = side // 2
                fallback = img_t[cy - half : cy + half, cx - half : cx + half, :]
                chw = fallback.permute(2, 0, 1).contiguous()
                resized = _resize(chw, (crop_size, crop_size)).permute(1, 2, 0)
                cropped_out.append(resized)
                masks_out.append(mask)
                print(f"[AnimeFaceDetect] no face detected on frame {i}; center-crop fallback")
                continue

            x, y, w, h = bbox
            mask[y : y + h, x : x + w] = 1.0

            # Expand to a padded square centered on the face.
            side = max(w, h)
            pad = int(side * padding)
            half = side // 2 + pad
            cx = x + w // 2
            cy = y + h // 2
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(W, cx + half)
            y1 = min(H, cy + half)

            cropped = img_t[y0:y1, x0:x1, :]
            ch, cw = cropped.shape[:2]
            if ch == 0 or cw == 0:
                cropped_out.append(torch.zeros(
                    (crop_size, crop_size, 3),
                    dtype=img_t.dtype, device=img_t.device,
                ))
                continue

            # Pad to square if the bbox clipped at a canvas edge.
            if ch != cw:
                side_target = max(ch, cw)
                pad_h = side_target - ch
                pad_w = side_target - cw
                chw = cropped.permute(2, 0, 1).unsqueeze(0)
                chw = F.pad(
                    chw,
                    (pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
                    mode="replicate",
                )
                cropped = chw.squeeze(0).permute(1, 2, 0)

            chw = cropped.permute(2, 0, 1).contiguous()
            resized = _resize(chw, (crop_size, crop_size)).permute(1, 2, 0)
            cropped_out.append(resized)
            masks_out.append(mask)

        return (
            torch.stack(cropped_out, dim=0),
            torch.stack(masks_out, dim=0),
        )


NODE_CLASS_MAPPINGS = {"AnimeFaceDetect": AnimeFaceDetectNode}
NODE_DISPLAY_NAME_MAPPINGS = {"AnimeFaceDetect": "Anime Face Detect & Crop"}
