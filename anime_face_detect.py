"""
Anime Face Detect & Crop — detect an anime/chibi face and return a
cropped face image plus a mask over the original image dimensions.

Uses YOLOv8 face detection (via ultralytics) as the primary detector —
the same model family ComfyUI-Yolo-Cropper uses, which handles chibi
art far more reliably than Haar cascades. Falls back to lbpcascade_
animeface if ultralytics isn't installed.

Models are auto-downloaded to ``ComfyUI/models/face_detection/`` on
first use.
"""

from __future__ import annotations

import os
import urllib.request

import cv2
import numpy as np
import torch
import torch.nn.functional as F

import folder_paths


_YOLO_URL = (
    "https://huggingface.co/spaces/cc1234/stashface/resolve/main/"
    ".deepface/weights/yolov8n-face.pt"
)
_YOLO_FILENAME = "yolov8n-face.pt"

_HAAR_URL = (
    "https://raw.githubusercontent.com/nagadomi/lbpcascade_animeface/"
    "master/lbpcascade_animeface.xml"
)
_HAAR_FILENAME = "lbpcascade_animeface.xml"


def _download_model(url: str, filename: str) -> str:
    target_dir = os.path.join(folder_paths.models_dir, "face_detection")
    os.makedirs(target_dir, exist_ok=True)
    path = os.path.join(target_dir, filename)
    if not os.path.exists(path):
        print(f"[AnimeFaceDetect] downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def _get_yolo_model():
    """Return a cached YOLO model, or None if ultralytics isn't available."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return None
    path = _download_model(_YOLO_URL, _YOLO_FILENAME)
    return YOLO(path)


def _detect_face_yolo(img_u8: np.ndarray, model):
    """Return largest (x, y, w, h) bbox from YOLO, or None."""
    results = model(img_u8, verbose=False)
    if not results:
        return None
    boxes = results[0].boxes
    if boxes is None or len(boxes) == 0:
        return None
    xyxy = boxes.xyxy.detach().cpu().numpy()
    areas = (xyxy[:, 2] - xyxy[:, 0]) * (xyxy[:, 3] - xyxy[:, 1])
    i = int(areas.argmax())
    x1, y1, x2, y2 = xyxy[i]
    return int(x1), int(y1), int(x2 - x1), int(y2 - y1)


def _haar_path() -> str:
    return _download_model(_HAAR_URL, _HAAR_FILENAME)


def _detect_faces_once(gray: np.ndarray, cascade: cv2.CascadeClassifier):
    """Run progressively looser passes on one gray image, return bbox tuple
    (x, y, w, h) or None."""
    for params in (
        dict(scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)),
        dict(scaleFactor=1.05, minNeighbors=3, minSize=(24, 24)),
        dict(scaleFactor=1.03, minNeighbors=2, minSize=(16, 16)),
    ):
        faces = cascade.detectMultiScale(gray, **params)
        if len(faces) > 0:
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            return int(x), int(y), int(w), int(h)
    return None


def _rotate_bbox_to_original(
    bbox: tuple[int, int, int, int],
    angle_deg: float,
    rotated_shape: tuple[int, int],
    original_shape: tuple[int, int],
):
    """Map a bbox detected on a rotated image back to original coords.

    Returns an axis-aligned bbox (x, y, w, h) that encloses the rotated
    original bbox — slightly larger than the tight box but good enough
    for masking/cropping.
    """
    x, y, w, h = bbox
    rH, rW = rotated_shape
    oH, oW = original_shape
    # Corners of the bbox in rotated-image coords, then transform back.
    corners = np.array([
        [x, y], [x + w, y], [x, y + h], [x + w, y + h],
    ], dtype=np.float32)
    M = cv2.getRotationMatrix2D((rW / 2, rH / 2), -angle_deg, 1.0)
    # Translate the rotated image back to original frame center.
    M[0, 2] += (oW / 2 - rW / 2)
    M[1, 2] += (oH / 2 - rH / 2)
    pts = cv2.transform(corners.reshape(-1, 1, 2), M).reshape(-1, 2)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    x0 = max(0, int(round(x0)))
    y0 = max(0, int(round(y0)))
    x1 = min(oW, int(round(x1)))
    y1 = min(oH, int(round(y1)))
    return (x0, y0, max(1, x1 - x0), max(1, y1 - y0))


def _detect_face(img_u8: np.ndarray, cascade: cv2.CascadeClassifier):
    """Return (x, y, w, h) of the largest detection, or None.

    Tries the upright image first, then ±15° and ±30° rotations since
    Haar cascades are rotation-sensitive and miss a lot of chibi frames
    where the head is tilted forward.
    """
    H, W = img_u8.shape[:2]
    gray = cv2.cvtColor(img_u8, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)

    bbox = _detect_faces_once(gray, cascade)
    if bbox is not None:
        return bbox

    for angle in (15, -15, 30, -30):
        M = cv2.getRotationMatrix2D((W / 2, H / 2), angle, 1.0)
        # Expand canvas to fit the rotated image without cropping.
        cos_a = abs(M[0, 0])
        sin_a = abs(M[0, 1])
        new_W = int(H * sin_a + W * cos_a)
        new_H = int(H * cos_a + W * sin_a)
        M[0, 2] += new_W / 2 - W / 2
        M[1, 2] += new_H / 2 - H / 2
        rotated = cv2.warpAffine(gray, M, (new_W, new_H), borderValue=255)
        bbox_r = _detect_faces_once(rotated, cascade)
        if bbox_r is not None:
            return _rotate_bbox_to_original(
                bbox_r, angle, (new_H, new_W), (H, W)
            )
    return None


def _resize(chw: torch.Tensor, size_hw: tuple[int, int]) -> torch.Tensor:
    return F.interpolate(
        chw.unsqueeze(0), size=size_hw, mode="bicubic", align_corners=False
    ).squeeze(0).clamp(0.0, 1.0)


class AnimeFaceDetectNode:
    _cascade: cv2.CascadeClassifier | None = None
    _yolo = None
    _yolo_init_tried = False

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
            cls._cascade = cv2.CascadeClassifier(_haar_path())
        return cls._cascade

    @classmethod
    def _get_yolo(cls):
        if not cls._yolo_init_tried:
            cls._yolo_init_tried = True
            cls._yolo = _get_yolo_model()
            if cls._yolo is None:
                print("[AnimeFaceDetect] ultralytics unavailable — falling back to lbpcascade")
        return cls._yolo

    def _detect(self, img_u8: np.ndarray):
        yolo = self._get_yolo()
        if yolo is not None:
            bbox = _detect_face_yolo(img_u8, yolo)
            if bbox is not None:
                return bbox
        return _detect_face(img_u8, self._get_cascade())

    def process(self, images: torch.Tensor, crop_size: int, padding: float):
        cropped_out: list[torch.Tensor] = []
        masks_out: list[torch.Tensor] = []
        # Warm the detector once per batch so the fallback log prints early.
        self._get_yolo()

        for i in range(images.shape[0]):
            img_t = images[i, ..., :3]
            H, W = img_t.shape[:2]
            img_u8 = (img_t.detach().cpu().numpy() * 255).astype(np.uint8)

            mask = torch.zeros((H, W), dtype=img_t.dtype, device=img_t.device)
            bbox = self._detect(img_u8)

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
