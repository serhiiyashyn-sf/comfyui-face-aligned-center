# ComfyUI Face-Aligned Center

Batch-aware character centering for ComfyUI. Point it at a character sheet
(typically 8 angles: front / 3-4 / side / back / etc.) and every image
comes out with the face at the **same size and the same canvas position**
— including back-view angles that don't have a visible face.

## How it works

1. **Auto-detect BG color** from the image border (works for white, green,
   black, any uniform fill), then build a foreground mask. Max-channel
   distance threshold — same behavior as the classic `min(RGB) < 245` rule
   on white BG, generalized to any color.
2. **Silhouette analysis** per image: trim thin weapon-tip rows off the
   top and bottom, then find the head x-center using per-row
   longest-contiguous-run midpoints (robust against sceptres/swords that
   extend sideways).
3. **One batch scale** for the whole batch: `target_body_h / median(body_h)`.
   Every image uses the same scale so all angles end up at the same
   physical size — including ones with extended props that made their own
   silhouette bigger.
4. **Translate** so the silhouette head-cx lands at the user-specified
   canvas position.
5. **Padding** on the output canvas is filled with the same BG color that
   was detected on the input, so a green-screen image comes out on a clean
   green canvas, a white-BG image stays white, etc.

No face-detection model required — the silhouette head-cx is stable even
on back views where a face detector would fail.

## Nodes

### Face-Aligned Center

Inputs:
- `images` (IMAGE batch) — the character sheet frames
- `canvas_size` (INT, default 928) — square output size in pixels
- `face_fill` (FLOAT, default 0.12) — face height as a fraction of the
  canvas. The one zoom knob. Lower = more empty space around the character,
  higher = tighter crop.
- `face_y` (FLOAT, default 0.30) — vertical position of the face center on
  the canvas (0 = top edge, 1 = bottom edge).
- `mask` (MASK, optional) — pre-computed foreground mask. Overrides the
  BG-color auto detection.

Outputs:
- `images` (IMAGE batch) — centered/scaled results, same count as input.
- `masks` (MASK batch) — the foreground mask after scaling/translation.
- `info` (STRING) — diagnostic: batch scale + per-image detected head
  coordinates.

### Face-Aligned Fine-Tune

Manual per-image adjustment on top of the auto-aligned output. Interactive
canvas preview with zoom / nudge buttons — all changes render client-side
(no workflow queue triggered per click), so iteration is instant. A
toggleable grid overlay (crosshair + diagonals + safe-box) helps verify
alignment. The overlay is view-only and never reaches the clean IMAGE
output.

Inputs:
- `images` (IMAGE batch) — typically from Face-Aligned Center.
- `zoom` / `dx` / `dy` — driven by the `+ −` and `← ↑ ↓ →` buttons.
  Auto-reset when a new batch of images flows in, so a pipeline running
  many times never applies stale adjustments to a fresh character.

Outputs:
- `images` (IMAGE batch) — zoom + translate applied, no grid baked in.

## Installation

Via ComfyUI Manager, or manually:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/serhiiyashyn-sf/comfyui-face-aligned-center
cd comfyui-face-aligned-center
pip install -r requirements.txt
```

## License

MIT
