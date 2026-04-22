import { app } from "../../scripts/app.js";

console.log("[FaceAlignedFineTune] extension loading");

const ZOOM_STEP = 0.05;
const NUDGE_STEP = 20;

let _gridOverlay = null;
const loadGridOverlay = () => {
    if (_gridOverlay) return _gridOverlay;
    _gridOverlay = new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = reject;
        img.src = "/extensions/comfyui-face-aligned-center/grid.png";
    });
    return _gridOverlay;
};

const hideWidget = (w) => {
    w.type = "hidden";
    w.computeSize = () => [0, -4];
    w.draw = function () {};
};

const BTN_BG = "rgba(30, 30, 30, 0.82)";
const BTN_BG_HOVER = "rgba(60, 60, 60, 0.92)";
const BTN_BORDER = "rgba(120, 120, 120, 0.5)";

const GRID_SVG =
    '<svg width="14" height="14" viewBox="0 0 14 14" fill="none" ' +
    'stroke="currentColor" stroke-width="1.4" stroke-linecap="round">' +
    '<rect x="1.5" y="1.5" width="11" height="11" rx="1"/>' +
    '<line x1="5.3" y1="1.5" x2="5.3" y2="12.5"/>' +
    '<line x1="8.7" y1="1.5" x2="8.7" y2="12.5"/>' +
    '<line x1="1.5" y1="5.3" x2="12.5" y2="5.3"/>' +
    '<line x1="1.5" y1="8.7" x2="12.5" y2="8.7"/>' +
    '</svg>';

const mkBtn = (label, onclick, title = "", isHTML = false) => {
    const b = document.createElement("button");
    if (isHTML) b.innerHTML = label;
    else b.textContent = label;
    b.title = title;
    b.style.cssText =
        `background:${BTN_BG};color:#eee;border:1px solid ${BTN_BORDER};` +
        "border-radius:4px;width:26px;height:26px;padding:0;cursor:pointer;" +
        "font-size:13px;line-height:1;user-select:none;display:flex;" +
        "align-items:center;justify-content:center;" +
        "backdrop-filter:blur(4px);-webkit-backdrop-filter:blur(4px);";
    b.onclick = (e) => { e.preventDefault(); e.stopPropagation(); onclick(); };
    b.onmouseenter = () => { b.style.background = BTN_BG_HOVER; };
    b.onmouseleave = () => {
        b.style.background = b.dataset.active === "1" ? "#1f6feb" : BTN_BG;
    };
    return b;
};

const drawGrid = (ctx, W, H, overlayImg) => {
    if (!overlayImg) return;
    ctx.save();
    // multiply: white → no change, dark lines/colors → visible on top.
    ctx.globalCompositeOperation = "multiply";
    ctx.drawImage(overlayImg, 0, 0, W, H);
    ctx.restore();
};

const sampleBG = (img) => {
    const W = img.naturalWidth;
    const H = img.naturalHeight;
    const c = document.createElement("canvas");
    c.width = W;
    c.height = H;
    const cx = c.getContext("2d");
    cx.drawImage(img, 0, 0);
    const chans = [
        cx.getImageData(0, 0, 1, 1).data,
        cx.getImageData(W - 1, 0, 1, 1).data,
        cx.getImageData(0, H - 1, 1, 1).data,
        cx.getImageData(W - 1, H - 1, 1, 1).data,
    ];
    const median = (vs) => {
        const s = [...vs].sort((a, b) => a - b);
        return Math.round((s[1] + s[2]) / 2);
    };
    const r = median(chans.map((p) => p[0]));
    const g = median(chans.map((p) => p[1]));
    const b = median(chans.map((p) => p[2]));
    return `rgb(${r},${g},${b})`;
};

app.registerExtension({
    name: "comfyui-face-aligned-center.FineTune",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "FaceAlignedFineTune") return;

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);

            const zoomW = this.widgets.find((w) => w.name === "zoom");
            const dxW = this.widgets.find((w) => w.name === "dx");
            const dyW = this.widgets.find((w) => w.name === "dy");
            if (!zoomW || !dxW || !dyW) {
                console.warn("[FaceAlignedFineTune] widgets missing");
                return;
            }
            hideWidget(zoomW);
            hideWidget(dxW);
            hideWidget(dyW);

            const state = {
                rawImgs: [], idx: 0, bg: "white", showGrid: true,
                overlay: null,
            };
            loadGridOverlay().then((img) => {
                state.overlay = img;
                render();
            }).catch((e) => {
                console.warn("[FaceAlignedFineTune] grid overlay failed:", e);
            });

            const wrapper = document.createElement("div");
            wrapper.style.cssText =
                "position:relative;width:100%;max-width:600px;" +
                "margin:0 auto;aspect-ratio:1/1;";

            const canvas = document.createElement("canvas");
            canvas.width = 1024;
            canvas.height = 1024;
            canvas.style.cssText =
                "display:block;width:100%;height:100%;" +
                "background:#1a1a1a;border-radius:3px;image-rendering:auto;";
            const ctx = canvas.getContext("2d");

            const render = () => {
                const W = canvas.width;
                const H = canvas.height;
                ctx.clearRect(0, 0, W, H);

                if (!state.rawImgs.length) {
                    ctx.fillStyle = "#2a2a2a";
                    ctx.fillRect(0, 0, W, H);
                    ctx.fillStyle = "#888";
                    ctx.font = "13px sans-serif";
                    ctx.textAlign = "center";
                    ctx.fillText("Queue workflow to load preview", W / 2, H / 2);
                    return;
                }

                const img = state.rawImgs[state.idx] || state.rawImgs[0];
                const imgW = img.naturalWidth;
                const imgH = img.naturalHeight;
                const baseScale = Math.min(W / imgW, H / imgH);

                ctx.fillStyle = state.bg || "white";
                ctx.fillRect(0, 0, W, H);

                const zoom = zoomW.value;
                const totalScale = baseScale * zoom;
                const tx = dxW.value * baseScale;
                const ty = dyW.value * baseScale;

                ctx.save();
                ctx.translate(W / 2 + tx, H / 2 + ty);
                ctx.scale(totalScale, totalScale);
                ctx.imageSmoothingEnabled = true;
                ctx.imageSmoothingQuality = "high";
                ctx.drawImage(img, -imgW / 2, -imgH / 2);
                ctx.restore();

                if (state.showGrid) drawGrid(ctx, W, H, state.overlay);
            };

            const loadPreviews = async (metas) => {
                try {
                    const imgs = await Promise.all(metas.map((m) =>
                        new Promise((resolve, reject) => {
                            const img = new Image();
                            img.onload = () => resolve(img);
                            img.onerror = reject;
                            const params = new URLSearchParams({
                                filename: m.filename,
                                subfolder: m.subfolder || "",
                                type: m.type || "temp",
                            });
                            img.src = `/view?${params}&r=${Date.now()}`;
                        })
                    ));
                    state.rawImgs = imgs;
                    state.idx = Math.min(state.idx, imgs.length - 1);
                    if (imgs.length) state.bg = sampleBG(imgs[0]);
                    render();
                } catch (err) {
                    console.error("[FaceAlignedFineTune] preview load failed:", err);
                }
            };

            const origOnExecuted = this.onExecuted;
            this.onExecuted = function (output) {
                origOnExecuted?.apply(this, arguments);
                if (output?.fa_new?.[0]) {
                    // Fresh input batch — drop stale adjustments so the
                    // canvas matches the clean output downstream.
                    zoomW.value = 1.0;
                    dxW.value = 0;
                    dyW.value = 0;
                    zoomW.callback?.(1.0);
                    dxW.callback?.(0);
                    dyW.callback?.(0);
                }
                if (output?.fa_raw?.length) loadPreviews(output.fa_raw);
            };

            const bump = (w, delta, min, max) => {
                const v = Math.min(max, Math.max(min, +(w.value + delta).toFixed(3)));
                w.value = v;
                w.callback?.(v);
                render();
            };

            const reset = () => {
                zoomW.value = 1.0;
                dxW.value = 0;
                dyW.value = 0;
                zoomW.callback?.(1.0);
                dxW.callback?.(0);
                dyW.callback?.(0);
                render();
            };

            const styleGridBtn = (btn) => {
                const on = state.showGrid;
                btn.dataset.active = on ? "1" : "0";
                btn.style.background = on ? "#1f6feb" : BTN_BG;
                btn.style.borderColor = on ? "#1f6feb" : BTN_BORDER;
            };

            const gridBtn = mkBtn(GRID_SVG, () => {
                state.showGrid = !state.showGrid;
                styleGridBtn(gridBtn);
                render();
            }, "Toggle grid overlay (view-only)", true);
            styleGridBtn(gridBtn);

            // Zoom group: bottom-left, vertical stack (+ above −).
            const zoomGroup = document.createElement("div");
            zoomGroup.style.cssText =
                "position:absolute;bottom:8px;left:8px;display:flex;" +
                "flex-direction:column;gap:3px;z-index:10;";
            zoomGroup.append(
                mkBtn("+", () => bump(zoomW, ZOOM_STEP, 0.1, 5), "Zoom in"),
                mkBtn("−", () => bump(zoomW, -ZOOM_STEP, 0.1, 5), "Zoom out"),
            );

            // D-pad: bottom-center, cross layout.
            const dpad = document.createElement("div");
            dpad.style.cssText =
                "position:absolute;bottom:8px;left:50%;transform:translateX(-50%);" +
                "display:grid;grid-template-columns:repeat(3, 26px);gap:3px;z-index:10;";
            const blank = () => {
                const d = document.createElement("div");
                d.style.cssText = "width:26px;height:26px;";
                return d;
            };
            dpad.append(
                blank(), mkBtn("↑", () => bump(dyW, -NUDGE_STEP, -4096, 4096), "Move up"), blank(),
                mkBtn("←", () => bump(dxW, -NUDGE_STEP, -4096, 4096), "Move left"),
                mkBtn("↓", () => bump(dyW, NUDGE_STEP, -4096, 4096), "Move down"),
                mkBtn("→", () => bump(dxW, NUDGE_STEP, -4096, 4096), "Move right"),
            );

            // Actions: bottom-right, vertical stack (reset above grid).
            const actions = document.createElement("div");
            actions.style.cssText =
                "position:absolute;bottom:8px;right:8px;display:flex;" +
                "flex-direction:column;gap:3px;z-index:10;";
            actions.append(mkBtn("⟲", reset, "Reset adjustments"), gridBtn);

            wrapper.append(canvas, zoomGroup, dpad, actions);
            this.addDOMWidget("preview", "div", wrapper, { serialize: false });

            render();
        };
    },
});
