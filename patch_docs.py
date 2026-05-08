#!/usr/bin/env python3
"""
Ports the parallel coordinates panel from templates/index.html into docs/index.html.
Run as the repo owner (kajayaku) since docs/ is owner-write only.

    python patch_docs.py
"""
from pathlib import Path

DOCS = Path(__file__).parent / "docs" / "index.html"
html = DOCS.read_text()

# ── 1. Guard: skip if already patched ────────────────────────────────────────
if "pc-panel" in html:
    print("Already patched — nothing to do.")
    raise SystemExit(0)

# ── 2. CSS ────────────────────────────────────────────────────────────────────
CSS = """
        /* ── Parallel Coordinates Panel ─── */
        #pc-panel {
            border-bottom: 1px solid var(--border);
            padding: 16px 28px 20px;
        }

        #pc-panel .view-tag-pc {
            background: rgba(34, 197, 94, .12);
            color: var(--green);
        }

        #pc-canvas-wrap {
            position: relative;
            width: 100%;
            overflow-x: auto;
        }

        #pc-canvas {
            display: block;
            background: var(--bg-card);
            border-radius: var(--radius);
            border: 1px solid var(--border);
        }

        .pc-token-legend {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 10px;
        }

        .pc-legend-item {
            display: flex;
            align-items: center;
            gap: 5px;
            font-family: var(--mono);
            font-size: 10px;
            color: var(--text-dim);
        }

        .pc-legend-swatch {
            width: 18px;
            height: 3px;
            border-radius: 2px;
        }
"""

CSS_ANCHOR = "        .bwd-strip .fcard-canvas {\n            cursor: crosshair;\n        }"
assert CSS_ANCHOR in html, "CSS anchor not found"
html = html.replace(CSS_ANCHOR, CSS_ANCHOR + "\n" + CSS, 1)

# ── 3. HTML panel ─────────────────────────────────────────────────────────────
PC_PANEL = """        <!-- Parallel Coordinates panel -->
        <div class="view-panel hidden" id="pc-panel">
            <div class="view-header">
                <span class="view-tag view-tag-pc">PARALLEL COORDS</span>
                <span class="view-desc">Peak-attention location per frame — line thickness = |Δ attention|, colour = change direction</span>
                <div class="view-actions" style="margin-left:auto">
                    <div class="btn-group-sm">
                        <button id="pc-tok-toggle" class="active">Per token</button>
                        <button id="pc-agg-toggle">Aggregate</button>
                    </div>
                </div>
            </div>
            <div id="pc-canvas-wrap"><canvas id="pc-canvas"></canvas></div>
            <div class="pc-token-legend" id="pc-legend"></div>
        </div>
"""

HTML_ANCHOR = '        <div class="view-panel" id="bwd-panel">'
assert HTML_ANCHOR in html, "HTML anchor not found"
html = html.replace(HTML_ANCHOR, PC_PANEL + HTML_ANCHOR, 1)

# ── 4. JavaScript ─────────────────────────────────────────────────────────────
PC_JS = """
        // ═══ Parallel Coordinates ═══
        const PC_THUMB_W = 120, PC_THUMB_H = 68, PC_MARGIN = 48, PC_PAD_TOP = 28, PC_PAD_BOT = 18;
        let pcMode = "per-token";
        let pcPerTokenHeatmaps = null;

        function pcTokenColor(idx) {
            const hues = [210, 40, 140, 0, 280, 170, 60, 320, 100, 200, 350, 80];
            return `hsl(${hues[idx % hues.length]},85%,62%)`;
        }

        function pcDeltaColor(delta, maxDelta) {
            if (maxDelta === 0) return { r: 180, g: 180, b: 180 };
            const t = Math.max(0, Math.min(1, (delta / maxDelta + 1) / 2));
            return {
                r: Math.round(255 + t * (30 - 255)),
                g: Math.round(255 + t * (60 - 255)),
                b: Math.round(255 + t * (180 - 255)),
            };
        }

        function pcDeltaThickness(absDelta, maxAbsDelta) {
            if (maxAbsDelta === 0) return 1;
            return 1 + (absDelta / maxAbsDelta) * 5;
        }

        function peakPatch(hm) {
            let best = -1, br = 0, bc = 0;
            for (let r = 0; r < hm.length; r++)
                for (let c = 0; c < hm[r].length; c++)
                    if (hm[r][c] > best) { best = hm[r][c]; br = r; bc = c; }
            return { row: br, col: bc, val: best };
        }

        function patchDotPos(row, col, thumbX, thumbY) {
            const pw = PC_THUMB_W / M.w_patches, ph = PC_THUMB_H / M.h_patches;
            return { x: thumbX + (col + 0.5) * pw, y: thumbY + (row + 0.5) * ph };
        }

        function renderPC() {
            const panel = document.getElementById("pc-panel");
            if (!fwdHeatmaps || fwdSel.size === 0) { panel.classList.add("hidden"); return; }
            panel.classList.remove("hidden");

            const nFrames = M.num_frames;
            const canvasW = Math.max(600, nFrames * (PC_THUMB_W + PC_MARGIN) + PC_MARGIN);
            const canvasH = PC_THUMB_H + PC_PAD_TOP + PC_PAD_BOT;
            const cvs = document.getElementById("pc-canvas");
            cvs.width = canvasW; cvs.height = canvasH;
            const ctx = cvs.getContext("2d");
            ctx.clearRect(0, 0, canvasW, canvasH);
            ctx.fillStyle = "#161921"; ctx.fillRect(0, 0, canvasW, canvasH);

            const thumbXs = Array.from({ length: nFrames }, (_, i) =>
                PC_MARGIN / 2 + i * (PC_THUMB_W + PC_MARGIN));
            const thumbY = PC_PAD_TOP;

            for (let fi = 0; fi < nFrames; fi++) {
                const img = fwdImgs[fi], tx = thumbXs[fi];
                if (img && img.complete && img.naturalWidth > 0) {
                    ctx.drawImage(img, tx, thumbY, PC_THUMB_W, PC_THUMB_H);
                    ctx.fillStyle = "rgba(0,0,0,0.35)";
                    ctx.fillRect(tx, thumbY, PC_THUMB_W, PC_THUMB_H);
                } else {
                    ctx.fillStyle = "#0a0b10";
                    ctx.fillRect(tx, thumbY, PC_THUMB_W, PC_THUMB_H);
                }
                ctx.strokeStyle = "#252936"; ctx.lineWidth = 1;
                ctx.strokeRect(tx + 0.5, thumbY + 0.5, PC_THUMB_W - 1, PC_THUMB_H - 1);
                ctx.fillStyle = "#5c6272"; ctx.font = "9px 'JetBrains Mono', monospace";
                ctx.textAlign = "center";
                ctx.fillText(`F${fi}`, tx + PC_THUMB_W / 2, thumbY - 6);
            }

            const selTokens = Array.from(fwdSel).sort((a, b) => a - b);
            const layer = +document.getElementById("layer-sl").value;
            const trajectories = [];

            if (pcMode === "per-token") {
                selTokens.forEach((ti, idx) => {
                    const hms = (pcPerTokenHeatmaps && pcPerTokenHeatmaps[ti])
                        ? pcPerTokenHeatmaps[ti]
                        : computeForwardHeatmaps([ti], layer, "mean");
                    const frames = [];
                    for (let fi = 0; fi < nFrames; fi++) {
                        const tg = Math.floor(fi / M.temporal_merge_factor);
                        const hm = hms[tg]; if (!hm) continue;
                        const { row, col, val } = peakPatch(hm);
                        frames.push({ fi, row, col, val });
                    }
                    trajectories.push({ tokenIdx: ti, colorIdx: idx, frames });
                });
            } else {
                const frames = [];
                for (let fi = 0; fi < nFrames; fi++) {
                    const tg = Math.floor(fi / M.temporal_merge_factor);
                    const hm = fwdHeatmaps[tg]; if (!hm) continue;
                    const { row, col, val } = peakPatch(hm);
                    frames.push({ fi, row, col, val });
                }
                trajectories.push({ tokenIdx: -1, colorIdx: 0, frames });
            }

            let globalMaxAbsDelta = 0;
            trajectories.forEach(({ frames }) => {
                for (let i = 1; i < frames.length; i++)
                    globalMaxAbsDelta = Math.max(globalMaxAbsDelta, Math.abs(frames[i].val - frames[i - 1].val));
            });

            // Lines
            trajectories.forEach(({ frames }) => {
                for (let i = 1; i < frames.length; i++) {
                    const a = frames[i - 1], b = frames[i];
                    const pa = patchDotPos(a.row, a.col, thumbXs[a.fi], thumbY);
                    const pb = patchDotPos(b.row, b.col, thumbXs[b.fi], thumbY);
                    const delta = b.val - a.val;
                    const dc = pcDeltaColor(delta, globalMaxAbsDelta);
                    const thickness = pcDeltaThickness(Math.abs(delta), globalMaxAbsDelta);
                    const alpha = pcMode === "per-token" ? 0.75 : 0.9;
                    const mx = (pa.x + pb.x) / 2;
                    ctx.beginPath();
                    ctx.moveTo(pa.x, pa.y);
                    ctx.bezierCurveTo(mx, pa.y, mx, pb.y, pb.x, pb.y);
                    ctx.strokeStyle = `rgba(${dc.r},${dc.g},${dc.b},${alpha})`;
                    ctx.lineWidth = thickness; ctx.lineCap = "round"; ctx.stroke();
                }
            });

            // Dots
            trajectories.forEach(({ frames, colorIdx }) => {
                const tokenColor = pcTokenColor(colorIdx);
                frames.forEach(({ fi, row, col }) => {
                    const { x, y } = patchDotPos(row, col, thumbXs[fi], thumbY);
                    ctx.beginPath(); ctx.arc(x, y, 5, 0, Math.PI * 2);
                    ctx.fillStyle = "rgba(0,0,0,0.6)"; ctx.fill();
                    ctx.beginPath(); ctx.arc(x, y, 3.5, 0, Math.PI * 2);
                    ctx.fillStyle = tokenColor; ctx.fill();
                });
            });

            // Legend
            const legend = document.getElementById("pc-legend"); legend.innerHTML = "";
            if (pcMode === "per-token") {
                selTokens.forEach((ti, idx) => {
                    const tok = M.generated_tokens[ti];
                    const item = document.createElement("div"); item.className = "pc-legend-item";
                    const swatch = document.createElement("div"); swatch.className = "pc-legend-swatch";
                    swatch.style.background = pcTokenColor(idx);
                    const label = document.createElement("span");
                    label.textContent = `${ti}: ${(tok || "·").replace(/ /g, "·").replace(/\\n/, "↵")}`;
                    item.appendChild(swatch); item.appendChild(label); legend.appendChild(item);
                });
            } else {
                const item = document.createElement("div"); item.className = "pc-legend-item";
                const swatch = document.createElement("div"); swatch.className = "pc-legend-swatch";
                swatch.style.background = pcTokenColor(0);
                const label = document.createElement("span"); label.textContent = "Aggregate";
                item.appendChild(swatch); item.appendChild(label); legend.appendChild(item);
            }

            const scaleItem = document.createElement("div"); scaleItem.className = "pc-legend-item";
            scaleItem.style.marginLeft = "auto";
            const sc = document.createElement("canvas"); sc.width = 80; sc.height = 6;
            const sctx = sc.getContext("2d");
            for (let x = 0; x < 80; x++) {
                const { r, g, b } = pcDeltaColor(x / 79 * 2 - 1, 1);
                sctx.fillStyle = `rgb(${r},${g},${b})`; sctx.fillRect(x, 0, 1, 6);
            }
            sc.style.cssText = "border-radius:2px;border:1px solid #252936";
            const sl = document.createElement("span"); sl.textContent = "decrease → increase";
            scaleItem.appendChild(sc); scaleItem.appendChild(sl); legend.appendChild(scaleItem);
        }

        function refreshPC() {
            if (!M || fwdSel.size === 0) { document.getElementById("pc-panel").classList.add("hidden"); return; }
            if (pcMode === "per-token" && fwdSel.size > 1) {
                const layer = +document.getElementById("layer-sl").value;
                pcPerTokenHeatmaps = {};
                fwdSel.forEach(ti => {
                    pcPerTokenHeatmaps[ti] = computeForwardHeatmaps([ti], layer, "mean");
                });
            } else {
                pcPerTokenHeatmaps = null;
            }
            renderPC();
        }

        document.getElementById("pc-tok-toggle").addEventListener("click", function () {
            pcMode = "per-token";
            this.classList.add("active"); document.getElementById("pc-agg-toggle").classList.remove("active");
            refreshPC();
        });
        document.getElementById("pc-agg-toggle").addEventListener("click", function () {
            pcMode = "aggregate";
            this.classList.add("active"); document.getElementById("pc-tok-toggle").classList.remove("active");
            refreshPC();
        });

"""

JS_ANCHOR = "        // ═══ Controls ═══"
assert JS_ANCHOR in html, "JS anchor not found"
html = html.replace(JS_ANCHOR, PC_JS + JS_ANCHOR, 1)

# ── 5. Hook refreshPC into doForward and fwd-clear ───────────────────────────
OLD_DO_FWD = "function doForward() { if (fwdSel.size === 0) { fwdHeatmaps = null; renderFwdFrames(); return; }"
NEW_DO_FWD = "function doForward() { if (fwdSel.size === 0) { fwdHeatmaps = null; renderFwdFrames(); document.getElementById('pc-panel').classList.add('hidden'); return; }"
assert OLD_DO_FWD in html, "doForward anchor not found"
html = html.replace(OLD_DO_FWD, NEW_DO_FWD, 1)

# Insert refreshPC() call at end of doForward after renderFwdFrames()
html = html.replace(
    "fwdHeatmaps = computeForwardHeatmaps(steps, layer, fwdAgg); renderFwdFrames(); }",
    "fwdHeatmaps = computeForwardHeatmaps(steps, layer, fwdAgg); renderFwdFrames(); refreshPC(); }",
    1,
)

OLD_CLEAR = "document.getElementById(\"fwd-clear\").addEventListener(\"click\", () => { fwdSel.clear(); updateFwdTokenUI(); fwdHeatmaps = null; renderFwdFrames(); });"
NEW_CLEAR = "document.getElementById(\"fwd-clear\").addEventListener(\"click\", () => { fwdSel.clear(); updateFwdTokenUI(); fwdHeatmaps = null; pcPerTokenHeatmaps = null; renderFwdFrames(); document.getElementById('pc-panel').classList.add('hidden'); });"
assert OLD_CLEAR in html, "fwd-clear anchor not found"
html = html.replace(OLD_CLEAR, NEW_CLEAR, 1)

# ── 6. Write ──────────────────────────────────────────────────────────────────
DOCS.write_text(html)
print(f"Patched {DOCS}")
