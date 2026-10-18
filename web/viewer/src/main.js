// @ts-check
import { PathologyApiError, PathologyClient, heatColor } from "@pathology/client";

/** @typedef {{ name: string, file: string, grid: number, labels: number[] }} DemoTile */

const serviceUrl = import.meta.env.VITE_SERVICE_URL ?? "http://localhost:8080";
// Local demo key only; a deployed viewer gets its key from the hosting environment, never from the repo.
const apiKey = import.meta.env.VITE_API_KEY ?? "local-dev-key-viewer-000001";
const client = new PathologyClient({ baseUrl: serviceUrl, apiKey, timeoutMs: 30_000 });

const $ = /** @param {string} id */ (id) => /** @type {HTMLElement} */ (document.getElementById(id));
const tileSelect = /** @type {HTMLSelectElement} */ ($("tile"));
const strideSelect = /** @type {HTMLSelectElement} */ ($("stride"));
const truthBox = /** @type {HTMLInputElement} */ ($("truth"));
const opacity = /** @type {HTMLInputElement} */ ($("opacity"));
const runButton = /** @type {HTMLButtonElement} */ ($("run"));
const tileCanvas = /** @type {HTMLCanvasElement} */ ($("tile-canvas"));
const overlay = /** @type {HTMLCanvasElement} */ ($("overlay"));

/** @type {DemoTile[]} */
let tiles = [];
/** @type {{ tile: DemoTile, res: import("@pathology/client").HeatmapResponse } | null} */
let last = null;

/** @param {Blob} blob @returns {Promise<string>} */
function toBase64(blob) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(String(reader.result).split(",", 2)[1]);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(blob);
  });
}

/** @param {string} text @param {boolean} [error] */
function setStatus(text, error = false) {
  const el = $("status");
  el.textContent = text;
  el.classList.toggle("error", error);
}

/** @param {DemoTile} tile */
async function drawTile(tile) {
  const img = new Image();
  img.src = `./demo/${tile.file}`;
  await img.decode();
  tileCanvas.width = overlay.width = img.width;
  tileCanvas.height = overlay.height = img.height;
  /** @type {CanvasRenderingContext2D} */ (tileCanvas.getContext("2d")).drawImage(img, 0, 0);
  overlay.getContext("2d")?.clearRect(0, 0, overlay.width, overlay.height);
  drawTruth(tile);
}

/** @param {DemoTile} tile */
function drawTruth(tile) {
  if (!truthBox.checked) return;
  const ctx = /** @type {CanvasRenderingContext2D} */ (overlay.getContext("2d"));
  ctx.strokeStyle = "#00e5ff";
  ctx.lineWidth = 3;
  tile.labels.forEach((label, i) => {
    if (label !== 1) return;
    const r = Math.floor(i / tile.grid);
    const c = i % tile.grid;
    ctx.strokeRect(c * 96 + 2, r * 96 + 2, 92, 92);
  });
}

function drawHeat() {
  if (!last) return;
  const { tile, res } = last;
  const ctx = /** @type {CanvasRenderingContext2D} */ (overlay.getContext("2d"));
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  ctx.globalAlpha = Number(opacity.value) / 100;
  for (let r = 0; r < res.rows; r += 1) {
    for (let c = 0; c < res.cols; c += 1) {
      const [red, green, blue, alpha] = heatColor(res.probabilities[r * res.cols + c]);
      if (alpha === 0) continue;
      ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, ${alpha / 255})`;
      // Each window covers 96 px; with overlapping strides, paint the stride-sized centre cell.
      const inset = (res.patch_size - res.stride) / 2;
      ctx.fillRect(c * res.stride + inset, r * res.stride + inset, res.stride, res.stride);
    }
  }
  ctx.globalAlpha = 1;
  drawTruth(tile);
}

/** Per-patch confusion at the service's threshold, only meaningful at stride 96. */
function summarise() {
  if (!last) return;
  const { tile, res } = last;
  const dl = $("summary");
  const rows = [
    ["Model version", res.model_version],
    ["Windows", `${res.rows} x ${res.cols} at stride ${res.stride}`],
    ["Max probability", res.max_probability.toFixed(3)],
    ["Above threshold", `${res.windows_above_threshold} (threshold ${res.threshold.toFixed(2)})`],
  ];
  if (res.stride === res.patch_size && res.probabilities.length === tile.labels.length) {
    let tp = 0; let fp = 0; let fn = 0; let tn = 0;
    res.probabilities.forEach((p, i) => {
      const predicted = p >= res.threshold;
      const actual = tile.labels[i] === 1;
      if (predicted && actual) tp += 1; else if (predicted) fp += 1; else if (actual) fn += 1; else tn += 1;
    });
    rows.push(["Agreement with labels", `${tp + tn} / ${tile.labels.length} (TP ${tp}, FP ${fp}, FN ${fn}, TN ${tn})`]);
  }
  dl.replaceChildren(...rows.flatMap(([k, v]) => {
    const dt = document.createElement("dt"); dt.textContent = k;
    const dd = document.createElement("dd"); dd.textContent = v;
    return [dt, dd];
  }));
  dl.dataset.state = "scored";
}

async function run() {
  const tile = tiles[tileSelect.selectedIndex];
  if (!tile) return;
  runButton.disabled = true;
  setStatus("Scoring...");
  try {
    const blob = await (await fetch(`./demo/${tile.file}`)).blob();
    const started = performance.now();
    const res = await client.heatmap(await toBase64(blob), { stride: Number(strideSelect.value) });
    last = { tile, res };
    drawHeat();
    summarise();
    $("summary").insertAdjacentHTML("beforeend",
      `<dt>Round trip</dt><dd>${Math.round(performance.now() - started)} ms</dd>`);
  } catch (err) {
    const message = err instanceof PathologyApiError ? `${err.code}: ${err.message}` : String(err);
    setStatus(`Failed: ${message}`, true);
  } finally {
    runButton.disabled = false;
  }
}

async function init() {
  tiles = /** @type {DemoTile[]} */ (await (await fetch("./demo/tiles.json")).json());
  tileSelect.replaceChildren(...tiles.map((t) => new Option(t.name, t.file)));
  await drawTile(tiles[0]);
  tileSelect.addEventListener("change", async () => { last = null; await drawTile(tiles[tileSelect.selectedIndex]); });
  truthBox.addEventListener("change", () => (last ? drawHeat() : drawTile(tiles[tileSelect.selectedIndex])));
  opacity.addEventListener("input", drawHeat);
  runButton.addEventListener("click", run);
}

init().catch((err) => setStatus(`Could not load demo tiles: ${err}`, true));
