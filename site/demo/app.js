// RABBiT demo: a 3D cortex whose colours track predicted fMRI. Clips play back
// precomputed predictions; the microphone runs the model live in a worker.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { DEFAULT_PARAMS, windowSamples, colorizeFrame } from "./pipeline.mjs";

THREE.ColorManagement.enabled = false;   // render the authored sRGB colours as-is

const VMAX = 0.78, THRESH = 0.05;
const NEUTRAL = [0xcd / 255, 0xca / 255, 0xc3 / 255];

// The mic-mode weight lives on Hugging Face (too big for Pages) and is cached on
// the device after the first download. Set this to enable mic mode.
const MODEL_HF_REPO = "omermosa/rabbit";   // public HF repo holding rabbit_fp32.onnx
const HF_MODEL = MODEL_HF_REPO ? `https://huggingface.co/${MODEL_HF_REPO}/resolve/main/rabbit_fp32.onnx` : null;
const LOCAL_MODEL = "assets/rabbit_fp32.onnx";   // dev: served from the symlink by serve.mjs
// On localhost use the local file (no download); on the live site use HF and let
// the local path fall through (it just 404s there).
const onLocalhost = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname);
const MODEL_URLS = (onLocalhost ? [LOCAL_MODEL, HF_MODEL] : [HF_MODEL, LOCAL_MODEL]).filter(Boolean);
const MODEL_CACHE = "rabbit-model-v1";   // bump to invalidate the cached weight

const $ = (id) => document.getElementById(id);
const setStatus = (s) => ($("status").textContent = s);
const setProg = (f) => { const pct = Math.round(f * 100); $("prog").style.width = `${pct}%`; const b = $("barEl"); if (b) b.setAttribute("aria-valuenow", String(pct)); };
const fetchBuf = async (u) => new Uint8Array(await (await fetch(u)).arrayBuffer());
const asF32 = (b) => new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
const asI32 = (b) => new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);

const S = { rm: null, params: null, clips: null, flatToFs6: null, baseColor: null, colorAttr: null,
  preds: null, nPred: 0, clip: null, playing: false, IN: 0, curClip: null, transcripts: null, clipGen: 0,
  worker: null, ep: "", pending: null, reqId: 0, mic: null, modelWarm: null, clipCtx: null,
  whisper: null, whisperInit: null, whisperEp: "", whisperPending: null, whisperReq: 0 };

function hasWebGL() {
  try { const c = document.createElement("canvas"); return !!(window.WebGLRenderingContext && (c.getContext("webgl") || c.getContext("experimental-webgl"))); }
  catch { return false; }
}

// Mic mode pulls a 422 MB model and runs heavy WASM — fine on a desktop, an
// uncatchable OOM crash on phones. Gate it rather than let it fall over.
function micSupport() {
  const why = [];
  if (!window.isSecureContext) why.push("needs HTTPS");
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) why.push("microphone unavailable");
  const coarse = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  const mobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent || "");
  if (coarse || mobile) why.push("desktop only — the 420 MB model can't run on phones");
  const mem = navigator.deviceMemory;            // GB, where the browser reports it
  if (typeof mem === "number" && mem > 0 && mem < 8) why.push("needs ≈ 8 GB RAM");
  return { ok: why.length === 0, why };
}

// ── Three.js brain ──────────────────────────────────────────────────────────
function buildScene(coords, faces) {
  const canvas = $("brain");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xfdfdfb);
  const nVert = coords.length / 3;
  const geom = new THREE.BufferGeometry();
  geom.setAttribute("position", new THREE.BufferAttribute(coords, 3));
  geom.setIndex(new THREE.BufferAttribute(Uint32Array.from(faces), 1));
  const base = new Float32Array(nVert * 3);
  for (let v = 0; v < nVert; v++) { base[v * 3] = NEUTRAL[0]; base[v * 3 + 1] = NEUTRAL[1]; base[v * 3 + 2] = NEUTRAL[2]; }
  const colorAttr = new THREE.BufferAttribute(base.slice(), 3); colorAttr.setUsage(THREE.DynamicDrawUsage);
  geom.setAttribute("color", colorAttr);
  geom.computeVertexNormals(); geom.computeBoundingBox();
  const c = new THREE.Vector3(); geom.boundingBox.getCenter(c); geom.translate(-c.x, -c.y, -c.z);
  geom.computeBoundingSphere(); const R = geom.boundingSphere.radius;
  scene.add(new THREE.Mesh(geom, new THREE.MeshStandardMaterial({ vertexColors: true, roughness: 0.62, metalness: 0.0 })));
  scene.add(new THREE.HemisphereLight(0xffffff, 0x4a525e, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 0.7); key.position.set(0.4, -1, 0.9); scene.add(key);
  const fill = new THREE.DirectionalLight(0xffffff, 0.25); fill.position.set(-0.6, 0.4, 0.3); scene.add(fill);
  const cam = new THREE.PerspectiveCamera(38, 1, R * 0.05, R * 20); cam.up.set(0, 0, 1); cam.position.set(-R * 2.9, 0, R * 0.08);
  const controls = new OrbitControls(cam, renderer.domElement); controls.enableDamping = true;
  function resize() { const w = canvas.clientWidth, h = canvas.clientHeight; if (w && h && (canvas.width !== w || canvas.height !== h)) { renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix(); } }
  (function loop() { resize(); controls.update(); renderer.render(scene, cam); requestAnimationFrame(loop); })();
  S.colorAttr = colorAttr; S.baseColor = base;
}

function paintFrame(flat) {
  colorizeFrame(flat, S.flatToFs6, S.colorAttr.array, S.baseColor, VMAX, THRESH);
  S.colorAttr.needsUpdate = true;
}
function resetBrain() { if (!S.colorAttr) return; S.colorAttr.array.set(S.baseColor); S.colorAttr.needsUpdate = true; }

// ── Rolling caption: stitch overlapping transcript windows ──
function stitchWords(prev, next) {
  const maxK = Math.min(prev.length, next.length, 12);
  for (let k = maxK; k > 0; k--) {
    if (prev.slice(-k).join(" ").toLowerCase() === next.slice(0, k).join(" ").toLowerCase()) return prev.concat(next.slice(k));
  }
  return prev.concat(next);
}
function pushCaption(text) {
  const words = (text || "").trim().split(/\s+/).filter(Boolean);
  if (!words.length) return;
  S.capWords = stitchWords(S.capWords || [], words).slice(-44);
  const el = $("caption"); el.textContent = S.capWords.join(" "); el.classList.add("show"); el.scrollTop = el.scrollHeight;
}
function clearCaption() { S.capWords = []; const el = $("caption"); el.textContent = ""; el.classList.remove("show"); }

// Reveal a clip's precomputed transcript progressively — word by word across each
// segment's [start,end] — so the caption tracks the audio even when a clip is one
// long segment (otherwise the whole line dumps at t=0, over the intro/laughter).
function clipCaptionAt(el, segs) {
  const out = [];
  for (const s of segs) {
    const st = s.start ?? 0;
    if (el < st) break;
    const en = (s.end != null && s.end > st) ? s.end : st + 3;
    const words = (s.text || "").trim().split(/\s+/).filter(Boolean);
    const n = Math.min(words.length, Math.ceil(Math.min(1, (el - st) / (en - st)) * words.length));
    for (let i = 0; i < n; i++) out.push(words[i]);
  }
  return out.slice(-44).join(" ");
}

// Light silence mask for live mic captions (Whisper invents text on silence).
function audioRms(a) { let s = 0; for (let i = 0; i < a.length; i++) s += a[i] * a[i]; return Math.sqrt(s / (a.length || 1)); }

// ── Startup ──
function showNoWebGL() {
  const v = $("view");
  if (v) v.innerHTML = '<div style="position:absolute;inset:0;display:flex;align-items:center;justify-content:center;padding:34px;text-align:center;color:var(--ink-soft);font-size:14px;line-height:1.65">The interactive brain needs&nbsp;<strong>&nbsp;WebGL&nbsp;</strong>, which your browser or GPU isn’t providing. Try a recent desktop Chrome, Edge, Firefox, or Safari.</div>';
  setStatus("✗ WebGL unavailable — the 3D brain can’t render in this browser.");
  $("clipCard").classList.add("off"); $("micCard").classList.add("off");
}
function gateMic() {
  const ms = micSupport();
  if (ms.ok) {
    $("micCard").classList.remove("off"); $("mic").disabled = false;
    const saveData = navigator.connection && navigator.connection.saveData;
    if (MODEL_URLS.length && !saveData) warmModel();   // prefetch so the mic is ready when clicked
    return;
  }
  $("micCard").classList.add("off"); $("mic").disabled = true;
  $("mic").textContent = "● Live mic — desktop only";
  const note = $("micNote");
  if (note) note.innerHTML = "<strong>Live mic isn’t available here</strong> (" + ms.why.join("; ") + "). The clip examples above run in any modern browser.";
}
async function init() {
  $("retry").hidden = true;
  if (!hasWebGL()) return showNoWebGL();
  try {
    setStatus("loading…");
    const [coordsB, facesB, ftfB, rm, clips, transcripts] = await Promise.all([
      fetchBuf("assets/pial_coords.f32"), fetchBuf("assets/faces.i32"), fetchBuf("assets/flat_to_fs6.i32"),
      fetch("assets/render_manifest.json").then((r) => r.json()),
      fetch("assets/clips.json").then((r) => r.json()),
      fetch("assets/transcripts.json").then((r) => r.json()).catch(() => ({})),
    ]);
    S.rm = rm; S.flatToFs6 = asI32(ftfB); S.clips = clips; S.transcripts = transcripts;
    S.params = { ...DEFAULT_PARAMS }; S.IN = (S.params.hrf_delay + 1) * windowSamples(S.params);
    if (!window.__brainReady) { buildScene(asF32(coordsB), asI32(facesB)); window.__brainReady = true; }
    $("clipSel").innerHTML = clips.map((c) => `<option value="${c.id}">${c.label} · ${c.duration_s}s</option>`).join("");
    $("clipCard").classList.remove("off");
    gateMic();
    setStatus("ready — pick a clip and press Play, or start the mic.");
  } catch (e) {
    console.error(e);
    setStatus("✗ couldn’t load demo data: " + e.message);
    $("retry").hidden = false;   // let the user retry instead of leaving a dead page
  }
}

// ── Clip playback ──
async function run() {
  stopMic();
  const gen = ++S.clipGen;                          // supersede any clip already playing
  const meta = S.clips.find((c) => c.id === $("clipSel").value);
  S.curClip = meta; clearCaption();
  try {
    setStatus(`loading ${meta.label}…`); setProg(0.2);
    const [predsB, clipB] = await Promise.all([fetchBuf(`assets/preds_${meta.id}.f32`), fetchBuf("assets/" + meta.file)]);
    if (S.clipGen !== gen) return;                  // user switched clips during the load
    S.preds = asF32(predsB); S.nPred = meta.n_tr; S.clip = asF32(clipB); S.params.sample_rate = meta.sample_rate;
    setProg(1); setStatus(`${meta.label}: playing…`);
    await playAudioAnimate(gen);
    if (S.clipGen !== gen) return;                  // superseded by a newer clip → leave it alone
    resetBrain(); clearCaption();
    setStatus(`${meta.label}: done — replay or pick another.`);
  } catch (e) { if (S.clipGen === gen) { setStatus("✗ " + e.message); console.error(e); } }
}
function playAudioAnimate(gen) {
  const p = S.params, clip = S.clip, preds = S.preds, nPred = S.nPred, OUT = S.rm.flat_dim;
  const dur = clip.length / p.sample_rate;
  const segs = (S.transcripts && S.curClip) ? (S.transcripts[S.curClip.id] || []) : [];
  try {
    if (S.clipCtx) { try { S.clipCtx.close(); } catch {} }     // stop the previous clip's audio
    const ctx = new (window.AudioContext || window.webkitAudioContext)(); S.clipCtx = ctx; ctx.resume();
    const b = ctx.createBuffer(1, clip.length, p.sample_rate); b.copyToChannel(clip, 0);
    const src = ctx.createBufferSource(); src.buffer = b; src.connect(ctx.destination); src.start();
  } catch (e) { console.warn("audio:", e.message); }
  const t0 = performance.now();
  return new Promise((resolve) => {
    (function tick() {
      if (S.clipGen !== gen) return resolve();                 // switched/stopped → end this loop
      const el = (performance.now() - t0) / 1000;
      if (el > dur + 0.3) return resolve();
      const tr = Math.min(nPred - 1, Math.floor(el / p.tr_length));
      paintFrame(preds.subarray(tr * OUT, (tr + 1) * OUT));
      const cap = clipCaptionAt(el, segs);
      if (cap) { const c = $("caption"); c.textContent = cap; c.classList.add("show"); }
      requestAnimationFrame(tick);
    })();
  });
}

// ── Live mic inference (in a worker) ──

// Download the weight at low priority (so it doesn't crowd the clip/page
// fetches), aborting if the connection stalls for 25 s. onProgress(got, total).
async function streamModel(url, onProgress) {
  const ctrl = new AbortController();
  let stallId;
  const arm = () => { clearTimeout(stallId); stallId = setTimeout(() => ctrl.abort(), 25000); };
  arm();
  let resp;
  try { resp = await fetch(url, { signal: ctrl.signal, mode: "cors", priority: "low" }); }
  catch (e) { clearTimeout(stallId); throw e; }
  if (!resp.ok) { clearTimeout(stallId); throw new Error(`model HTTP ${resp.status}`); }
  const total = +(resp.headers.get("content-length") || 0);
  const reader = resp.body.getReader(); const chunks = []; let got = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read(); if (done) break;
      chunks.push(value); got += value.length; arm();
      onProgress && onProgress(got, total);
    }
  } finally { clearTimeout(stallId); }
  const buf = new Uint8Array(got); let o = 0; for (const c of chunks) { buf.set(c, o); o += c.length; } chunks.length = 0;
  return buf;
}
// Return the cached weight if we have it, else download it (one retry) and cache
// it so the device never refetches 422 MB.
async function getModelBytes(onProgress) {
  let cache = null;
  try { cache = await caches.open(MODEL_CACHE); } catch {}
  if (cache) {
    for (const u of MODEL_URLS) {
      try { const hit = await cache.match(u); if (hit && hit.ok) return new Uint8Array(await hit.arrayBuffer()); } catch {}
    }
  }
  let lastErr;
  for (const u of MODEL_URLS) {
    for (let attempt = 0; attempt < 2; attempt++) {
      try {
        const buf = await streamModel(u, onProgress);
        if (cache) { try { await cache.put(u, new Response(buf, { headers: { "content-type": "application/octet-stream" } })); } catch (e) { console.warn("model cache put:", e.message); } }
        return buf;
      } catch (e) { lastErr = e; if (e.name !== "AbortError") console.warn("model fetch:", e.message); }
    }
  }
  throw new Error(`couldn’t load the model (it may still be deploying). ${lastErr ? lastErr.message : ""}`.trim());
}

// Show model-download progress on the mic card (not the clip status line).
function setMicStatus(text, busy) { const el = $("micStatus"); if (!el) return; el.textContent = text || ""; el.classList.toggle("busy", !!busy); }
function micProg(got, total) { setMicStatus(total ? `loading model… ${(got / 1e6) | 0}/${(total / 1e6) | 0} MB` : `loading model… ${(got / 1e6) | 0} MB`, true); }

// Warm the model into the cache in the background so the mic is ready before
// it's clicked. De-duped: the click path awaits this same download.
function warmModel() {
  // Prefetch only when Cache Storage exists to hold it; otherwise the bytes would
  // be discarded here and re-downloaded on click. No cache → fetch on demand.
  if (S.worker || S.modelWarm || !("caches" in window)) return S.modelWarm;
  setMicStatus("preparing live mic…", true);
  S.modelWarm = getModelBytes(micProg)
    .then(() => { setMicStatus("live mic ready", false); })
    .catch((e) => { S.modelWarm = null; setMicStatus(""); console.warn("prefetch:", e.message); });
  return S.modelWarm;
}
async function ensureWorker() {
  if (S.worker) return;
  const w = new Worker("./rabbit_worker.mjs", { type: "module" });
  if (S.modelWarm) { try { await S.modelWarm; } catch {} }   // reuse the background prefetch
  const buf = await getModelBytes(micProg);                  // cache hit if prefetched → instant
  setMicStatus("", false);
  setStatus("initialising model in worker…");
  await new Promise((resolve, reject) => {
    w.onmessage = (e) => { if (e.data.type === "ready") { S.ep = e.data.ep; resolve(); } else if (e.data.type === "error") reject(new Error(e.data.error)); };
    w.postMessage({ type: "init", model: buf.buffer, eps: ["wasm"] }, [buf.buffer]);   // CPU path (WebGPU output is unreliable for this model)
  });
  S.pending = new Map(); S.reqId = 0;
  w.onmessage = (e) => { const m = e.data; const p = S.pending.get(m.id); if (!p) return; S.pending.delete(m.id); m.type === "result" ? p.resolve(m.preds) : p.reject(new Error(m.error)); };
  S.worker = w;
}
function workerRun(data) {
  const id = ++S.reqId;
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => { if (S.pending.delete(id)) reject(new Error("inference timeout")); }, 30000);
    const wrap = (fn) => (v) => { clearTimeout(to); fn(v); };
    S.pending.set(id, { resolve: wrap(resolve), reject: wrap(reject) });
    S.worker.postMessage({ type: "run", id, data, rows: 1, cols: S.IN }, [data.buffer]);
  });
}

// Live captions in a separate worker (display only — not fed to the model).
// Single-flight: repeat/concurrent callers await the same init.
function ensureWhisper() {
  if (S.whisperInit) return S.whisperInit;
  S.whisperInit = (async () => {
    const w = new Worker("./whisper_worker.mjs", { type: "module" });
    S.whisperPending = new Map(); S.whisperReq = 0;
    try {
      await new Promise((resolve, reject) => {
        w.onmessage = (e) => { if (e.data.type === "ready") { S.whisperEp = e.data.ep; resolve(); } else if (e.data.type === "error") reject(new Error(e.data.error)); };
        w.postMessage({ type: "init" });
      });
    } catch (e) { w.terminate(); S.whisperInit = null; throw e; }   // don't leak a half-built worker
    // Steady state: route replies by id; a null-id error = a fatal worker crash.
    w.onmessage = (e) => {
      const m = e.data;
      if (m.type === "error" && m.id == null) { for (const p of S.whisperPending.values()) p.reject(new Error(m.error || "whisper error")); S.whisperPending.clear(); return; }
      const p = S.whisperPending.get(m.id); if (!p) return; S.whisperPending.delete(m.id);
      m.type === "text" ? p.resolve(m.text) : p.reject(new Error(m.error));
    };
    w.onerror = () => { for (const p of S.whisperPending.values()) p.reject(new Error("whisper worker error")); S.whisperPending.clear(); };
    S.whisper = w;
  })();
  return S.whisperInit;
}
function whisperRun(audio) {
  const id = ++S.whisperReq;
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => {
      if (S.whisperPending.delete(id)) {
        try { S.whisper && S.whisper.terminate(); } catch {}
        S.whisper = null; S.whisperInit = null;       // recycle a wedged worker; captionTick rebuilds it
        reject(new Error("whisper timeout"));
      }
    }, window.__whisperTimeoutMs || 15000);
    const wrap = (fn) => (v) => { clearTimeout(to); fn(v); };
    S.whisperPending.set(id, { resolve: wrap(resolve), reject: wrap(reject) });
    S.whisper.postMessage({ type: "transcribe", id, audio }, [audio.buffer]);
  });
}
async function captionTick() {
  const m = S.mic; if (!m || m.capBusy || !m.ring || m.ring.length < 8000) return;
  if (!S.whisper) { ensureWhisper().catch(() => {}); return; }      // (re)build if missing/recycled
  const win = m.ring.slice(Math.max(0, m.ring.length - 5 * 16000));
  if (audioRms(win) < 0.004) return;         // light silence mask: skip only near-silent windows
  m.capBusy = true;
  try { const text = await whisperRun(win); if (S.mic === m && text) pushCaption(text); }
  catch (e) { console.warn("whisper:", e.message); }
  finally { m.capBusy = false; }            // always release so a hiccup can't freeze captions
}

// ── Microphone ──
async function startMic() {
  if (S.mic) return stopMic();
  $("mic").disabled = true; clearCaption();
  setStatus("loading the model — one moment…");
  try {
    await ensureWorker();
    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    await ctx.resume();
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const srcNode = ctx.createMediaStreamSource(stream);
    const proc = ctx.createScriptProcessor(4096, 1, 1);
    const KEEP = S.IN + 16000;
    const mic = { ctx, stream, proc, srcNode, ring: new Float32Array(0), busy: false };
    proc.onaudioprocess = (e) => {
      if (S.mic !== mic) return;               // ignore audio from a torn-down session
      const ch = e.inputBuffer.getChannelData(0);
      const merged = new Float32Array(mic.ring.length + ch.length);
      merged.set(mic.ring); merged.set(ch, mic.ring.length);
      mic.ring = merged.length > KEEP ? merged.slice(merged.length - KEEP) : merged;
    };
    srcNode.connect(proc); proc.connect(ctx.destination);
    S.mic = mic;
    $("mic").textContent = "■ Stop microphone"; $("mic").style.background = "#555"; $("mic").disabled = false;
    $("mic").setAttribute("aria-pressed", "true");
    setStatus("listening — running on CPU, the brain refreshes every few seconds; speak continuously.");
    mic.timer = setInterval(predictNow, 500);
    // captions run in their own worker so they don't compete with inference
    ensureWhisper().then(() => { if (S.mic === mic && !mic.capTimer) mic.capTimer = setInterval(captionTick, 1200); })
      .catch((e) => console.warn("whisper init:", e.message));
  } catch (e) {
    console.error(e);
    let msg = e.message;
    if (e.name === "NotAllowedError" || e.name === "SecurityError") msg = "microphone permission denied — allow it and try again";
    else if (e.name === "NotFoundError") msg = "no microphone found";
    else if (!window.isSecureContext) msg = "microphone needs HTTPS (or localhost)";
    setStatus("✗ mic: " + msg);
    $("mic").disabled = false; $("mic").setAttribute("aria-pressed", "false");
  }
}
async function predictNow() {
  const m = S.mic; if (!m || m.busy || !m.ring || m.ring.length < S.IN) return;
  m.busy = true;
  try {
    const out = await workerRun(m.ring.slice(m.ring.length - S.IN));
    if (S.mic !== m) return;                    // stopped/restarted mid-inference -> don't repaint
    paintFrame(out);
  } catch (e) { console.error(e); }
  finally { m.busy = false; }
}
function stopMic() {
  const m = S.mic; if (!m) return;
  clearInterval(m.timer); clearInterval(m.capTimer);
  try { m.proc.disconnect(); m.srcNode.disconnect(); m.stream.getTracks().forEach((t) => t.stop()); m.ctx.close(); } catch {}
  S.mic = null; clearCaption(); resetBrain();
  $("mic").textContent = "● Start microphone"; $("mic").style.background = ""; $("mic").setAttribute("aria-pressed", "false");
  setStatus("mic stopped — pick a clip or start again.");
}

$("run").addEventListener("click", run);
$("mic").addEventListener("click", startMic);
$("retry").addEventListener("click", init);
init();

// Hooks for the node_check tests; the page itself doesn't use these.
window.__appLoaded = true;
window.__paintTest = (flat) => paintFrame(flat);
window.__activeCount = () => {   // number of "lit" vertices (deviating from neutral)
  if (!S.colorAttr || !S.baseColor) return -1;
  const a = S.colorAttr.array, b = S.baseColor; let n = 0;
  for (let i = 0; i < a.length; i += 3) {
    if (Math.abs(a[i] - b[i]) + Math.abs(a[i + 1] - b[i + 1]) + Math.abs(a[i + 2] - b[i + 2]) > 0.02) n++;
  }
  return n;
};
