// RABBiT demo: a 3D cortex whose colours track predicted fMRI. Clips play back
// precomputed predictions; the microphone runs the model live in a worker.

import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { MODEL, modelURLs, getModelBytes, initializeWorker } from "./model-loader.mjs";
import { DEFAULT_PARAMS, windowSamples, colorizeFrame } from "./pipeline.mjs";

THREE.ColorManagement.enabled = false;   // render the authored sRGB colours as-is

const VMAX = 0.78, THRESH = 0.05;
const NEUTRAL = [0xcd / 255, 0xca / 255, 0xc3 / 255];

const MODEL_URLS = modelURLs(location.hostname);

const $ = (id) => document.getElementById(id);
const setStatus = text => { if ($("status").textContent !== text) $("status").textContent = text; };
const setProg = (f) => { const pct = Math.round(f * 100); $("prog").style.width = `${pct}%`; const b = $("barEl"); if (b) b.setAttribute("aria-valuenow", String(pct)); };
async function fetchData(url, signal) {
  const response = await fetch(url, { signal });
  if (!response.ok) throw new Error(`${url.split('/').pop()}: HTTP ${response.status}`);
  return response;
}
const fetchBuf = async (url, signal) => new Uint8Array(await (await fetchData(url, signal)).arrayBuffer());
const asF32 = (b) => new Float32Array(b.buffer, b.byteOffset, b.byteLength / 4);
const asI32 = (b) => new Int32Array(b.buffer, b.byteOffset, b.byteLength / 4);

const S = { rm: null, params: null, clips: null, flatToFs6: null, baseColor: null, colorAttr: null,
  IN: 0, transcripts: null, clipGen: 0, playback: null, clipLoad: null,
  worker: null, ep: "", pending: new Map(), reqId: 0, mic: null, micGen: 0, modelLoad: null,
  whisper: null, whisperInit: null, whisperEp: "", whisperPending: new Map(), whisperReq: 0 };

function hasWebGL() {
  try { const c = document.createElement("canvas"); return !!(window.WebGLRenderingContext && (c.getContext("webgl") || c.getContext("experimental-webgl"))); }
  catch { return false; }
}

// Mic mode pulls a 422 MB model and runs heavy WASM — fine on a desktop, an
// uncatchable OOM crash on phones. Gate it rather than let it fall over.
function micSupport() {
  const why = [];
  if (!window.isSecureContext) why.push("needs HTTPS or localhost");
  if (!(navigator.mediaDevices && navigator.mediaDevices.getUserMedia)) why.push("microphone unavailable");
  const coarse = window.matchMedia && window.matchMedia("(pointer: coarse)").matches;
  const mobile = /Mobi|Android|iPhone|iPad|iPod/i.test(navigator.userAgent || "");
  if (coarse || mobile) why.push("experimental microphone mode is supported on desktops only");
  const mem = navigator.deviceMemory;            // GB, where the browser reports it
  if (typeof mem === "number" && mem > 0 && mem < 8) why.push("needs ≈ 8 GB RAM");
  return { ok: why.length === 0, why };
}

// ── Three.js brain ──────────────────────────────────────────────────────────
function buildScene(coords, faces) {
  const canvas = $("brain");
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0xffffff);
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
  const cam = new THREE.PerspectiveCamera(38, 1, R * 0.05, R * 30); cam.up.set(0, 0, 1);
  const controls = new OrbitControls(cam, renderer.domElement);
  controls.enableDamping = !matchMedia('(prefers-reduced-motion: reduce)').matches;
  controls.enablePan = false;
  let lastWidth = 0, lastHeight = 0, fitDistance = 0;
  function fit() {
    const halfFov = Math.atan(Math.tan(THREE.MathUtils.degToRad(cam.fov / 2)) * Math.min(1, cam.aspect));
    return R / Math.sin(halfFov) * 1.08;
  }
  function resetView() {
    controls.target.set(0, 0, 0); cam.position.set(-fitDistance, 0, R * 0.08);
    cam.lookAt(controls.target); controls.update();
  }
  function resize() {
    const w = canvas.clientWidth, h = canvas.clientHeight;
    if (!w || !h || (w === lastWidth && h === lastHeight)) return;
    lastWidth = w; lastHeight = h;
    renderer.setSize(w, h, false); cam.aspect = w / h; cam.updateProjectionMatrix();
    const next = fit();
    if (fitDistance) cam.position.multiplyScalar(next / fitDistance);
    else { fitDistance = next; resetView(); }
    fitDistance = next;
    controls.minDistance = R * 1.3; controls.maxDistance = Math.max(R * 12, next * 2);
  }
  $('resetView').addEventListener('click', resetView); $('resetView').disabled = false;
  canvas.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', '+', '=', '-', '_'].includes(event.key)) return;
    event.preventDefault();
    if (event.key.startsWith('Arrow')) {
      const axis = event.key === 'ArrowLeft' || event.key === 'ArrowRight'
        ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0).applyQuaternion(cam.quaternion).cross(cam.position).normalize();
      cam.position.applyAxisAngle(axis, ['ArrowLeft', 'ArrowUp'].includes(event.key) ? 0.12 : -0.12);
    } else cam.position.multiplyScalar(['+', '='].includes(event.key) ? 0.9 : 1.1);
    controls.update();
  });
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
  ["run", "clipSel", "mic", "loadModel", "liveCaptions"].forEach(id => $(id).disabled = true);
}
function gateMic() {
  const support = micSupport();
  $('micCard').classList.toggle('off', !support.ok);
  $('loadModel').disabled = !support.ok;
  if (!support.ok) {
    $('micNote').textContent = 'Microphone mode is unavailable here: ' + support.why.join('; ') + '. You can still replay the recorded examples.';
    $('loadModel').textContent = 'Microphone model unavailable';
  }
}
async function init() {
  $("retry").hidden = true;
  if (!hasWebGL()) return showNoWebGL();
  try {
    setStatus("Loading brain surface and examples…");
    const [coordsB, facesB, ftfB, rm, clips, transcripts] = await Promise.all([
      fetchBuf("assets/pial_coords.f32"), fetchBuf("assets/faces.i32"), fetchBuf("assets/flat_to_fs6.i32"),
      fetchData("assets/render_manifest.json").then((r) => r.json()),
      fetchData("assets/clips.json").then((r) => r.json()),
      fetchData("assets/transcripts.json").then((r) => r.json()).catch(() => ({})),
    ]);
    S.rm = rm; S.flatToFs6 = asI32(ftfB); S.clips = clips; S.transcripts = transcripts;
    S.params = { ...DEFAULT_PARAMS }; S.IN = (S.params.hrf_delay + 1) * windowSamples(S.params);
    if (!window.__brainReady) { buildScene(asF32(coordsB), asI32(facesB)); window.__brainReady = true; }
    $('clipSel').replaceChildren(...clips.map(c => new Option(`${c.label.replace(' · ', ' ')} (${c.duration_s}s)`, c.id)));
    $('clipSel').disabled = false; $('run').disabled = false;
    $("clipCard").classList.remove("off");
    gateMic(); $("barEl").hidden = true;
    setStatus("Ready. Choose an example and press Play.");
  } catch (e) {
    console.error(e);
    setStatus("✗ couldn’t load demo data: " + e.message);
    $("barEl").hidden = true;
    $("retry").hidden = false;   // let the user retry instead of leaving a dead page
  }
}

// Clip audio is the clock for both the cortical frames and transcript.
const clockText = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
function clipControls(state = 'idle') {
  $('run').textContent = state === 'playing' ? 'Pause' : state === 'paused' ? 'Resume' : state === 'loading' ? 'Loading…' : 'Play';
  $('run').disabled = state === 'loading';
  $('stopClip').disabled = state === 'idle';
  $('clipCard').setAttribute('aria-busy', String(state === 'loading'));
}
function stopClip(message = '') {
  ++S.clipGen; S.clipLoad?.abort(); S.clipLoad = null;
  const playback = S.playback; S.playback = null;
  if (playback) {
    cancelAnimationFrame(playback.frame);
    if (playback.source) { playback.source.onended = null; try { playback.source.stop(); } catch {} }
    playback.ctx.close().catch(() => {});
  }
  clipControls(); setProg(0); $('barEl').hidden = true;
  $('clipProgress').value = 0;
  const meta = S.clips?.find(c => c.id === $('clipSel').value);
  $('clipTime').textContent = `0:00 / ${clockText(meta?.duration_s || 24)}`;
  clearCaption(); resetBrain(); if (message) setStatus(message);
}
async function run() {
  if (S.playback?.source) {
    const playback = S.playback;
    $('run').disabled = true;
    try {
      if (playback.ctx.state === 'running') await playback.ctx.suspend();
      else await playback.ctx.resume();
      if (S.playback !== playback) return;
      const paused = playback.ctx.state !== 'running';
      clipControls(paused ? 'paused' : 'playing');
      setStatus(`${playback.meta.label.replace(' · ', ' ')}: ${paused ? 'paused' : 'playing'}.`);
    } catch (error) { if (S.playback === playback) stopClip(`Audio could not resume. Press Play to retry. ${error.message}`); }
    return;
  }
  stopMic(); stopClip();
  const gen = S.clipGen;
  const meta = S.clips.find(c => c.id === $('clipSel').value);
  const controller = new AbortController(); S.clipLoad = controller;
  clipControls('loading'); setStatus(`Loading ${meta.label.replace(' · ', ' ')}…`);
  $('barEl').hidden = false; setProg(0.1);
  try {
    // Resume in the click gesture, before waiting for the clip download.
    const ctx = new (window.AudioContext || window.webkitAudioContext)();
    const playback = { ctx, meta, source: null, frame: null }; S.playback = playback;
    await ctx.resume();
    const [predsB, clipB] = await Promise.all([
      fetchBuf(`assets/preds_${meta.id}.f32`, controller.signal), fetchBuf('assets/' + meta.file, controller.signal),
    ]);
    if (gen !== S.clipGen) return;
    const preds = asF32(predsB), clip = asF32(clipB), OUT = S.rm.flat_dim;
    if (preds.length !== meta.n_tr * OUT || clip.length !== meta.n_samples) throw new Error('Incomplete clip data.');
    const buffer = ctx.createBuffer(1, clip.length, meta.sample_rate); buffer.copyToChannel(clip, 0);
    const source = ctx.createBufferSource(); source.buffer = buffer; source.connect(ctx.destination);
    playback.source = source; const started = ctx.currentTime;
    const duration = clip.length / meta.sample_rate, segs = S.transcripts[meta.id] || [];
    source.onended = () => { if (gen === S.clipGen) stopClip('Example finished. Replay or choose another.'); };
    source.start(); S.clipLoad = null; $('barEl').hidden = true;
    clipControls('playing'); setStatus(`${meta.label.replace(' · ', ' ')}: playing.`);
    const tick = () => {
      if (gen !== S.clipGen) return;
      const elapsed = Math.min(duration, Math.max(0, ctx.currentTime - started));
      const tr = Math.min(meta.n_tr - 1, Math.floor(elapsed / S.params.tr_length));
      paintFrame(preds.subarray(tr * OUT, (tr + 1) * OUT));
      const text = clipCaptionAt(elapsed, segs), caption = $('caption');
      if (caption.textContent !== text) caption.textContent = text;
      caption.classList.toggle('show', !!text);
      $('clipProgress').value = elapsed / duration;
      $('clipTime').textContent = `${clockText(elapsed)} / ${clockText(duration)}`;
      playback.frame = requestAnimationFrame(tick);
    };
    tick();
  } catch (error) { if (gen === S.clipGen) stopClip(`Example could not play. Press Play to retry. ${error.message}`); }
}

// Model preparation is explicit and independent of example playback.
function setMicStatus(text, busy = false) {
  $('micStatus').textContent = text;
  $('micStatus').classList.toggle('busy', busy);
}
function disposeModel(error) {
  S.worker?.terminate(); S.worker = null;
  for (const pending of S.pending.values()) pending.reject(error);
  S.pending.clear(); stopMic();
  $('mic').disabled = true; $('liveCaptions').disabled = true;
  $('loadModel').hidden = false; $('loadModel').disabled = false;
  $('loadModel').textContent = 'Retry microphone model';
  setMicStatus(error.message);
}
async function prepareModel() {
  if (S.modelLoad || S.worker) return;
  const controller = new AbortController(); S.modelLoad = controller;
  $('loadModel').hidden = true; $('cancelModel').hidden = false;
  $('modelProgress').hidden = false; $('modelProgress').removeAttribute('value');
  setMicStatus('Checking for saved weights…', true);
  try {
    let lastMB = -1, lastProgressUpdate = 0;
    const bytes = await getModelBytes(MODEL_URLS, controller.signal, (got, total) => {
      const mb = Math.floor(got / 1e6), size = total || MODEL.bytes;
      $('modelProgress').value = got / size;
      if (mb !== lastMB && (performance.now() - lastProgressUpdate >= 500 || got >= size)) { lastMB = mb; lastProgressUpdate = performance.now(); setMicStatus(`Downloading weights: ${mb} / ${Math.ceil(size / 1e6)} MB`, true); }
    });
    controller.signal.throwIfAborted();
    setMicStatus('Weights loaded. Initializing the model on CPU…', true);
    $('modelProgress').removeAttribute('value');
    const worker = new Worker('./rabbit_worker.mjs', { type: 'module' });
    const ready = await initializeWorker(worker, { type: 'init', model: bytes.buffer, eps: ['wasm'] }, [bytes.buffer], controller.signal);
    S.ep = ready.ep; S.worker = worker;
    worker.onmessage = ({ data }) => {
      if (data.type === 'error' && data.id == null) return disposeModel(new Error(data.error || 'The model stopped. Reload it to retry.'));
      const pending = S.pending.get(data.id); if (!pending) return;
      S.pending.delete(data.id);
      data.type === 'result' ? pending.resolve(data.preds) : pending.reject(new Error(data.error));
    };
    worker.onerror = worker.onmessageerror = () => disposeModel(new Error('The model worker stopped. Reload it to retry.'));
    $('mic').disabled = false; $('liveCaptions').disabled = false;
    setMicStatus('Model ready. Start the microphone when you are ready.');
  } catch (error) {
    $('loadModel').hidden = false; $('loadModel').textContent = 'Retry microphone model (422 MB)';
    setMicStatus(controller.signal.aborted ? 'Loading cancelled. Recorded examples are available.' : `Could not prepare the model. ${error.message}`);
  } finally {
    S.modelLoad = null; $('cancelModel').hidden = true; $('modelProgress').hidden = true;
  }
}
function workerRun(data) {
  const id = ++S.reqId;
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => { if (S.pending.delete(id)) reject(new Error("inference timeout")); }, 30000);
    const wrap = (fn) => (v) => { clearTimeout(to); fn(v); };
    S.pending.set(id, { resolve: wrap(resolve), reject: wrap(reject) });
    try { S.worker.postMessage({ type: "run", id, data, rows: 1, cols: S.IN }, [data.buffer]); }
    catch (error) { S.pending.delete(id); clearTimeout(to); reject(error); }
  });
}

// Live captions in a separate worker (display only — not fed to the model).
// Single-flight: repeat/concurrent callers await the same init.
function stopWhisper() {
  S.whisperLoad?.abort(); S.whisperLoad = null;
  S.whisper?.terminate(); S.whisper = null; S.whisperInit = null;
  for (const pending of S.whisperPending.values()) pending.reject(new Error('Captions stopped.'));
  S.whisperPending.clear();
  $("captionNote").textContent = "Optional captions download a separate speech-recognition model. They are display-only.";
}
function ensureWhisper() {
  if (S.whisperInit) return S.whisperInit;
  const controller = new AbortController(); S.whisperLoad = controller;
  $('captionNote').textContent = 'Loading the separate live-caption model…';
  S.whisperInit = (async () => {
    const worker = new Worker('./whisper_worker.mjs', { type: 'module' });
    const ready = await initializeWorker(worker, { type: 'init' }, [], controller.signal);
    S.whisperEp = ready.ep; S.whisper = worker;
    worker.onmessage = ({ data }) => {
      if (data.type === 'error' && data.id == null) {
        stopWhisper(); $('captionNote').textContent = 'Live captions unavailable. Brain prediction can continue.'; return;
      }
      const pending = S.whisperPending.get(data.id); if (!pending) return;
      S.whisperPending.delete(data.id);
      data.type === 'text' ? pending.resolve(data.text) : pending.reject(new Error(data.error));
    };
    worker.onerror = worker.onmessageerror = () => { stopWhisper(); $('captionNote').textContent = 'Live captions unavailable. Brain prediction can continue.'; };
    $('captionNote').textContent = 'Live captions are display-only; RABBiT receives audio.';
  })().catch(error => {
    if (S.whisperLoad === controller) { S.whisperInit = null; $('captionNote').textContent = 'Live captions could not load. Brain prediction can continue. Uncheck and recheck captions to retry.'; }
    throw error;
  });
  return S.whisperInit;
}
function whisperRun(audio) {
  const id = ++S.whisperReq;
  return new Promise((resolve, reject) => {
    const to = setTimeout(() => {
      if (S.whisperPending.delete(id)) {
        stopWhisper();
        reject(new Error("Live captions timed out."));
      }
    }, window.__whisperTimeoutMs || 15000);
    const wrap = (fn) => (v) => { clearTimeout(to); fn(v); };
    S.whisperPending.set(id, { resolve: wrap(resolve), reject: wrap(reject) });
    try { S.whisper.postMessage({ type: "transcribe", id, audio }, [audio.buffer]); }
    catch (error) { S.whisperPending.delete(id); clearTimeout(to); reject(error); }
  });
}
async function captionTick() {
  const m = S.mic; if (!m || m.capBusy || !m.ring || m.ring.length < 8000) return;
  if (!S.whisper) return;      // The checkbox offers an explicit retry after a failure.
  const win = m.ring.slice(Math.max(0, m.ring.length - 5 * 16000));
  if (audioRms(win) < 0.004) return;         // light silence mask: skip only near-silent windows
  m.capBusy = true;
  try { const text = await whisperRun(win); if (S.mic === m && text) pushCaption(text); }
  catch (error) { if (S.mic === m && $("liveCaptions").checked) $("captionNote").textContent = "Live captions unavailable. Uncheck and recheck captions to retry; brain prediction can continue."; }
  finally { m.capBusy = false; }            // always release so a hiccup can't freeze captions
}

// ── Microphone ──
function closeMicResources(mic) {
  clearInterval(mic.timer); clearInterval(mic.capTimer);
  try { mic.proc?.disconnect(); mic.srcNode?.disconnect(); } catch {}
  mic.stream?.getTracks().forEach(track => track.stop());
  mic.ctx?.close().catch(() => {});
}
function startCaptions(mic) {
  if (!$('liveCaptions').checked) return;
  ensureWhisper().then(() => {
    if (S.mic === mic && !mic.capTimer) mic.capTimer = setInterval(captionTick, 1200);
  }).catch(() => {});
}
async function startMic() {
  if (S.mic) return stopMic('Microphone stopped. Choose an example or start again.');
  if (!S.worker) return;
  stopClip(); clearCaption();
  const gen = ++S.micGen;
  const mic = { ctx: null, stream: null, ring: new Float32Array(0), busy: false, started: false };
  S.mic = mic;
  $('mic').textContent = 'Cancel microphone';
  setStatus('Requesting microphone access…');
  try {
    mic.ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
    await mic.ctx.resume();
    if (gen !== S.micGen) return;
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    if (gen !== S.micGen) { stream.getTracks().forEach(track => track.stop()); return; }
    mic.stream = stream;
    if (mic.ctx.sampleRate !== S.params.sample_rate) throw new Error('This browser cannot capture at the required 16 kHz. Try another desktop browser.');
    mic.srcNode = mic.ctx.createMediaStreamSource(stream);
    mic.proc = mic.ctx.createScriptProcessor(4096, 1, 1);
    const KEEP = S.IN + S.params.sample_rate;
    mic.proc.onaudioprocess = event => {
      if (S.mic !== mic) return;
      const channel = event.inputBuffer.getChannelData(0);
      const merged = new Float32Array(mic.ring.length + channel.length);
      merged.set(mic.ring); merged.set(channel, mic.ring.length);
      mic.ring = merged.length > KEEP ? merged.slice(merged.length - KEEP) : merged;
    };
    stream.getTracks().forEach(track => track.addEventListener('ended', () => {
      if (S.mic === mic) stopMic('Microphone disconnected. Reconnect it and start again.');
    }));
    mic.srcNode.connect(mic.proc); mic.proc.connect(mic.ctx.destination); mic.started = true;
    $('mic').textContent = 'Stop microphone'; $('mic').setAttribute('aria-pressed', 'true');
    setStatus('Listening. Collecting about 10 seconds of audio context before the first prediction…');
    mic.timer = setInterval(predictNow, 500); startCaptions(mic);
  } catch (error) {
    closeMicResources(mic);
    if (gen !== S.micGen) return;
    let message = error.message;
    if (['NotAllowedError', 'SecurityError'].includes(error.name)) message = 'Microphone permission was denied. Allow access in your browser and try again.';
    else if (error.name === 'NotFoundError') message = 'No microphone found. Connect one and try again.';
    stopMic(message);
  }
}
async function predictNow() {
  const m = S.mic; if (!m || m.busy || !m.ring || m.ring.length < S.IN) return;
  m.busy = true;
  try {
    const out = await workerRun(m.ring.slice(m.ring.length - S.IN));
    if (S.mic !== m) return;                    // stopped/restarted mid-inference -> don't repaint
    paintFrame(out);
    setStatus("Listening. Predictions update after each CPU inference.");
  } catch (error) { if (S.mic === m) { disposeModel(error); setStatus("Microphone prediction stopped. Reload the model to retry, or play an example."); } }
  finally { m.busy = false; }
}
function stopMic(message = '') {
  ++S.micGen;
  const mic = S.mic; S.mic = null;
  if (mic) closeMicResources(mic);
  stopWhisper(); clearCaption(); resetBrain();
  $('mic').textContent = 'Start microphone'; $('mic').setAttribute('aria-pressed', 'false');
  if (message) setStatus(message);
}

$('run').addEventListener('click', run);
$('stopClip').addEventListener('click', () => stopClip('Example stopped. Press Play to start again.'));
$('clipSel').addEventListener('change', () => { stopMic(); stopClip('Example selected. Press Play to start.'); });
$('loadModel').addEventListener('click', prepareModel);
$('cancelModel').addEventListener('click', () => S.modelLoad?.abort());
$('mic').addEventListener('click', startMic);
$('liveCaptions').addEventListener('change', () => {
  if (S.mic?.capTimer) { clearInterval(S.mic.capTimer); S.mic.capTimer = null; }
  stopWhisper(); clearCaption();
  if (S.mic?.started) startCaptions(S.mic);
});
$('retry').addEventListener('click', init);
window.addEventListener('pagehide', () => { stopMic(); stopClip(); S.modelLoad?.abort(); S.worker?.terminate(); S.worker = null; });
window.addEventListener('pageshow', event => { if (event.persisted) location.reload(); });
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
