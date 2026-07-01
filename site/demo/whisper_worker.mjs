// Live captions for mic mode (Transformers.js / Whisper). Its own worker, on the
// GPU when available, so it doesn't compete with the CPU-bound RABBiT worker.
// Messages: {type:'init'} -> {type:'ready', ep};
//   {type:'transcribe', id, audio} -> {type:'text', id, text}.
import { pipeline, env } from "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.7.1/dist/transformers.min.js";

env.allowLocalModels = false;          // always fetch from the HF hub
let asr = null, ep = "";

// If the worker dies outside the request handler (module load, OOM, uncaught
// async), still notify the main thread (id:null = fatal) instead of going silent.
self.onerror = (msg) => { try { self.postMessage({ type: "error", id: null, error: String(msg) }); } catch {} };
self.addEventListener("unhandledrejection", (ev) => { try { self.postMessage({ type: "error", id: null, error: String((ev.reason && ev.reason.message) || ev.reason) }); } catch {} });

// whisper-tiny: small and fast, for snappy captions.
const build = (device) => pipeline("automatic-speech-recognition", "onnx-community/whisper-tiny", { device });

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      // Probe for a real GPU before trying WebGPU — a failed build can leave it broken.
      let hasGpu = false;
      try { hasGpu = !!(navigator.gpu && await navigator.gpu.requestAdapter()); } catch { hasGpu = false; }
      const order = hasGpu ? ["webgpu", "wasm"] : ["wasm"];
      let last = "";
      for (const dev of order) { try { asr = await build(dev); ep = dev; break; } catch (err) { last = `${dev}: ${err.message}`; asr = null; } }
      if (!asr) throw new Error(last || "no backend");
      self.postMessage({ type: "ready", ep });
    } else if (m.type === "transcribe") {
      const r = await asr(m.audio, { chunk_length_s: 30, return_timestamps: false });
      self.postMessage({ type: "text", id: m.id, text: (r.text || "").trim() });
    }
  } catch (err) { self.postMessage({ type: "error", id: m.id, error: err.message }); }
};
