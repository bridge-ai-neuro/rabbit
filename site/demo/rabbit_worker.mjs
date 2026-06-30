// Runs ONNX inference off the main thread. Messages:
//   {type:'init', model} -> {type:'ready', ep}
//   {type:'run', id, data, rows, cols} -> {type:'result', id, preds}
import * as ort from "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/ort.webgpu.bundle.min.mjs";

let session = null, ep = "";

self.onmessage = async (e) => {
  const m = e.data;
  try {
    if (m.type === "init") {
      const eps = m.eps || ["webgpu", "wasm"];
      let last = "";
      for (const cand of eps) {
        try { session = await ort.InferenceSession.create(new Uint8Array(m.model), { executionProviders: [cand], graphOptimizationLevel: "all" }); ep = cand; break; }
        catch (err) { last = `${cand}: ${err.message}`; session = null; }
      }
      if (!session) throw new Error("no backend (" + last + ")");
      self.postMessage({ type: "ready", ep });
    } else if (m.type === "run") {
      const t = (await session.run({ input_wav: new ort.Tensor("float32", m.data, [m.rows, m.cols]) })).flat_predictions;
      const d = (t.data && t.data.length) ? t.data : await t.getData();
      const out = d instanceof Float32Array ? d : new Float32Array(d);
      self.postMessage({ type: "result", id: m.id, preds: out }, [out.buffer]);
    } else if (m.type === "switch") {
      session = await ort.InferenceSession.create(new Uint8Array(m.model), { executionProviders: [m.ep], graphOptimizationLevel: "all" });
      ep = m.ep; self.postMessage({ type: "ready", ep });
    }
  } catch (err) { self.postMessage({ type: "error", id: m.id, error: err.message }); }
};
