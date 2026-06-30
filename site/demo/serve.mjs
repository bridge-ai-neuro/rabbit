// Standalone static server for the demo folder (range requests so the .onnx
// weight streams). For the full site, run ../serve.mjs from the site root.
// No COOP/COEP: it breaks onnxruntime-web's worker.
//   node serve.mjs   ->  http://localhost:8000
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 8000;
const MIME = { ".html": "text/html; charset=utf-8", ".js": "text/javascript",
  ".mjs": "text/javascript", ".json": "application/json", ".css": "text/css",
  ".f32": "application/octet-stream", ".i32": "application/octet-stream",
  ".onnx": "application/octet-stream", ".wasm": "application/wasm" };

http.createServer((req, res) => {
  // NOTE: deliberately NO COOP/COEP. Cross-origin isolation flips ort-web to
  // threaded WASM (SharedArrayBuffer + workers), which fails to create a
  // session in some browsers. Single-threaded WASM is slower but reliable;
  // WebGPU (the fast path) doesn't need isolation anyway.
  const base = { "Access-Control-Allow-Origin": "*" };
  let rel = decodeURIComponent(req.url.split("?")[0]);
  if (rel === "/") rel = "/index.html";
  const p = path.join(ROOT, rel);
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
    res.writeHead(404, base); return res.end("404");
  }
  const size = fs.statSync(p).size;
  const type = MIME[path.extname(p)] || "application/octet-stream";
  const range = req.headers.range;
  if (range) {
    const m = /bytes=(\d+)-(\d*)/.exec(range);
    const start = +m[1], end = m[2] ? +m[2] : size - 1;
    res.writeHead(206, { ...base, "Content-Type": type, "Accept-Ranges": "bytes",
      "Content-Range": `bytes ${start}-${end}/${size}`, "Content-Length": end - start + 1 });
    fs.createReadStream(p, { start, end }).pipe(res);
  } else {
    res.writeHead(200, { ...base, "Content-Type": type, "Content-Length": size, "Accept-Ranges": "bytes" });
    fs.createReadStream(p).pipe(res);
  }
}).listen(PORT, () => console.log(`RABBiT demo: http://localhost:${PORT}`));
