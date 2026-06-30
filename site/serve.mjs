// Static server for the whole site: landing at "/", demo at "/demo/". Unlike
// `python -m http.server` it sets the right MIME types for .onnx/.f32/.wasm and
// supports HTTP range requests so the large weight can stream.
//   node serve.mjs   ->  http://localhost:8000  (demo at /demo/)
//
// No COOP/COEP on purpose: cross-origin isolation switches onnxruntime-web to
// threaded WASM, whose pthread workers (built from the CDN script) fail to create
// a session in some browsers. CORS-only keeps it reliable.
import http from "node:http";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = path.dirname(fileURLToPath(import.meta.url));
const PORT = process.env.PORT || 8000;
const MIME = {
  ".html": "text/html; charset=utf-8", ".js": "text/javascript",
  ".mjs": "text/javascript", ".json": "application/json", ".css": "text/css",
  ".svg": "image/svg+xml", ".png": "image/png", ".jpg": "image/jpeg",
  ".jpeg": "image/jpeg", ".gif": "image/gif", ".webp": "image/webp",
  ".webm": "video/webm", ".mp4": "video/mp4", ".pdf": "application/pdf",
  ".ipynb": "application/json",
  ".f32": "application/octet-stream", ".i32": "application/octet-stream",
  ".onnx": "application/octet-stream", ".wasm": "application/wasm",
};

http.createServer((req, res) => {
  const base = { "Access-Control-Allow-Origin": "*" };
  let rel = decodeURIComponent(req.url.split("?")[0]);
  if (rel.endsWith("/")) rel += "index.html";
  const p = path.join(ROOT, rel);
  // Reject path traversal; symlinked files (the dev .onnx) still resolve.
  if (!p.startsWith(ROOT) || !fs.existsSync(p) || fs.statSync(p).isDirectory()) {
    res.writeHead(404, base); return res.end("404");
  }
  const size = fs.statSync(p).size;          // follows symlinks -> real size
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
}).listen(PORT, () => {
  console.log(`RABBiT site:  http://localhost:${PORT}`);
  console.log(`live demo:    http://localhost:${PORT}/demo/`);
});
