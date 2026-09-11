// One source for the browser export. The research page never prefetches weights.
export const MODEL = Object.freeze({
  repository: 'omermosa/rabbit',
  revision: 'c6405bf1788c46e7fa710f5c246f16b443ab024e',
  filename: 'rabbit_fp32.onnx',
  bytes: 422176732,
  cache: 'rabbit-model-v2',
});
export function modelURLs(hostname) {
  const remote = `https://huggingface.co/${MODEL.repository}/resolve/${MODEL.revision}/${MODEL.filename}`;
  return /^(localhost|127\.0\.0\.1|\[::1\])$/.test(hostname)
    ? [`assets/${MODEL.filename}`, remote] : [remote];
}

export async function streamModel(url, signal, onProgress, stallMs = 25000) {
  signal.throwIfAborted();
  const controller = new AbortController();
  const abort = () => controller.abort(signal.reason);
  signal.addEventListener('abort', abort, { once: true });
  let timer;
  const arm = () => {
    clearTimeout(timer);
    timer = setTimeout(() => controller.abort(new Error('Download stalled. Please retry.')), stallMs);
  };
  arm();
  let reader;
  try {
    const response = await fetch(url, { signal: controller.signal, mode: 'cors' });
    if (!response.ok) throw new Error(`Model download returned HTTP ${response.status}.`);
    if (!response.body) throw new Error('This browser cannot stream the model download.');
    const total = Number(response.headers.get('content-length')) || 0;
    reader = response.body.getReader();
    const chunks = []; let size = 0;
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value); size += value.length; arm();
      onProgress?.(size, total);
    }
    signal.throwIfAborted();
    if (!size || (total && size !== total)) throw new Error('Incomplete model download. Please retry.');
    const bytes = new Uint8Array(size); let offset = 0;
    for (const chunk of chunks) { bytes.set(chunk, offset); offset += chunk.length; }
    return bytes;
  } finally {
    clearTimeout(timer);
    signal.removeEventListener('abort', abort);
    if (controller.signal.aborted) await reader?.cancel().catch(() => {});
    reader?.releaseLock();
  }
}

export async function getModelBytes(urls, signal, onProgress) {
  let cache;
  try { cache = await caches.open(MODEL.cache); } catch { /* Storage may be unavailable. */ }
  signal.throwIfAborted();
  if (cache) for (const url of urls) {
    const hit = await cache.match(url).catch(() => null);
    if (hit?.ok) {
      const bytes = new Uint8Array(await hit.arrayBuffer());
      signal.throwIfAborted();
      if (bytes.length === MODEL.bytes) return bytes;
      await cache.delete(url).catch(() => {});
    }
  }
  let lastError;
  for (const url of urls) {
    signal.throwIfAborted();
    try {
      const bytes = await streamModel(url, signal, onProgress);
      if (bytes.length !== MODEL.bytes) throw new Error('Model file has an unexpected size. Please retry.');
      signal.throwIfAborted();
      try { await cache?.put(url, new Response(bytes, { headers: { 'content-type': 'application/octet-stream' } })); } catch { /* Quota/private browsing: use the downloaded bytes. */ }
      signal.throwIfAborted();
      return bytes;
    } catch (error) { signal.throwIfAborted(); lastError = error; }
  }
  throw lastError || new Error('No microphone model is configured.');
}

// A failed module import, aborted preparation or silent worker must not leave
// the page indefinitely claiming to initialize. The caller installs run handlers.
export function initializeWorker(worker, message, transfers, signal, timeoutMs = 120000) {
  return new Promise((resolve, reject) => {
    let timer;
    const cleanup = () => {
      clearTimeout(timer); signal.removeEventListener('abort', abort);
      worker.onmessage = null; worker.onerror = null; worker.onmessageerror = null;
    };
    const fail = error => { cleanup(); worker.terminate(); reject(error); };
    const abort = () => fail(signal.reason || new DOMException('Cancelled', 'AbortError'));
    if (signal.aborted) return abort();
    signal.addEventListener('abort', abort, { once: true });
    timer = setTimeout(() => fail(new Error('Model initialization timed out. Please retry.')), timeoutMs);
    worker.onerror = () => fail(new Error('Model worker could not load. Check your connection and retry.'));
    worker.onmessageerror = () => fail(new Error('Model worker returned an unreadable message.'));
    worker.onmessage = ({ data }) => {
      if (data.type === 'ready') { cleanup(); resolve(data); }
      else if (data.type === 'error') fail(new Error(data.error || 'Model initialization failed.'));
    };
    try { worker.postMessage(message, transfers); } catch (error) { fail(error); }
  });
}
