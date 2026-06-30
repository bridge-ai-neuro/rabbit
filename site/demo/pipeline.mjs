// Pipeline helpers shared by the demo and the Node parity test — pure math, no
// DOM/WebGL/ORT. The preprocessing mirrors rabbit/inference/audio.py.

export const DEFAULT_PARAMS = {
  sample_rate: 16000,
  tr_length: 1.49,
  hrf_delay: 6,
  window_seconds: 1.49,   // predictor default: window_seconds = tr_length
  audio_onset: 0.0,
};

export function windowSamples(p) {
  return Math.round(p.window_seconds * p.sample_rate);
}

// Python round(): round-half-to-even.
export function pyRound(x) {
  const f = Math.floor(x), d = x - f;
  if (d < 0.5) return f;
  if (d > 0.5) return f + 1;
  return f % 2 === 0 ? f : f + 1;
}

export const trTime = (k, p) => p.audio_onset + (k + 1) * p.tr_length;

// One EXP-sample audio window ending at TR k (mirrors align_wav_to_trs).
export function alignedWindow(wav, k, p, EXP) {
  const out = new Float32Array(EXP);
  const t = trTime(k, p), SR = p.sample_rate;
  if (t < p.window_seconds) {
    const nHead = Math.trunc(Math.max(0, t) * SR);
    out.set(wav.subarray(0, nHead), EXP - nHead);
  } else {
    const sidx = SR * pyRound(t - p.window_seconds);
    const eidx = SR * pyRound(t);
    const chunk = wav.subarray(sidx, eidx);
    if (chunk.length < EXP) out.set(chunk, EXP - chunk.length);
    else if (chunk.length > EXP) out.set(chunk.subarray(chunk.length - EXP));
    else out.set(chunk);
  }
  return out;
}

// Per-TR model input: concat over delays [hrf..0] of aligned(i-d) (zeros if <0).
export function delayedRow(wav, i, p, EXP) {
  const row = new Float32Array((p.hrf_delay + 1) * EXP);
  let off = 0;
  for (let d = p.hrf_delay; d >= 0; d--) {
    const src = i - d;
    if (src >= 0) row.set(alignedWindow(wav, src, p, EXP), off);
    off += EXP;
  }
  return row;
}

// (indices.length, (hrf+1)*EXP) flattened input batch.
export function buildInputBatch(wav, indices, p) {
  const EXP = windowSamples(p);
  const IN = (p.hrf_delay + 1) * EXP;
  const batch = new Float32Array(indices.length * IN);
  indices.forEach((i, r) => batch.set(delayedRow(wav, i, p, EXP), r * IN));
  return { data: batch, rows: indices.length, cols: IN };
}

// How many full TRs fit in a clip of `nSamples` (matches predictor n_TRs_total).
export function numTRs(nSamples, p) {
  const usable = nSamples / p.sample_rate - p.audio_onset;
  return Math.max(0, Math.floor(usable / p.tr_length));
}

// flat (flat_dim) -> full (nVertices) fs6 array; unpredicted vertices = fillNaN.
export function scatterToFs6(flat, flatToFs6, nVertices, fill = NaN) {
  const full = new Float32Array(nVertices).fill(fill);
  for (let i = 0; i < flatToFs6.length; i++) full[flatToFs6[i]] = flat[i];
  return full;
}

// Diverging colour map matching the hero video: steel-blue (deactivation) ->
// neutral at zero -> orange/red/crimson (activation). sRGB, 0..1.
const BOLD_STOPS = [
  [0.00, 0x2f, 0x66, 0x99], [0.16, 0x57, 0x83, 0xac], [0.34, 0x93, 0xaa, 0xc3],
  [0.45, 0xbc, 0xc2, 0xc4], [0.50, 0xcd, 0xca, 0xc3], [0.56, 0xe7, 0xbb, 0x9c],
  [0.66, 0xef, 0x92, 0x55], [0.78, 0xea, 0x5f, 0x29], [0.89, 0xd8, 0x38, 0x1c],
  [1.00, 0xc0, 0x16, 0x16],
];
// Reset outColor to baseColor, then paint predicted vertices with |value| >=
// thresh using the colour map. Returns the number of vertices painted.
export function colorizeFrame(flat, flatToFs6, outColor, baseColor, vmax = 0.85, thresh = 0.10) {
  outColor.set(baseColor);
  let painted = 0;
  for (let i = 0; i < flatToFs6.length; i++) {
    const val = flat[i];
    if (Math.abs(val) < thresh) continue;
    const [r, g, b] = boldColor(val, vmax);
    const v = flatToFs6[i] * 3;
    outColor[v] = r; outColor[v + 1] = g; outColor[v + 2] = b;
    painted++;
  }
  return painted;
}

export function boldColor(value, vmax = 0.85) {
  let t = (value / vmax + 1) / 2;
  if (!Number.isFinite(t)) t = 0.5;
  t = Math.min(1, Math.max(0, t));
  for (let s = 1; s < BOLD_STOPS.length; s++) {
    const [p1, r1, g1, b1] = BOLD_STOPS[s];
    if (t <= p1) {
      const [p0, r0, g0, b0] = BOLD_STOPS[s - 1];
      const f = (t - p0) / (p1 - p0);
      return [(r0 + (r1 - r0) * f) / 255, (g0 + (g1 - g0) * f) / 255, (b0 + (b1 - b0) * f) / 255];
    }
  }
  const [, r, g, b] = BOLD_STOPS[BOLD_STOPS.length - 1];
  return [r / 255, g / 255, b / 255];
}
