import { NativeModules } from 'react-native';
import * as FileSystem from 'expo-file-system/legacy';
import { base64ToBytes, bytesToInt16 } from './bytes';
import { ensureDiarizationModels } from './models';
import { silenceBetween, SpeechRegion } from './vad';
import type { SpeakerSegment } from './types';

// whisper.rn segments use centiseconds (100 units = 1 second)
const SPEAKER_CHANGE_GAP_CS = 100;   // pause > 1s = candidate turn boundary
// With VAD regions, silence is measured (not inferred from drifty Whisper
// timestamps). 50cs of VAD-measured silence ≈ a real ~0.9s pause after the
// 200ms speech padding on each side.
const VAD_TURN_SILENCE_CS = 50;
const MAX_SPEAKERS = 4;
const ANALYSIS_WINDOW_SEC = 1.5;     // audio analyzed per turn (from its middle)
const FRAME = 512;                   // 32 ms @ 16 kHz
const MIN_F0 = 60;                   // Hz — male low end
const MAX_F0 = 400;                  // Hz — female/child high end
const VOICED_RMS = 0.008;            // normalized RMS gate for voiced frames
const VOICED_AC_PEAK = 0.4;          // autocorrelation peak gate
// Clusters whose pitch centroids are closer than ~1 semitone are the same voice.
const MIN_CLUSTER_F0_GAP_LOG2 = 0.08;

interface RawSegment {
  text: string;
  t0: number;
  t1: number;
}

interface Turn {
  text: string;
  t0: number;
  t1: number;
  f0?: number;        // median fundamental frequency (Hz)
  zcr?: number;       // mean zero-crossing rate of voiced frames
  cluster?: number;
}

/**
 * Identifies who is speaking, best method first:
 *
 * 1. Native sherpa-onnx diarization (pyannote segmentation + speaker
 *    embeddings) — detects speaker changes acoustically, so it splits turns
 *    even when speakers don't pause between each other.
 * 2. Pause/pitch heuristic: VAD- or gap-based turn breaks + F0 clustering.
 * 3. Pause-alternation guess.
 *
 * Returns [] for a monologue.
 */
export async function assignSpeakers(
  segments: RawSegment[],
  wavUri?: string,
  speechRegions?: SpeechRegion[],
): Promise<SpeakerSegment[]> {
  if (!segments.length) return [];

  if (wavUri) {
    const spans = await diarizeNative(wavUri);
    // A successful native run is authoritative — even a one-speaker verdict
    // (mapped to [] = monologue) beats the heuristics' guesses.
    if (spans) return speakersFromNativeSpans(segments, spans);
  }

  const turns = buildTurns(segments, speechRegions);
  if (turns.length < 2) return [];

  if (wavUri) {
    try {
      await extractVoiceFeatures(turns, wavUri);
      const clustered = clusterTurns(turns);
      if (clustered) return toSpeakerSegments(turns);
    } catch {
      // fall through to gap heuristic
    }
  }

  return assignSpeakersByGaps(segments);
}

// ---------------------------------------------------------------------------
// Native diarization (sherpa-onnx via DiarizationModule.kt)
// ---------------------------------------------------------------------------

interface NativeSpan {
  start: number;   // seconds
  end: number;     // seconds
  speaker: number; // 0-based cluster id
}

/**
 * Runs on-device sherpa-onnx diarization. Returns null when unavailable
 * (old dev build without the native module, models not downloadable,
 * processing error) — callers fall back to the heuristics.
 */
async function diarizeNative(wavUri: string): Promise<NativeSpan[] | null> {
  const mod = NativeModules.SpeakerDiarization;
  if (!mod?.diarize) return null;
  try {
    const models = await ensureDiarizationModels();
    if (!models) return null;
    const bare = wavUri.replace(/^file:\/\//, '');
    // -1 speakers = automatic count; 0.6 clustering threshold matches desktop
    // (diarize.py CLUSTER_THRESHOLD — benchmarked with the zh-en CAM++
    // embeddings; sherpa's 0.5 default over-splits, 0.8 risks merging voices)
    const spans: NativeSpan[] = await mod.diarize(bare, models.segmentation, models.embedding, -1, 0.6);
    return spans?.length ? spans : null;
  } catch {
    return null;
  }
}

/** Assigns each whisper segment the native speaker with the largest time overlap. */
function speakersFromNativeSpans(segments: RawSegment[], spans: NativeSpan[]): SpeakerSegment[] {
  const spansCs = spans.map((s) => ({ t0: s.start * 100, t1: s.end * 100, speaker: s.speaker }));

  const turns: Turn[] = segments.map((seg) => {
    let best = spansCs[0].speaker;
    let bestOverlap = 0;
    let bestDist = Infinity;
    for (const sp of spansCs) {
      const overlap = Math.min(sp.t1, seg.t1) - Math.max(sp.t0, seg.t0);
      if (overlap > bestOverlap) {
        bestOverlap = overlap;
        best = sp.speaker;
      } else if (bestOverlap <= 0) {
        // Segment sits in a diarization gap — take the nearest span
        const mid = (seg.t0 + seg.t1) / 2;
        const dist = mid < sp.t0 ? sp.t0 - mid : mid > sp.t1 ? mid - sp.t1 : 0;
        if (dist < bestDist) {
          bestDist = dist;
          best = sp.speaker;
        }
      }
    }
    return { text: seg.text.trim(), t0: seg.t0, t1: seg.t1, cluster: best };
  });

  // toSpeakerSegments merges consecutive same-speaker turns and returns []
  // for a single speaker — the app's monologue convention.
  return toSpeakerSegments(turns);
}

/** Legacy heuristic: alternate Person 1 / Person 2 on long pauses. */
export function assignSpeakersByGaps(segments: RawSegment[]): SpeakerSegment[] {
  if (!segments.length) return [];

  const turns: SpeakerSegment[] = [];
  let speakerIndex = 0;
  let currentText = segments[0].text.trim();
  let currentT0 = segments[0].t0;
  let currentT1 = segments[0].t1;

  for (let i = 1; i < segments.length; i++) {
    const gap = segments[i].t0 - segments[i - 1].t1;
    if (gap > SPEAKER_CHANGE_GAP_CS) {
      turns.push({ speaker: `Person ${speakerIndex + 1}`, text: currentText, t0: currentT0, t1: currentT1 });
      speakerIndex = (speakerIndex + 1) % 2;
      currentText = segments[i].text.trim();
      currentT0 = segments[i].t0;
      currentT1 = segments[i].t1;
    } else {
      currentText += ' ' + segments[i].text.trim();
      currentT1 = segments[i].t1;
    }
  }
  turns.push({ speaker: `Person ${speakerIndex + 1}`, text: currentText, t0: currentT0, t1: currentT1 });

  const hasTwoSpeakers = turns.some(t => t.speaker === 'Person 2');
  return hasTwoSpeakers ? turns : [];
}

/** Formats speaker turns as a plain-text conversation block for the LLM prompt. */
export function formatConversation(
  turns: SpeakerSegment[],
  names?: Record<string, string>,
): string {
  return turns.map(t => `[${names?.[t.speaker] ?? t.speaker}] ${t.text}`).join('\n');
}

// ---------------------------------------------------------------------------
// Turn building
// ---------------------------------------------------------------------------

function buildTurns(segments: RawSegment[], speechRegions?: SpeechRegion[]): Turn[] {
  const turns: Turn[] = [];
  let cur: Turn = { text: segments[0].text.trim(), t0: segments[0].t0, t1: segments[0].t1 };

  // When VAD regions are available, boundaries come from measured silence —
  // Whisper timestamps routinely hallucinate or swallow gaps.
  const isBoundary = (prev: RawSegment, next: RawSegment): boolean =>
    speechRegions?.length
      ? silenceBetween(speechRegions, prev.t1, next.t0) >= VAD_TURN_SILENCE_CS
      : next.t0 - prev.t1 > SPEAKER_CHANGE_GAP_CS;

  for (let i = 1; i < segments.length; i++) {
    if (isBoundary(segments[i - 1], segments[i])) {
      turns.push(cur);
      cur = { text: segments[i].text.trim(), t0: segments[i].t0, t1: segments[i].t1 };
    } else {
      cur.text += ' ' + segments[i].text.trim();
      cur.t1 = segments[i].t1;
    }
  }
  turns.push(cur);
  return turns;
}

// ---------------------------------------------------------------------------
// Voice feature extraction (pitch via autocorrelation)
// ---------------------------------------------------------------------------

async function extractVoiceFeatures(turns: Turn[], wavUri: string): Promise<void> {
  const uri = wavUri.startsWith('file://') ? wavUri : 'file://' + wavUri;
  const info = await FileSystem.getInfoAsync(uri);
  if (!info.exists) throw new Error('WAV not found');
  const fileSize: number = (info as any).size ?? 0;

  // Our recorder writes a fixed 44-byte header; read the sample rate from it.
  const headerB64 = await FileSystem.readAsStringAsync(uri, {
    encoding: FileSystem.EncodingType.Base64, position: 0, length: 44,
  });
  const header = base64ToBytes(headerB64);
  const sampleRate =
    header.length >= 28
      ? header[24] | (header[25] << 8) | (header[26] << 16) | (header[27] << 24)
      : 16000;
  const dataStart = 44;

  for (const turn of turns) {
    const startSample = Math.floor((turn.t0 / 100) * sampleRate);
    const endSample = Math.floor((turn.t1 / 100) * sampleRate);
    const turnSamples = Math.max(0, endSample - startSample);
    const windowSamples = Math.min(turnSamples, Math.floor(ANALYSIS_WINDOW_SEC * sampleRate));
    if (windowSamples < FRAME) continue;

    // Sample from the middle of the turn — avoids cross-talk at boundaries
    const midOffset = startSample + Math.floor((turnSamples - windowSamples) / 2);
    let bytePos = dataStart + midOffset * 2;
    bytePos -= bytePos % 2;
    const byteLen = Math.min(windowSamples * 2, Math.max(0, fileSize - bytePos));
    if (byteLen < FRAME * 2) continue;

    const b64 = await FileSystem.readAsStringAsync(uri, {
      encoding: FileSystem.EncodingType.Base64, position: bytePos, length: byteLen,
    });
    const pcm = bytesToInt16(base64ToBytes(b64));

    const { f0, zcr } = analyzePitch(pcm, sampleRate);
    if (f0) { turn.f0 = f0; turn.zcr = zcr; }
  }
}

function analyzePitch(pcm: Int16Array, sampleRate: number): { f0?: number; zcr?: number } {
  const minLag = Math.floor(sampleRate / MAX_F0);
  const maxLag = Math.min(Math.floor(sampleRate / MIN_F0), FRAME - 1);
  const f0s: number[] = [];
  const zcrs: number[] = [];
  const frame = new Float32Array(FRAME);

  for (let off = 0; off + FRAME <= pcm.length; off += FRAME) {
    // Normalize and mean-remove
    let mean = 0;
    for (let i = 0; i < FRAME; i++) mean += pcm[off + i];
    mean /= FRAME;
    let energy = 0;
    let crossings = 0;
    for (let i = 0; i < FRAME; i++) {
      const v = (pcm[off + i] - mean) / 32768;
      frame[i] = v;
      energy += v * v;
      if (i > 0 && ((frame[i - 1] >= 0) !== (v >= 0))) crossings++;
    }
    const rms = Math.sqrt(energy / FRAME);
    if (rms < VOICED_RMS) continue;

    // Normalized autocorrelation over the speech pitch range
    const r0 = energy;
    let bestLag = 0;
    let bestVal = 0;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let sum = 0;
      for (let i = 0; i + lag < FRAME; i++) sum += frame[i] * frame[i + lag];
      const norm = sum / r0;
      if (norm > bestVal) { bestVal = norm; bestLag = lag; }
    }
    if (bestVal > VOICED_AC_PEAK && bestLag > 0) {
      f0s.push(sampleRate / bestLag);
      zcrs.push(crossings / FRAME);
    }
  }

  if (f0s.length < 3) return {};
  return { f0: median(f0s), zcr: median(zcrs) };
}

function median(xs: number[]): number {
  const s = [...xs].sort((a, b) => a - b);
  const m = s.length >> 1;
  return s.length % 2 ? s[m] : (s[m - 1] + s[m]) / 2;
}

// ---------------------------------------------------------------------------
// Clustering (k-means over [log2 F0, zcr] with silhouette-based k selection)
// ---------------------------------------------------------------------------

/** Returns true when clustering succeeded AND found more than one speaker. */
function clusterTurns(turns: Turn[]): boolean {
  const voiced = turns.filter(t => t.f0 !== undefined);
  if (voiced.length < 2) return false;

  // Feature matrix: pitch dominates, zcr is a mild tiebreaker
  const logF0 = voiced.map(t => Math.log2(t.f0!));
  const zcr = voiced.map(t => t.zcr ?? 0);
  const f0Norm = zScore(logF0);
  const zcrNorm = zScore(zcr).map(v => v * 0.4);
  const points = voiced.map((_, i) => [f0Norm[i], zcrNorm[i]]);

  let best: { k: number; labels: number[]; score: number } = {
    k: 1, labels: points.map(() => 0), score: -1,
  };

  const kMax = Math.min(MAX_SPEAKERS, voiced.length);
  for (let k = 2; k <= kMax; k++) {
    const labels = kMeans(points, k);
    const clustersUsed = new Set(labels).size;
    if (clustersUsed < k) continue;
    const score = silhouette(points, labels, k);
    if (score > best.score) best = { k, labels, score };
  }

  // Weak separation = probably one voice (or indistinguishable voices)
  if (best.k === 1 || best.score < 0.45) return false;

  // Merge clusters whose median pitch is within ~1 semitone — same voice
  const labels = mergeClosePitchClusters(logF0, best.labels);
  if (new Set(labels).size < 2) return false;

  voiced.forEach((t, i) => { t.cluster = labels[i]; });

  // Turns without voice features inherit the nearest labeled neighbor in time
  for (let i = 0; i < turns.length; i++) {
    if (turns[i].cluster !== undefined) continue;
    let prev: number | undefined;
    let next: number | undefined;
    for (let j = i - 1; j >= 0; j--) if (turns[j].cluster !== undefined) { prev = turns[j].cluster; break; }
    for (let j = i + 1; j < turns.length; j++) if (turns[j].cluster !== undefined) { next = turns[j].cluster; break; }
    turns[i].cluster = prev ?? next ?? 0;
  }
  return true;
}

function zScore(xs: number[]): number[] {
  const mean = xs.reduce((a, b) => a + b, 0) / xs.length;
  const sd = Math.sqrt(xs.reduce((a, b) => a + (b - mean) ** 2, 0) / xs.length) || 1;
  return xs.map(x => (x - mean) / sd);
}

function kMeans(points: number[][], k: number): number[] {
  // Deterministic seeding: spread initial centroids across the pitch range
  const order = points.map((p, i) => ({ v: p[0], i })).sort((a, b) => a.v - b.v);
  const centroids: number[][] = [];
  for (let c = 0; c < k; c++) {
    const idx = order[Math.floor(((c + 0.5) / k) * order.length)].i;
    centroids.push([...points[idx]]);
  }

  let labels = points.map(() => 0);
  for (let iter = 0; iter < 20; iter++) {
    let changed = false;
    for (let i = 0; i < points.length; i++) {
      let bestC = 0;
      let bestD = Infinity;
      for (let c = 0; c < k; c++) {
        const d = dist2(points[i], centroids[c]);
        if (d < bestD) { bestD = d; bestC = c; }
      }
      if (labels[i] !== bestC) { labels[i] = bestC; changed = true; }
    }
    for (let c = 0; c < k; c++) {
      const members = points.filter((_, i) => labels[i] === c);
      if (!members.length) continue;
      for (let d = 0; d < points[0].length; d++) {
        centroids[c][d] = members.reduce((a, p) => a + p[d], 0) / members.length;
      }
    }
    if (!changed) break;
  }
  return labels;
}

function dist2(a: number[], b: number[]): number {
  let s = 0;
  for (let i = 0; i < a.length; i++) s += (a[i] - b[i]) ** 2;
  return s;
}

function silhouette(points: number[][], labels: number[], k: number): number {
  let total = 0;
  let counted = 0;
  for (let i = 0; i < points.length; i++) {
    const own: number[] = [];
    const other: number[][] = Array.from({ length: k }, () => []);
    for (let j = 0; j < points.length; j++) {
      if (i === j) continue;
      const d = Math.sqrt(dist2(points[i], points[j]));
      if (labels[j] === labels[i]) own.push(d);
      else other[labels[j]].push(d);
    }
    if (!own.length) continue;
    const a = own.reduce((x, y) => x + y, 0) / own.length;
    const bs = other.filter(o => o.length).map(o => o.reduce((x, y) => x + y, 0) / o.length);
    if (!bs.length) continue;
    const b = Math.min(...bs);
    total += (b - a) / Math.max(a, b);
    counted++;
  }
  return counted ? total / counted : -1;
}

function mergeClosePitchClusters(logF0: number[], labels: number[]): number[] {
  const ids = [...new Set(labels)];
  const centers = new Map<number, number>();
  for (const id of ids) {
    centers.set(id, median(logF0.filter((_, i) => labels[i] === id)));
  }
  const remap = new Map<number, number>();
  const sorted = [...ids].sort((a, b) => centers.get(a)! - centers.get(b)!);
  for (let i = 0; i < sorted.length; i++) {
    if (remap.has(sorted[i])) continue;
    remap.set(sorted[i], sorted[i]);
    for (let j = i + 1; j < sorted.length; j++) {
      if (Math.abs(centers.get(sorted[j])! - centers.get(sorted[i])!) < MIN_CLUSTER_F0_GAP_LOG2) {
        remap.set(sorted[j], sorted[i]);
      }
    }
  }
  return labels.map(l => remap.get(l)!);
}

function toSpeakerSegments(turns: Turn[]): SpeakerSegment[] {
  // Person numbers follow order of first appearance in the conversation
  const personByCluster = new Map<number, number>();
  let nextPerson = 1;
  const out: SpeakerSegment[] = [];

  for (const t of turns) {
    const cluster = t.cluster ?? 0;
    if (!personByCluster.has(cluster)) personByCluster.set(cluster, nextPerson++);
    out.push({
      speaker: `Person ${personByCluster.get(cluster)}`,
      text: t.text,
      t0: t.t0,
      t1: t.t1,
    });
  }

  // Merge consecutive turns from the same speaker for a cleaner conversation
  const merged: SpeakerSegment[] = [];
  for (const t of out) {
    const last = merged[merged.length - 1];
    if (last && last.speaker === t.speaker) {
      last.text += ' ' + t.text;
      last.t1 = t.t1;
    } else {
      merged.push({ ...t });
    }
  }
  return merged.length > 1 ? merged : [];
}
