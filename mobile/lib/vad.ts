/**
 * Voice activity detection (Silero VAD via whisper.rn).
 *
 * Detects where speech actually happens in a recording so that:
 *  - transcription skips silent stretches (whisper.ts) — the big speed win
 *  - diarization can split turns on real, precisely-timed silences (diarize.ts)
 *
 * All timestamps are centiseconds (100 = 1s), matching whisper.rn segments.
 */
import * as FileSystem from 'expo-file-system/legacy';
import { initWhisperVad, WhisperVadContext } from 'whisper.rn';
import { ensureVadModel } from './models';

/** Speech region in centiseconds. */
export interface SpeechRegion {
  t0: number;
  t1: number;
}

// Regions closer than this are merged: Whisper encodes 30s windows anyway, so
// skipping sub-2s silences saves nothing and only adds per-call overhead.
const MERGE_GAP_CS = 200;
// If speech covers most of the recording, region-by-region transcription has
// no benefit — signal callers to do a plain full-file pass.
const MAX_SPEECH_COVERAGE = 0.85;
// WAV produced by wavRecorder: 16 kHz mono 16-bit PCM, 44-byte header.
const WAV_HEADER_BYTES = 44;
const WAV_BYTES_PER_SEC = 16000 * 2;

let _ctx: WhisperVadContext | null = null;

async function getVadContext(): Promise<WhisperVadContext | null> {
  if (_ctx) return _ctx;
  const path = await ensureVadModel();
  if (!path) return null;
  _ctx = await initWhisperVad({ filePath: path });
  return _ctx;
}

export function releaseVad(): void {
  _ctx?.release().catch(() => {});
  _ctx = null;
}

/**
 * Detects speech regions in a WAV file.
 *
 * Returns null when the caller should transcribe the whole file instead:
 * VAD model unavailable, detection failed, no speech found (a VAD false
 * negative must not eat the user's recording), or speech covers nearly the
 * entire recording.
 */
export async function detectSpeechRegions(wavPath: string): Promise<SpeechRegion[] | null> {
  try {
    const ctx = await getVadContext();
    if (!ctx) return null;

    const bare = wavPath.replace(/^file:\/\//, '');
    const segments = await ctx.detectSpeech(bare, {
      threshold: 0.5,
      minSpeechDurationMs: 250,
      minSilenceDurationMs: 300,
      // Generous padding so region edges never clip word onsets/offsets
      speechPadMs: 200,
    });
    if (!segments.length) return null;

    // Merge near-adjacent segments into contiguous regions
    const regions: SpeechRegion[] = [{ ...segments[0] }];
    for (let i = 1; i < segments.length; i++) {
      const last = regions[regions.length - 1];
      if (segments[i].t0 - last.t1 <= MERGE_GAP_CS) {
        last.t1 = Math.max(last.t1, segments[i].t1);
      } else {
        regions.push({ ...segments[i] });
      }
    }

    // Coverage check against the recording's real duration
    const info = await FileSystem.getInfoAsync(
      wavPath.startsWith('file://') ? wavPath : 'file://' + wavPath,
    );
    const bytes: number = (info as any).size ?? 0;
    const durationCs = ((bytes - WAV_HEADER_BYTES) / WAV_BYTES_PER_SEC) * 100;
    const speechCs = regions.reduce((acc, r) => acc + (r.t1 - r.t0), 0);
    if (durationCs > 0 && speechCs / durationCs > MAX_SPEECH_COVERAGE) return null;

    return regions;
  } catch {
    return null;
  }
}

/**
 * VAD-measured silence (cs) inside [fromCs, toCs] — time not covered by any
 * speech region. Far more precise than gaps between Whisper segment
 * timestamps, which drift by design.
 */
export function silenceBetween(regions: SpeechRegion[], fromCs: number, toCs: number): number {
  if (toCs <= fromCs) return 0;
  let speech = 0;
  for (const r of regions) {
    const overlap = Math.min(r.t1, toCs) - Math.max(r.t0, fromCs);
    if (overlap > 0) speech += overlap;
  }
  return Math.max(0, toCs - fromCs - speech);
}
