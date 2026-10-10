import { initWhisper, WhisperContext } from 'whisper.rn';
import { getActiveWhisperModel, modelPath } from './models';
import { detectSpeechRegions, SpeechRegion } from './vad';

let _ctx: WhisperContext | null = null;
let _ctxFilename: string | null = null;

export async function getWhisperContext(): Promise<WhisperContext> {
  const model = await getActiveWhisperModel();
  if (_ctx && _ctxFilename === model.filename) return _ctx;

  if (_ctx) { await _ctx.release().catch(() => {}); _ctx = null; _ctxFilename = null; }
  try {
    _ctx = await initWhisper({ filePath: modelPath(model.filename) });
    _ctxFilename = model.filename;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    if (msg.includes('install') || msg.includes('null') || msg.includes('native')) {
      throw new Error(
        'Whisper native module unavailable.\n' +
        'This app must run as a custom dev build, not Expo Go.\n' +
        'Run: npx expo run:android  (or run:ios)',
      );
    }
    throw err;
  }
  return _ctx;
}

export function releaseWhisper(): void {
  _ctx?.release();
  _ctx = null;
  _ctxFilename = null;
}

export interface TranscribeOutput {
  text: string;
  language: string;
  segments: Array<{ text: string; t0: number; t1: number }>;
  /** VAD speech regions used to skip silence (cs) — absent on full-file passes. */
  speechRegions?: SpeechRegion[];
}

// whisper.rn caps its default thread count at min(4, cores) — and uses only 2
// on quad-core devices — so the full-quality pass leaves big cores idle on
// modern 6–8 core phones. Requesting more threads is a free speedup with no
// accuracy cost (unlike shrinking the model). 6 is a safe bet for octa-core
// phones without oversubscribing typical hexa-core devices too badly; lower it
// if you target older quad-core hardware.
const FULL_PASS_THREADS = 6;

export async function transcribeAudio(
  audioUri: string,
  opts: {
    language?: string | null;
    onProgress?: (pct: number) => void;
  } = {},
): Promise<TranscribeOutput> {
  const ctx = await getWhisperContext();

  // strip file:// if present — whisper.rn needs a bare path on Android
  const filePath = audioUri.replace(/^file:\/\//, '');

  try {
    // Silence skipping: transcribe only VAD-detected speech regions, so the
    // Whisper pass scales with speech time instead of recording time.
    // null = no benefit or VAD unavailable — do the plain full-file pass.
    const regions = await detectSpeechRegions(filePath);
    if (!regions) {
      const res = await transcribeRange(ctx, filePath, opts.language, undefined, opts.onProgress);
      return {
        text: (res.result ?? '').trim(),
        language: opts.language ?? res.language ?? 'auto',
        segments: res.segments ?? [],
      };
    }

    const totalCs = regions.reduce((acc, r) => acc + (r.t1 - r.t0), 0);
    let doneCs = 0;
    // Lock language after the first region so auto-detect can't flip-flop
    // mid-meeting (and later regions skip detection entirely).
    let language = opts.language ?? undefined;
    const texts: string[] = [];
    const segments: TranscribeOutput['segments'] = [];

    for (const region of regions) {
      const regionCs = region.t1 - region.t0;
      const res = await transcribeRange(ctx, filePath, language, region, (p) =>
        opts.onProgress?.((doneCs + p * regionCs) / totalCs),
      );
      doneCs += regionCs;
      const text = (res.result ?? '').trim();
      if (text) texts.push(text);
      // `offset` seeks in the source file, so segment timestamps come back
      // absolute — no re-basing needed.
      segments.push(...(res.segments ?? []));
      if (!language && res.language && res.language !== 'auto') language = res.language;
    }

    return {
      text: texts.join(' '),
      language: language ?? 'auto',
      segments,
      speechRegions: regions,
    };
  } catch (err) {
    if (err instanceof Error) throw err;
    const e = err as Record<string, unknown>;
    throw new Error(String(e?.message ?? e?.error ?? JSON.stringify(err)));
  }
}

/** One ctx.transcribe call, optionally windowed to a speech region (cs). */
async function transcribeRange(
  ctx: WhisperContext,
  filePath: string,
  language: string | null | undefined,
  region?: { t0: number; t1: number },
  onProgress?: (fraction: number) => void,
) {
  const { promise } = ctx.transcribe(filePath, {
    language: language ?? undefined,
    translate: false,
    maxThreads: FULL_PASS_THREADS,
    // whisper.cpp takes milliseconds; regions are centiseconds
    offset: region ? Math.round(region.t0 * 10) : undefined,
    duration: region ? Math.round((region.t1 - region.t0) * 10) : undefined,
    onProgress: onProgress ? (p: number) => onProgress(p / 100) : undefined,
  });
  return promise;
}
