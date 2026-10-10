/**
 * Live captions while recording.
 *
 * The native WavRecorder keeps writing the full-quality WAV; it also emits
 * each PCM buffer to JS, where a whisper.rn RealtimeTranscriber consumes the
 * stream and transcribes slice by slice. The final notes still come from the
 * full-file pass after recording stops — live text is preview only.
 *
 * To keep up with speech on a phone, captions deliberately trade accuracy
 * for speed: the smallest downloaded Whisper model, greedy decoding, short
 * slices, and a multi-second re-transcription throttle.
 */
// NOTE: deliberately importing via src/ — whisper.rn 0.6.0's exports map points
// this subpath at a lib/module build that isn't shipped, so Metro can't resolve
// the "official" specifier. The src/ path is the README-sanctioned fallback.
import { RealtimeTranscriber } from 'whisper.rn/src/realtime-transcription';
import type {
  AudioStreamConfig,
  AudioStreamData,
  AudioStreamInterface,
  RealtimeTranscribeEvent,
} from 'whisper.rn/src/realtime-transcription';
import { initWhisper, WhisperContext } from 'whisper.rn';
import { base64ToBytes } from './bytes';
import {
  WHISPER_MODELS,
  getActiveWhisperModel,
  isModelDownloaded,
  modelPath,
} from './models';
import { RECORDER_SAMPLE_RATE, subscribeAudioChunks } from './wavRecorder';
import { getWhisperContext } from './whisper';

/** Taps the native recorder's PCM chunk events — never touches the mic itself. */
class WavRecorderStreamAdapter implements AudioStreamInterface {
  private unsubscribe?: () => void;
  private recording = false;
  private dataCallback?: (data: AudioStreamData) => void;
  private errorCallback?: (error: string) => void;
  private statusCallback?: (isRecording: boolean) => void;

  async initialize(_config: AudioStreamConfig): Promise<void> {}

  async start(): Promise<void> {
    if (this.recording) return;
    this.unsubscribe = subscribeAudioChunks((b64) => {
      try {
        this.dataCallback?.({
          data: base64ToBytes(b64),
          sampleRate: RECORDER_SAMPLE_RATE,
          channels: 1,
          timestamp: Date.now(),
        });
      } catch (err) {
        this.errorCallback?.(err instanceof Error ? err.message : String(err));
      }
    });
    this.recording = true;
    this.statusCallback?.(true);
  }

  async stop(): Promise<void> {
    this.unsubscribe?.();
    this.unsubscribe = undefined;
    this.recording = false;
    this.statusCallback?.(false);
  }

  isRecording(): boolean {
    return this.recording;
  }

  onData(callback: (data: AudioStreamData) => void): void {
    this.dataCallback = callback;
  }

  onError(callback: (error: string) => void): void {
    this.errorCallback = callback;
  }

  onStatusChange(callback: (isRecording: boolean) => void): void {
    this.statusCallback = callback;
  }

  async release(): Promise<void> {
    await this.stop();
  }
}

/**
 * Captions don't need the accuracy of the user's selected model — they need
 * to keep up. Use the smallest downloaded multilingual model; fall back to
 * the shared (selected) context when nothing smaller is on the device.
 */
async function getLiveWhisperContext(): Promise<{ ctx: WhisperContext; dedicated: boolean }> {
  const active = await getActiveWhisperModel();
  for (const key of ['TINY', 'BASE']) {
    const m = WHISPER_MODELS[key];
    if (m.sizeMB < active.sizeMB && (await isModelDownloaded(m.filename, m.sizeMB))) {
      const ctx = await initWhisper({ filePath: modelPath(m.filename) });
      return { ctx, dedicated: true };
    }
  }
  return { ctx: await getWhisperContext(), dedicated: false };
}

export interface LiveSession {
  stop: () => Promise<void>;
}

/**
 * Start live transcription off the active recording session.
 * Call AFTER startWavRecording(true). onText receives the rolling caption text.
 */
export async function startLiveTranscription(
  onText: (text: string) => void,
  opts: { language?: string } = {},
): Promise<LiveSession> {
  const { ctx: whisperContext, dedicated } = await getLiveWhisperContext();

  const sliceTexts = new Map<number, string>();
  const emit = () => {
    const ordered = [...sliceTexts.entries()].sort((a, b) => a[0] - b[0]);
    onText(ordered.map(([, t]) => t).join(' ').replace(/\s+/g, ' ').trim());
  };

  const transcriber = new RealtimeTranscriber(
    {
      whisperContext,
      audioStream: new WavRecorderStreamAdapter(),
    },
    {
      // Short slices cap how much audio each pass re-transcribes; sealed
      // slices are never re-processed.
      audioSliceSec: 10,
      audioMinSec: 1,
      maxSlicesInMemory: 3,
      // Without VAD the transcriber re-transcribes the growing slice on every
      // chunk (default pause 200ms) — on a phone that snowballs into lag.
      // One pass every ~3s keeps captions close to real time.
      realtimeProcessingPauseMs: 3000,
      initRealtimeAfterMs: 1500,
      transcribeOptions: {
        language: opts.language,
        translate: false,
        // Greedy decode — fastest path through whisper.cpp
        beamSize: 1,
        bestOf: 1,
        temperature: 0,
        maxThreads: 4,
      },
    },
    {
      onTranscribe: (event: RealtimeTranscribeEvent) => {
        const text = event.data?.result?.trim();
        if (text) {
          sliceTexts.set(event.sliceIndex, text);
          emit();
        }
      },
      onError: () => {
        // Live captions are best-effort; the full pass still runs at stop.
      },
    },
  );

  await transcriber.start();

  return {
    stop: async () => {
      try {
        await transcriber.stop();
      } finally {
        await transcriber.release().catch(() => {});
        if (dedicated) await whisperContext.release().catch(() => {});
      }
    },
  };
}
