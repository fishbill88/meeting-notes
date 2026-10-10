/**
 * Microphone and system-audio capture, producing 16 kHz mono WAV.
 *
 * Recording happens in the webview rather than in Rust. That keeps the audio
 * path in TypeScript (no native module to maintain per platform) and the
 * WebView2/WKWebView Web Audio implementations are more than adequate for
 * 16 kHz speech.
 *
 * WAV rather than the browser-native webm/opus: the pipeline can decode opus
 * via PyAV, but lossy compression before ASR is a quality risk that buys us
 * nothing on a local disk.
 */

const TARGET_SAMPLE_RATE = 16_000;
const WORKLET_URL = "/pcm-worklet.js";

export type RecordingSource = "microphone" | "system";

export interface RecorderOptions {
  /** Called roughly every 8 ms with the current peak level, 0–1. */
  onLevel?: (peak: number) => void;
  /** Called about once a second with elapsed seconds. */
  onTick?: (seconds: number) => void;
  deviceId?: string;
  source?: RecordingSource;
}

export interface Recording {
  blob: Blob;
  durationSeconds: number;
  sampleRate: number;
}

/** Thrown when the user denies permission or no input device exists. */
export class RecorderError extends Error {
  constructor(message: string, readonly cause?: unknown) {
    super(message);
    this.name = "RecorderError";
  }
}

export async function listInputDevices(): Promise<MediaDeviceInfo[]> {
  // Labels are empty until permission has been granted at least once, so the
  // caller should list again after the first successful start().
  const devices = await navigator.mediaDevices.enumerateDevices();
  return devices.filter((d) => d.kind === "audioinput");
}

export class Recorder {
  private context: AudioContext | null = null;
  private stream: MediaStream | null = null;
  private node: AudioWorkletNode | null = null;
  private source: MediaStreamAudioSourceNode | null = null;
  private chunks: Float32Array[] = [];
  private frames = 0;
  private tickHandle: number | null = null;
  private startedAt = 0;
  private pausedFor = 0;
  private pausedAt: number | null = null;

  get isRecording(): boolean {
    return this.context !== null;
  }

  get isPaused(): boolean {
    return this.pausedAt !== null;
  }

  async start(options: RecorderOptions = {}): Promise<void> {
    if (this.context) throw new RecorderError("Already recording");

    const stream = await this.acquireStream(options);

    // Requesting the context at 16 kHz makes the whole graph run there, so
    // the worklet emits frames at the rate the pipeline wants.
    const context = new AudioContext({ sampleRate: TARGET_SAMPLE_RATE });
    try {
      await context.audioWorklet.addModule(WORKLET_URL);
    } catch (error) {
      stream.getTracks().forEach((t) => t.stop());
      await context.close();
      throw new RecorderError(
        "Could not load the audio worklet. Is public/pcm-worklet.js being served?",
        error,
      );
    }

    const node = new AudioWorkletNode(context, "pcm-capture");
    node.port.onmessage = (event: MessageEvent) => {
      const data = event.data as { samples: Float32Array; peak: number };
      if (this.pausedAt !== null) return;
      this.chunks.push(data.samples);
      this.frames += data.samples.length;
      options.onLevel?.(data.peak);
    };

    const sourceNode = context.createMediaStreamSource(stream);
    sourceNode.connect(node);
    // Not connected to the destination: routing the mic to the speakers would
    // cause feedback in a meeting room.

    this.context = context;
    this.stream = stream;
    this.node = node;
    this.source = sourceNode;
    this.chunks = [];
    this.frames = 0;
    this.startedAt = performance.now();
    this.pausedFor = 0;
    this.pausedAt = null;

    if (options.onTick) {
      this.tickHandle = window.setInterval(() => {
        options.onTick?.(Math.floor(this.elapsedSeconds()));
      }, 500);
    }
  }

  private async acquireStream(options: RecorderOptions): Promise<MediaStream> {
    try {
      if (options.source === "system") {
        // Remote meetings: capture the far end from the conferencing app
        // (Zoom, Teams, Meet). The user picks the window or screen; Chromium
        // offers its audio on Windows. The video track is discarded
        // immediately.
        const display = await navigator.mediaDevices.getDisplayMedia({
          video: true,
          audio: true,
        });
        display.getVideoTracks().forEach((track) => {
          track.stop();
          display.removeTrack(track);
        });
        if (!display.getAudioTracks().length) {
          throw new RecorderError(
            "That source has no audio. Re-share and tick “Share system audio”.",
          );
        }
        return display;
      }

      return await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
          ...(options.deviceId ? { deviceId: { exact: options.deviceId } } : {}),
        },
      });
    } catch (error) {
      if (error instanceof RecorderError) throw error;
      const name = error instanceof Error ? error.name : "";
      if (name === "NotAllowedError") {
        throw new RecorderError(
          "Microphone access was denied. Grant it in system settings and try again.",
          error,
        );
      }
      if (name === "NotFoundError") {
        throw new RecorderError("No microphone was found.", error);
      }
      throw new RecorderError("Could not start recording.", error);
    }
  }

  pause(): void {
    if (!this.context || this.pausedAt !== null) return;
    this.pausedAt = performance.now();
  }

  resume(): void {
    if (!this.context || this.pausedAt === null) return;
    this.pausedFor += performance.now() - this.pausedAt;
    this.pausedAt = null;
  }

  elapsedSeconds(): number {
    if (!this.context) return 0;
    const paused =
      this.pausedFor + (this.pausedAt ? performance.now() - this.pausedAt : 0);
    return (performance.now() - this.startedAt - paused) / 1000;
  }

  /** Stop, tear down the graph, and return the recorded audio as WAV. */
  async stop(): Promise<Recording> {
    if (!this.context) throw new RecorderError("Not recording");

    const context = this.context;
    const sampleRate = context.sampleRate;

    this.node?.port.postMessage("stop");
    this.node?.disconnect();
    this.source?.disconnect();
    this.stream?.getTracks().forEach((track) => track.stop());
    if (this.tickHandle !== null) window.clearInterval(this.tickHandle);
    await context.close();

    const samples = concat(this.chunks, this.frames);

    this.context = null;
    this.stream = null;
    this.node = null;
    this.source = null;
    this.chunks = [];
    this.tickHandle = null;
    this.pausedAt = null;

    return {
      blob: encodeWav(samples, sampleRate),
      durationSeconds: samples.length / sampleRate,
      sampleRate,
    };
  }

  /** Abandon a recording without producing a file. */
  async cancel(): Promise<void> {
    if (!this.context) return;
    await this.stop().catch(() => undefined);
  }
}

function concat(chunks: Float32Array[], total: number): Float32Array {
  const out = new Float32Array(total);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.length;
  }
  return out;
}

/**
 * Encode float samples as 16-bit PCM WAV — the format `audio.py` writes and
 * every backend reads natively.
 */
export function encodeWav(samples: Float32Array, sampleRate: number): Blob {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);

  const writeString = (offset: number, value: string) => {
    for (let i = 0; i < value.length; i += 1) {
      view.setUint8(offset + i, value.charCodeAt(i));
    }
  };

  const dataBytes = samples.length * 2;
  writeString(0, "RIFF");
  view.setUint32(4, 36 + dataBytes, true);
  writeString(8, "WAVE");
  writeString(12, "fmt ");
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); // byte rate
  view.setUint16(32, 2, true); // block align
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, "data");
  view.setUint32(40, dataBytes, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    const sample = Math.max(-1, Math.min(1, samples[i] ?? 0));
    view.setInt16(offset, sample < 0 ? sample * 0x8000 : sample * 0x7fff, true);
    offset += 2;
  }

  return new Blob([buffer], { type: "audio/wav" });
}

/** `mm:ss`, or `h:mm:ss` past an hour. */
export function formatDuration(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds));
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`;
}
