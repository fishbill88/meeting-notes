/**
 * Captures raw PCM from the microphone graph and forwards it to the main
 * thread.
 *
 * Served from `public/` rather than built from a blob URL on purpose: the app
 * runs under a `default-src 'self'` CSP, which blocks blob-backed worklets.
 *
 * The AudioContext is created at 16 kHz, so the frames arriving here are
 * already at the rate the transcription pipeline expects — no resampling.
 */
class PcmCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this._running = true;
    this.port.onmessage = (event) => {
      if (event.data === "stop") this._running = false;
    };
  }

  process(inputs) {
    const input = inputs[0];
    if (!input || !input.length) return this._running;

    const channel = input[0];
    if (!channel || !channel.length) return this._running;

    // Peak rather than RMS: a level meter exists to show the mic is live and
    // not clipping, and peaks are what clip.
    let peak = 0;
    for (let i = 0; i < channel.length; i += 1) {
      const value = channel[i] < 0 ? -channel[i] : channel[i];
      if (value > peak) peak = value;
    }

    // Copy — the underlying buffer is reused by the audio thread.
    this.port.postMessage({ samples: new Float32Array(channel), peak });
    return this._running;
  }
}

registerProcessor("pcm-capture", PcmCaptureProcessor);
