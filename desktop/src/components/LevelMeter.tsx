/**
 * Live input level.
 *
 * Its job is answering "is the microphone actually hearing us?" before
 * someone records twenty minutes of silence — so it shows a decayed peak,
 * warns when the signal is too quiet to transcribe, and flags clipping.
 */

import { memo } from "react";

const SEGMENTS = 24;

/** Below this the signal is usually too quiet for reliable transcription. */
const QUIET_THRESHOLD = 0.02;

/** At or above this the input is clipping and will distort. */
const CLIP_THRESHOLD = 0.98;

interface Props {
  /** Current peak, 0–1. */
  level: number;
  active: boolean;
}

function LevelMeterInner({ level, active }: Props) {
  // Perceptual rather than linear: speech sits low in a linear scale and the
  // meter would look dead through normal conversation.
  const scaled = active ? Math.min(1, Math.sqrt(level)) : 0;
  const lit = Math.round(scaled * SEGMENTS);

  const status = !active
    ? "idle"
    : level >= CLIP_THRESHOLD
      ? "clipping"
      : level < QUIET_THRESHOLD
        ? "quiet"
        : "ok";

  return (
    <div className="meter" data-status={status}>
      <div
        className="meter-bars"
        role="meter"
        aria-label="Input level"
        aria-valuenow={Math.round(scaled * 100)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        {Array.from({ length: SEGMENTS }, (_, i) => (
          <span
            key={i}
            className="meter-bar"
            data-lit={i < lit}
            data-zone={i > SEGMENTS - 3 ? "hot" : i > SEGMENTS - 7 ? "warm" : "normal"}
          />
        ))}
      </div>

      {/* Never colour-only: the status is spelled out in words too. */}
      <p className="meter-status" role="status">
        {status === "clipping" && "Too loud — move the mic further away"}
        {status === "quiet" && "Very quiet — check the microphone"}
        {status === "ok" && "Receiving audio"}
        {status === "idle" && "Not recording"}
      </p>
    </div>
  );
}

export const LevelMeter = memo(LevelMeterInner);
