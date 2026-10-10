/**
 * Tests for WAV encoding and duration formatting.
 *
 * The WAV encoder is the highest-risk piece of the capture path: a wrong
 * header field or byte order produces a file that looks fine and transcribes
 * to silence. These assertions read the bytes back rather than trusting it.
 */

import { describe, expect, it } from "vitest";

import { encodeWav, formatDuration } from "../recorder";

async function bytes(blob: Blob): Promise<DataView> {
  return new DataView(await blob.arrayBuffer());
}

function ascii(view: DataView, offset: number, length: number): string {
  let out = "";
  for (let i = 0; i < length; i += 1) {
    out += String.fromCharCode(view.getUint8(offset + i));
  }
  return out;
}

describe("encodeWav", () => {
  it("writes a valid RIFF/WAVE header", async () => {
    const view = await bytes(encodeWav(new Float32Array(100), 16_000));

    expect(ascii(view, 0, 4)).toBe("RIFF");
    expect(ascii(view, 8, 4)).toBe("WAVE");
    expect(ascii(view, 12, 4)).toBe("fmt ");
    expect(ascii(view, 36, 4)).toBe("data");
  });

  it("declares 16-bit mono PCM at the given sample rate", async () => {
    const view = await bytes(encodeWav(new Float32Array(10), 16_000));

    expect(view.getUint32(16, true)).toBe(16); // fmt chunk size
    expect(view.getUint16(20, true)).toBe(1); // PCM
    expect(view.getUint16(22, true)).toBe(1); // mono
    expect(view.getUint32(24, true)).toBe(16_000); // sample rate
    expect(view.getUint32(28, true)).toBe(32_000); // byte rate = rate * 2
    expect(view.getUint16(32, true)).toBe(2); // block align
    expect(view.getUint16(34, true)).toBe(16); // bits per sample
  });

  it("sizes the chunks to the sample count", async () => {
    const samples = new Float32Array(1_000);
    const view = await bytes(encodeWav(samples, 16_000));

    expect(view.getUint32(40, true)).toBe(2_000); // data bytes
    expect(view.getUint32(4, true)).toBe(36 + 2_000); // RIFF size
    expect(view.byteLength).toBe(44 + 2_000);
  });

  it("converts floats to little-endian 16-bit samples", async () => {
    const view = await bytes(
      encodeWav(Float32Array.from([0, 1, -1, 0.5]), 16_000),
    );

    expect(view.getInt16(44, true)).toBe(0);
    expect(view.getInt16(46, true)).toBe(32_767);
    expect(view.getInt16(48, true)).toBe(-32_768);
    expect(view.getInt16(50, true)).toBeCloseTo(16_383, -1);
  });

  it("clamps out-of-range samples instead of wrapping", async () => {
    // Without clamping these overflow and become loud noise at the opposite
    // polarity — an audible, and easily missed, corruption.
    const view = await bytes(encodeWav(Float32Array.from([4, -4]), 16_000));

    expect(view.getInt16(44, true)).toBe(32_767);
    expect(view.getInt16(46, true)).toBe(-32_768);
  });

  it("produces a header-only file for empty input", async () => {
    const blob = encodeWav(new Float32Array(0), 16_000);
    expect(blob.size).toBe(44);
    expect(blob.type).toBe("audio/wav");
  });
});

describe("formatDuration", () => {
  it("formats under an hour as mm:ss", () => {
    expect(formatDuration(0)).toBe("00:00");
    expect(formatDuration(9)).toBe("00:09");
    expect(formatDuration(65)).toBe("01:05");
    expect(formatDuration(3_599)).toBe("59:59");
  });

  it("adds hours past 60 minutes", () => {
    expect(formatDuration(3_600)).toBe("1:00:00");
    expect(formatDuration(3_725)).toBe("1:02:05");
  });

  it("floors fractional seconds and clamps negatives", () => {
    expect(formatDuration(1.9)).toBe("00:01");
    expect(formatDuration(-5)).toBe("00:00");
  });
});
