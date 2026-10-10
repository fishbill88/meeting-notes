import { describe, expect, it } from "vitest";

import { segmentsToText, speakerColorSlot } from "../TranscriptEditor";
import type { SpeakerSegment } from "../../lib/types";

const seg = (
  text: string,
  t0: number,
  speaker?: string | null,
): SpeakerSegment => ({ text, t0, t1: t0 + 100, speaker });

describe("segmentsToText", () => {
  it("joins plain segments with spaces, as before diarization", () => {
    const text = segmentsToText([seg("Hello.", 0), seg("Hi there.", 100)]);
    expect(text).toBe("Hello. Hi there.");
  });

  it("prefixes each speaker turn once, not every segment", () => {
    const text = segmentsToText([
      seg("Masakit ang dibdib ko.", 0, "Person 1"),
      seg("Tatlong araw na.", 100, "Person 1"),
      seg("Gaano kadalas?", 200, "Person 2"),
    ]);
    expect(text).toBe(
      "Person 1: Masakit ang dibdib ko. Tatlong araw na.\nPerson 2: Gaano kadalas?",
    );
  });

  it("uses assigned display names", () => {
    const text = segmentsToText(
      [seg("Kumusta po.", 0, "Person 1"), seg("Mabuti naman.", 100, "Person 2")],
      { "Person 1": "Dr. Cruz" },
    );
    expect(text).toBe("Dr. Cruz: Kumusta po.\nPerson 2: Mabuti naman.");
  });

  it("starts a new turn when the same speaker resumes after another", () => {
    const text = segmentsToText([
      seg("First.", 0, "Person 1"),
      seg("Reply.", 100, "Person 2"),
      seg("Again.", 200, "Person 1"),
    ]);
    expect(text).toBe("Person 1: First.\nPerson 2: Reply.\nPerson 1: Again.");
  });

  it("skips empty segments without breaking turns", () => {
    const text = segmentsToText([
      seg("Start.", 0, "Person 1"),
      seg("  ", 100, "Person 1"),
      seg("End.", 200, "Person 1"),
    ]);
    expect(text).toBe("Person 1: Start. End.");
  });
});

describe("speakerColorSlot", () => {
  it("maps Person 1–4 to slots 1–4", () => {
    expect(speakerColorSlot("Person 1")).toBe(1);
    expect(speakerColorSlot("Person 4")).toBe(4);
  });

  it("wraps around after four speakers", () => {
    expect(speakerColorSlot("Person 5")).toBe(1);
  });

  it("tolerates labels without a number", () => {
    expect(speakerColorSlot("Narrator")).toBe(1);
  });
});
