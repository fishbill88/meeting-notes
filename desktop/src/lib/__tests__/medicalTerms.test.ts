/**
 * Tests for the medical lexicon.
 *
 * Covers the two behaviours the highlighting depends on: longest-match wins,
 * and overlapping matches merge rather than fragmenting a phrase.
 *
 * Also pins the split that keeps clinical vocabulary out of corporate
 * meetings: `findMedicalTerms` owns the lexicon and runs only for the
 * consultation template, while `highlightSpans` highlights exactly the terms
 * it is handed — so no terms means no highlighting.
 */

import { describe, expect, it } from "vitest";

import { findMedicalTerms, highlightSpans } from "../medicalTerms";

const highlighted = (text: string, extra: string[] = []) =>
  highlightSpans(text, extra)
    .filter((s) => s.highlighted)
    .map((s) => s.text);

describe("findMedicalTerms", () => {
  it("finds English clinical terms", () => {
    expect(findMedicalTerms("Patient reports chest pain and fever")).toEqual(
      expect.arrayContaining(["chest pain", "fever"]),
    );
  });

  it("finds Tagalog terms", () => {
    const found = findMedicalTerms("May lagnat at ubo po, tapos sipon");
    expect(found).toEqual(expect.arrayContaining(["lagnat", "ubo", "sipon"]));
  });

  it("finds PH brand names", () => {
    expect(findMedicalTerms("uminom ng Biogesic")).toContain("Biogesic");
  });

  it("finds dosages", () => {
    expect(findMedicalTerms("take 500 mg twice a day")).toEqual(
      expect.arrayContaining(["500 mg", "twice a day"]),
    );
  });

  it("is case insensitive but preserves the original casing", () => {
    expect(findMedicalTerms("HYPERTENSION noted")).toContain("HYPERTENSION");
  });

  it("deduplicates repeats", () => {
    const found = findMedicalTerms("fever, then more fever, still fever");
    expect(found.filter((t) => t.toLowerCase() === "fever")).toHaveLength(1);
  });

  it("does not match inside longer words", () => {
    // "uti" must not fire on "beauty".
    expect(findMedicalTerms("a beauty routine")).not.toContain("uti");
  });

  it("returns nothing for non-clinical text", () => {
    expect(findMedicalTerms("the quarterly roadmap review")).toEqual([]);
  });
});

describe("highlightSpans", () => {
  it("returns a single unhighlighted span when nothing matches", () => {
    const spans = highlightSpans("nothing clinical here", ["fever"]);
    expect(spans).toEqual([{ text: "nothing clinical here", highlighted: false }]);
  });

  it("highlights nothing when given no terms", () => {
    // The corporate guarantee: a meeting carries no medical_terms, so even
    // text full of lexicon words must render clean. Without this, "operation"
    // and "heart" light up in the middle of a project sync.
    const text = "the operation runs at the heart of Q3, no surgery needed";
    expect(highlighted(text)).toEqual([]);
    expect(highlightSpans(text)).toEqual([{ text, highlighted: false }]);
  });

  it("prefers the longest match over its substring", () => {
    // "blood pressure" must win over "blood", or the highlight fragments.
    expect(
      highlighted("check blood pressure now", ["blood", "blood pressure"]),
    ).toEqual(["blood pressure"]);
  });

  it("merges overlapping matches into one span", () => {
    const spans = highlightSpans("chest pain", ["chest pain", "pain"]);
    expect(spans.filter((s) => s.highlighted)).toHaveLength(1);
  });

  it("reassembles to the original text exactly", () => {
    const text = "Nag-reklamo ng chest pain, binigyan ng 500 mg paracetamol.";
    const rebuilt = highlightSpans(text, findMedicalTerms(text))
      .map((s) => s.text)
      .join("");
    expect(rebuilt).toBe(text);
  });

  it("highlights the lexicon terms findMedicalTerms hands it", () => {
    // The consultation path end to end: sweep, then highlight the sweep.
    const text = "Nag-reklamo ng chest pain, binigyan ng 500 mg paracetamol.";
    expect(highlighted(text, findMedicalTerms(text))).toEqual(
      expect.arrayContaining(["chest pain", "500 mg", "paracetamol"]),
    );
  });

  it("highlights extra LLM-flagged terms", () => {
    expect(highlighted("referred to Dr. Santos", ["Santos"])).toContain("Santos");
  });

  it("keeps two-character terms like bp", () => {
    expect(highlighted("bp was elevated", ["bp"])).toEqual(["bp"]);
  });

  it("ignores terms shorter than two characters", () => {
    // Guards against a stray "a" highlighting the whole transcript.
    expect(highlighted("a plain sentence", ["a"])).toEqual([]);
  });

  it("does not break on regex metacharacters in extra terms", () => {
    expect(() => highlightSpans("some text", ["c++ (v2)"])).not.toThrow();
  });

  it("handles empty text", () => {
    expect(highlightSpans("", ["fever"])).toEqual([
      { text: "", highlighted: false },
    ]);
  });
});
