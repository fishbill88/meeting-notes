/**
 * Editable transcript — the feature that makes the product usable.
 *
 * Tagalog conversational WER runs around 16%, and English is not perfect
 * either. Someone with a keyboard fixing the four wrong words in ninety
 * seconds beats any model improvement available this quarter, so editing is
 * the primary interaction here, not an afterthought.
 *
 * Design decisions worth keeping:
 *
 * * **Segment-level editing.** One textarea for the whole transcript loses
 *   timestamps and makes an hour-long meeting unnavigable.
 * * **Low-confidence segments are marked.** The engine already tells us where
 *   it struggled; pointing at those first is most of the value.
 * * **Edits are explicit.** An edited segment stays flagged so the user can
 *   see what they changed before regenerating notes.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { CheckIcon, PencilIcon, RefreshIcon } from "./Icons";
import { HighlightedText } from "./HighlightedText";
import type { SpeakerSegment } from "../lib/types";

/** Below this the engine was unsure enough to be worth a second look. */
const LOW_CONFIDENCE = 0.55;

interface Props {
  segments: SpeakerSegment[];
  /** Terms to highlight. Consultation-only; empty for a meeting. */
  terms?: string[];
  /** User-assigned display names, e.g. { 'Person 1': 'Dr. Cruz' }. */
  speakerNames?: Record<string, string>;
  dirty: boolean;
  regenerating: boolean;
  onChange: (segments: SpeakerSegment[]) => void;
  onRegenerate: () => void;
  /** Called with the canonical label ('Person 1') and the new display name.
   *  An empty name reverts to the canonical label. */
  onRenameSpeaker?: (speaker: string, name: string) => void;
}

function timestamp(centiseconds: number): string {
  const total = Math.floor(centiseconds / 100);
  const m = Math.floor(total / 60);
  const s = total % 60;
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}

/** Stable 1–4 colour slot for a 'Person N' label, matching mobile. */
export function speakerColorSlot(speaker: string): number {
  const n = parseInt(speaker.replace(/\D+/g, ""), 10);
  return (Number.isNaN(n) ? 0 : (n - 1) % 4) + 1;
}

export function TranscriptEditor({
  segments,
  terms = [],
  speakerNames = {},
  dirty,
  regenerating,
  onChange,
  onRegenerate,
  onRenameSpeaker,
}: Props) {
  const [editingIndex, setEditingIndex] = useState<number | null>(null);
  const [draft, setDraft] = useState("");
  const [edited, setEdited] = useState<Set<number>>(new Set());
  const [renaming, setRenaming] = useState<string | null>(null);
  const [nameDraft, setNameDraft] = useState("");
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const nameInputRef = useRef<HTMLInputElement>(null);
  // Escape unmounts the focused input, and the browser fires a synchronous
  // blur whose handler is the previous render's closure — without this flag
  // that stale onBlur would commit the very edit Escape was cancelling.
  const skipCommit = useRef(false);

  useEffect(() => {
    if (editingIndex !== null) {
      const node = textareaRef.current;
      node?.focus();
      node?.setSelectionRange(node.value.length, node.value.length);
    }
  }, [editingIndex]);

  useEffect(() => {
    if (renaming !== null) {
      nameInputRef.current?.focus();
      nameInputRef.current?.select();
    }
  }, [renaming]);

  const beginRename = (speaker: string) => {
    setRenaming(speaker);
    setNameDraft(speakerNames[speaker] ?? "");
  };

  const commitRename = () => {
    if (skipCommit.current) {
      skipCommit.current = false;
      return;
    }
    if (renaming === null) return;
    onRenameSpeaker?.(renaming, nameDraft.trim());
    setRenaming(null);
  };

  const beginEdit = (index: number) => {
    setEditingIndex(index);
    setDraft(segments[index]?.text ?? "");
  };

  const commit = useCallback(() => {
    if (skipCommit.current) {
      skipCommit.current = false;
      return;
    }
    if (editingIndex === null) return;
    const current = segments[editingIndex];
    if (!current) {
      setEditingIndex(null);
      return;
    }

    const trimmed = draft.trim();
    if (trimmed !== current.text.trim()) {
      const next = segments.map((segment, i) =>
        i === editingIndex ? { ...segment, text: trimmed } : segment,
      );
      setEdited((prev) => new Set(prev).add(editingIndex));
      onChange(next);
    }
    setEditingIndex(null);
  }, [draft, editingIndex, onChange, segments]);

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Escape") {
      event.preventDefault();
      skipCommit.current = true;
      setEditingIndex(null);
    }
    // Enter commits; Shift+Enter inserts a newline. Faster than reaching for
    // the mouse when fixing a run of segments.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      commit();
    }
  };

  if (!segments.length) {
    return <p className="muted">No transcript segments were produced.</p>;
  }

  const lowConfidence = segments.filter(
    (s) => s.confidence !== undefined && s.confidence < LOW_CONFIDENCE,
  ).length;

  return (
    <section className="panel" aria-labelledby="transcript-heading">
      <div className="panel-head">
        <h2 id="transcript-heading">Transcript</h2>
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={onRegenerate}
          disabled={!dirty || regenerating}
        >
          <RefreshIcon size={16} />
          {regenerating ? "Regenerating…" : "Update notes"}
        </button>
      </div>

      <p className="hint">
        Click any line to correct it. Enter saves, Escape cancels.
        {segments.some((s) => s.speaker) && (
          <> Click a speaker's name to rename them.</>
        )}
        {lowConfidence > 0 && (
          <>
            {" "}
            <strong>{lowConfidence}</strong>{" "}
            {lowConfidence === 1 ? "line is" : "lines are"} marked as uncertain.
          </>
        )}
      </p>

      {dirty && (
        <p className="alert alert-info" role="status">
          Transcript edited. Choose <strong>Update notes</strong> to rebuild the
          notes from your corrections.
        </p>
      )}

      <ol className="segments">
        {segments.map((segment, index) => {
          const uncertain =
            segment.confidence !== undefined && segment.confidence < LOW_CONFIDENCE;
          // A new speaker turn starts here: label it. Unlabelled transcripts
          // (diarization off or a monologue) never enter this branch.
          const turnStart =
            !!segment.speaker && segment.speaker !== segments[index - 1]?.speaker;

          return (
            <li
              key={`${segment.t0}-${index}`}
              className="segment-row"
              data-uncertain={uncertain}
              data-edited={edited.has(index)}
              data-speaker-slot={segment.speaker ? speakerColorSlot(segment.speaker) : undefined}
            >
              {turnStart && segment.speaker && (
                <div className="speaker-turn">
                  {renaming === segment.speaker ? (
                    <input
                      ref={nameInputRef}
                      className="speaker-name-input"
                      value={nameDraft}
                      placeholder={segment.speaker}
                      onChange={(e) => setNameDraft(e.target.value)}
                      onBlur={commitRename}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          e.preventDefault();
                          commitRename();
                        }
                        if (e.key === "Escape") {
                          e.preventDefault();
                          skipCommit.current = true;
                          setRenaming(null);
                        }
                      }}
                      aria-label={`Rename ${segment.speaker}`}
                    />
                  ) : (
                    <button
                      type="button"
                      className="speaker-name"
                      onClick={() => beginRename(segment.speaker!)}
                      title="Click to rename this speaker"
                    >
                      {speakerNames[segment.speaker] ?? segment.speaker}
                      <PencilIcon size={12} className="segment-pencil" />
                    </button>
                  )}
                </div>
              )}
              <span className="segment-time">{timestamp(segment.t0)}</span>

              {editingIndex === index ? (
                <textarea
                  ref={textareaRef}
                  className="segment-input"
                  value={draft}
                  rows={Math.max(1, Math.ceil(draft.length / 70))}
                  onChange={(e) => setDraft(e.target.value)}
                  onBlur={commit}
                  onKeyDown={onKeyDown}
                  aria-label={`Edit line at ${timestamp(segment.t0)}`}
                />
              ) : (
                <button
                  type="button"
                  className="segment-text"
                  onClick={() => beginEdit(index)}
                  aria-label={`Edit line at ${timestamp(segment.t0)}`}
                >
                  <HighlightedText text={segment.text} terms={terms} />
                  <PencilIcon size={14} className="segment-pencil" />
                </button>
              )}

              <span className="segment-flags">
                {edited.has(index) && (
                  <span className="flag flag-edited">
                    <CheckIcon size={12} /> edited
                  </span>
                )}
                {uncertain && !edited.has(index) && (
                  <span className="flag flag-uncertain">uncertain</span>
                )}
                {segment.lang && segment.lang !== "en" && (
                  <span className="flag">{segment.lang}</span>
                )}
              </span>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

/**
 * Rebuild plain transcript text from segments, for re-summarisation.
 *
 * When segments carry speakers, each turn is prefixed with the speaker's
 * display name on its own line — the summariser attributes statements
 * correctly, and mobile's `formatConversation` does the same for its LLM.
 */
export function segmentsToText(
  segments: SpeakerSegment[],
  speakerNames: Record<string, string> = {},
): string {
  // One entry per speaker turn (or one total when nothing is labelled).
  const turns: { speaker: string | null; texts: string[] }[] = [];

  for (const segment of segments) {
    const text = segment.text.trim();
    if (!text) continue;

    const speaker = segment.speaker ?? null;
    const last = turns[turns.length - 1];
    if (last && last.speaker === speaker) {
      last.texts.push(text);
    } else {
      turns.push({ speaker, texts: [text] });
    }
  }

  return turns
    .map(({ speaker, texts }) => {
      const line = texts.join(" ");
      if (!speaker) return line;
      return `${speakerNames[speaker] ?? speaker}: ${line}`;
    })
    .join("\n");
}
