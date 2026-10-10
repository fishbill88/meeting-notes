/**
 * Structured notes.
 *
 * Shows which engine produced the transcript. That is not a debugging detail:
 * the Tagalog router silently degrades to Whisper when the NeMo model has not
 * been exported, and anyone deciding whether to trust a note should be able
 * to see which engine wrote it.
 *
 * The term highlighting and the term list below are consultation-only —
 * `medical_terms` is empty for a meeting, so both vanish.
 */

import { HighlightedText } from "./HighlightedText";
import { ShieldIcon } from "./Icons";
import type { NotesData, TranscriptData } from "../lib/types";

interface Props {
  notes: NotesData;
  transcript: TranscriptData;
}

const ENGINE_LABELS: Record<string, string> = {
  "faster-whisper": "Whisper",
  "nemo-tagalog": "Tagalog FastConformer",
  "omnilingual-ctc": "Meta Omnilingual",
};

/** Languages that have a dedicated engine; Whisper handling them is a fallback. */
const DEDICATED_ENGINE_LANGUAGES: Record<string, string> = {
  tl: "Tagalog",
  ceb: "Cebuano",
};

export function NotesView({ notes, transcript }: Props) {
  const terms = notes.medical_terms ?? [];
  const engine = transcript.backend
    ? (ENGINE_LABELS[transcript.backend] ?? transcript.backend)
    : null;

  const dedicated = DEDICATED_ENGINE_LANGUAGES[transcript.language];
  const degraded = dedicated !== undefined && transcript.backend === "faster-whisper";

  return (
    <section className="panel" aria-labelledby="notes-heading">
      <div className="panel-head">
        <h2 id="notes-heading">{notes.title || "Meeting notes"}</h2>
      </div>

      <p className="provenance">
        <ShieldIcon size={14} />
        {transcript.language_name}
        {engine && <> · transcribed by {engine}</>}
      </p>

      {degraded && (
        <p className="alert alert-warn" role="status">
          This was transcribed by Whisper, not the dedicated {dedicated} engine.
          Accuracy is lower than it could be — review carefully.
        </p>
      )}

      <h3>Summary</h3>
      <p className="prose">
        <HighlightedText text={notes.summary} terms={terms} />
      </p>

      <NoteList title="Key topics" items={notes.key_topics} terms={terms} />
      <NoteList title="Decisions" items={notes.decisions} terms={terms} />
      <NoteList title="Action items" items={notes.action_items} terms={terms} />

      {terms.length > 0 && (
        <>
          <h3>Medical terms detected</h3>
          <ul className="term-chips">
            {terms.map((term) => (
              <li key={term} className="term-chip">
                {term}
              </li>
            ))}
          </ul>
        </>
      )}
    </section>
  );
}

function NoteList({
  title,
  items,
  terms,
}: {
  title: string;
  items: string[];
  terms: string[];
}) {
  if (!items.length) return null;
  return (
    <>
      <h3>{title}</h3>
      <ul className="prose-list">
        {items.map((item, index) => (
          <li key={index}>
            <HighlightedText text={item} terms={terms} />
          </li>
        ))}
      </ul>
    </>
  );
}
