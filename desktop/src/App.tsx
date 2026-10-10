/**
 * Application shell.
 *
 * Flow: record (or import) → transcribe → review and correct → notes → saved.
 *
 * The transcript is the editable source of truth; notes are derived from it.
 * Correcting the transcript rebuilds the notes from the corrected text rather
 * than from the original audio.
 *
 * Meetings are saved automatically once processed. Losing a meeting because
 * someone forgot to press Save is not an acceptable failure mode, and the
 * store is local and encrypted, so there is nothing to weigh against it.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { ExportBar } from "./components/ExportBar";
import { HistoryPanel } from "./components/HistoryPanel";
import { NotesView } from "./components/NotesView";
import { RecordPanel } from "./components/RecordPanel";
import {
  TranscriptEditor,
  segmentsToText,
} from "./components/TranscriptEditor";
import { AlertIcon, ShieldIcon, TrashIcon } from "./components/Icons";
import {
  backends,
  getMeeting,
  processFile,
  saveMeeting,
  summarize,
  waitForHealth,
} from "./lib/api";
import { findMedicalTerms } from "./lib/medicalTerms";
import type { Recording } from "./lib/recorder";
import type {
  BackendStatus,
  LanguageMode,
  NotesData,
  NoteTemplate,
  SpeakerSegment,
  TranscriptData,
} from "./lib/types";

type Startup =
  | { state: "starting" }
  | { state: "ready"; engines: BackendStatus }
  | { state: "failed"; message: string };

type View = "record" | "history";

interface Session {
  id: string | null;
  transcript: TranscriptData;
  notes: NotesData;
  segments: SpeakerSegment[];
  /** User-assigned display names keyed by canonical label ('Person 1'). */
  speakerNames: Record<string, string>;
  durationSeconds: number;
  dirty: boolean;
  saveError: string | null;
}

export default function App() {
  const [startup, setStartup] = useState<Startup>({ state: "starting" });
  const [view, setView] = useState<View>("record");
  const [language, setLanguage] = useState<LanguageMode>("auto");
  // The note-format picker is hidden for now: corporate meetings are the whole
  // product surface today. The template still flows through to the sidecar and
  // to `enrich()`, so restoring the picker is a UI change only.
  const template: NoteTemplate = "meeting";
  const [includeTranscript, setIncludeTranscript] = useState(true);
  const [busy, setBusy] = useState<string | null>(null);
  const [regenerating, setRegenerating] = useState(false);
  const [session, setSession] = useState<Session | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savedCount, setSavedCount] = useState(0);

  useEffect(() => {
    let cancelled = false;

    (async () => {
      try {
        await waitForHealth();
        const engines = await backends();
        if (!cancelled) setStartup({ state: "ready", engines });
      } catch (err) {
        if (!cancelled) setStartup({ state: "failed", message: String(err) });
      }
    })();

    return () => {
      cancelled = true;
    };
  }, []);

  /** Persist, and remember the id so later edits update rather than duplicate. */
  const persist = useCallback(async (next: Session): Promise<Session> => {
    try {
      const { id } = await saveMeeting({
        transcript: next.transcript,
        notes: next.notes,
        durationSeconds: next.durationSeconds,
        speakerNames: next.speakerNames,
        ...(next.id ? { id: next.id } : {}),
      });
      setSavedCount((n) => n + 1);
      return { ...next, id, saveError: null };
    } catch (err) {
      // A failed save must not discard the work — keep it on screen and say so.
      return {
        ...next,
        saveError: err instanceof Error ? err.message : String(err),
      };
    }
  }, []);

  const transcribe = useCallback(
    async (blob: Blob, filename: string, durationSeconds = 0) => {
      setBusy("Transcribing and generating notes…");
      setError(null);
      setSession(null);
      setView("record");
      try {
        const result = await processFile(blob, filename, { language, template });
        const next: Session = {
          id: null,
          transcript: result.transcript,
          notes: enrich(result.notes, result.transcript.text, template),
          segments: result.transcript.segments ?? [],
          speakerNames: {},
          durationSeconds,
          dirty: false,
          saveError: null,
        };
        setSession(await persist(next));
      } catch (err) {
        setError(err instanceof Error ? err.message : String(err));
      } finally {
        setBusy(null);
      }
    },
    [language, template, persist],
  );

  const onRecorded = useCallback(
    (recording: Recording) => {
      void transcribe(recording.blob, "meeting.wav", recording.durationSeconds);
    },
    [transcribe],
  );

  const onFilePicked = useCallback(
    (file: File) => void transcribe(file, file.name),
    [transcribe],
  );

  const onSegmentsChange = useCallback((segments: SpeakerSegment[]) => {
    setSession((prev) => (prev ? { ...prev, segments, dirty: true } : prev));
  }, []);

  /** Guards rename saves: a slow older save must not clobber a newer one. */
  const renameSeq = useRef(0);

  /** Rename persists immediately — a name is metadata, not a transcript edit,
   *  so it should survive without a "Update notes" round-trip. */
  const onRenameSpeaker = useCallback(
    (speaker: string, name: string) => {
      if (!session) return;
      const speakerNames = { ...session.speakerNames };
      if (name) {
        speakerNames[speaker] = name;
      } else {
        delete speakerNames[speaker];
      }
      const next = { ...session, speakerNames };
      setSession(next);
      const seq = ++renameSeq.current;
      void persist(next).then((saved) => {
        if (renameSeq.current === seq) setSession(saved);
      });
    },
    [session, persist],
  );

  const onRegenerate = useCallback(async () => {
    if (!session) return;

    setRegenerating(true);
    setError(null);
    try {
      const text = segmentsToText(session.segments, session.speakerNames);
      const notes = await summarize(text, session.transcript.language, template);
      const next: Session = {
        ...session,
        transcript: { ...session.transcript, text },
        notes: enrich(notes, text, template),
        dirty: false,
      };
      setSession(await persist(next));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setRegenerating(false);
    }
  }, [session, template, persist]);

  const openSaved = useCallback(async (id: string) => {
    setBusy("Opening…");
    setError(null);
    try {
      const record = await getMeeting(id);
      setSession({
        id: record.id,
        transcript: record.transcript,
        notes: record.notes,
        segments: record.transcript.segments ?? [],
        speakerNames: record.speaker_names ?? {},
        durationSeconds: record.duration_seconds,
        dirty: false,
        saveError: null,
      });
      setView("record");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  }, []);

  if (startup.state === "starting") {
    return (
      <main className="centered">
        <div className="startup">
          <span className="spinner" aria-hidden="true" />
          <h1>Meeting Notes AI</h1>
          <p className="muted">Starting the transcription engine…</p>
          <p className="hint">
            The first launch loads the speech model and can take a minute.
          </p>
        </div>
      </main>
    );
  }

  if (startup.state === "failed") {
    return (
      <main className="centered">
        <div className="startup">
          <AlertIcon size={32} />
          <h1>Could not start</h1>
          <p className="muted">
            The transcription service did not respond. Restarting the app
            usually fixes this.
          </p>
          <pre className="alert alert-error">{startup.message}</pre>
        </div>
      </main>
    );
  }

  const tagalogReady = startup.engines["nemo-tagalog"] === true;
  const cebuanoReady = startup.engines["omnilingual-ctc"] === true;
  const transcriptText = session
    ? session.transcript.timed_transcript || session.transcript.text
    : null;

  return (
    <>
      <header className="app-header">
        <div className="app-header-inner">
          <h1>Meeting Notes AI</h1>
          <nav className="tabs" aria-label="Sections">
            <button
              type="button"
              className="tab"
              aria-current={view === "record"}
              onClick={() => setView("record")}
            >
              Record
            </button>
            <button
              type="button"
              className="tab"
              aria-current={view === "history"}
              onClick={() => setView("history")}
            >
              History
            </button>
          </nav>
          <p className="privacy">
            <ShieldIcon size={14} />
            On-device — no audio leaves this computer
          </p>
        </div>
      </header>

      <main>
        {!tagalogReady && (
          <p className="alert alert-warn" role="status">
            <AlertIcon size={16} />
            <span>
              The Tagalog engine is not installed, so Tagalog will be
              transcribed by Whisper at lower accuracy. Run{" "}
              <code>python scripts/export_nemo_to_onnx.py</code> to install it.
            </span>
          </p>
        )}

        {view === "history" ? (
          <HistoryPanel onOpen={openSaved} refreshToken={savedCount} />
        ) : (
          <RecordPanel
            busy={busy !== null}
            language={language}
            onLanguageChange={setLanguage}
            onRecorded={onRecorded}
            onFilePicked={onFilePicked}
            tagalogReady={tagalogReady}
            cebuanoReady={cebuanoReady}
          />
        )}

        {busy && (
          <p className="alert alert-info" role="status">
            <span className="spinner spinner-sm" aria-hidden="true" />
            {busy}
          </p>
        )}

        {error && (
          <p className="alert alert-error" role="alert">
            <AlertIcon size={16} />
            <span>{error}</span>
          </p>
        )}

        {view === "record" && session && (
          <>
            {session.saveError && (
              <p className="alert alert-warn" role="alert">
                <AlertIcon size={16} />
                <span>
                  This meeting is on screen but was not saved:{" "}
                  {session.saveError}. Export it before closing the app.
                </span>
              </p>
            )}

            <TranscriptEditor
              segments={session.segments}
              terms={session.notes.medical_terms}
              speakerNames={session.speakerNames}
              dirty={session.dirty}
              regenerating={regenerating}
              onChange={onSegmentsChange}
              onRegenerate={onRegenerate}
              onRenameSpeaker={onRenameSpeaker}
            />

            <NotesView notes={session.notes} transcript={session.transcript} />

            <ExportBar
              notes={session.notes}
              transcript={transcriptText}
              includeTranscript={includeTranscript}
              onIncludeTranscriptChange={setIncludeTranscript}
            />

            <div className="session-actions">
              <button
                type="button"
                className="btn btn-ghost"
                onClick={() => setSession(null)}
              >
                <TrashIcon size={16} />
                Close this meeting
              </button>
            </div>
          </>
        )}
      </main>
    </>
  );
}

/**
 * Merge lexicon matches into whatever the LLM flagged.
 *
 * The extractive fallback flags nothing at all, so without this the
 * highlighting would silently disappear whenever Ollama is unavailable.
 *
 * Consultation-only, and deliberately so: this is the single place the
 * medical lexicon runs, and `medical_terms` is in turn the only thing
 * `highlightSpans` will highlight. Skipping it for a meeting is what keeps
 * clinical vocabulary out of a corporate transcript — the backend's "meeting"
 * template never asks for `medical_terms` either.
 */
function enrich(
  notes: NotesData,
  transcriptText: string,
  template: NoteTemplate,
): NotesData {
  if (template !== "consultation") return notes;
  const found = findMedicalTerms(transcriptText);
  const merged = new Set([...(notes.medical_terms ?? []), ...found]);
  return { ...notes, medical_terms: [...merged] };
}
