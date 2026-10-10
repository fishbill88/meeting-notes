/**
 * Wire contract shared with the mobile app.
 *
 * Kept in sync with `mobile/lib/types.ts` by hand — deliberately duplicated
 * rather than shared through a package, because the two apps ship on different
 * release cadences and a breaking change should surface as a type error here
 * rather than silently at runtime there.
 */

export interface SpeakerSegment {
  /** 'Person 1', 'Person 2', … from diarization. Absent/null when unlabelled:
   *  diarization off, unavailable, or a monologue (flat transcript shown). */
  speaker?: string | null;
  text: string;
  t0: number; // centiseconds
  t1: number;
  /** ISO code of the language recognised in this segment. */
  lang?: string;
  /** Engine confidence 0–1, when the engine reports one. Not comparable
   *  across engines. Populated for the Taglish two-pass merge. */
  confidence?: number;
}

export interface TranscriptData {
  text: string;
  language: string;
  language_name: string;
  timed_transcript: string;
  speakers?: SpeakerSegment[];
  segments?: SpeakerSegment[];
  /** Which ASR engine produced this — e.g. 'faster-whisper', 'nemo-tagalog'.
   *  Surfaced in the UI because the Tagalog router falls back to Whisper when
   *  the NeMo model has not been exported. */
  backend?: string;
}

export interface NotesData {
  title: string;
  summary: string;
  key_topics: string[];
  decisions: string[];
  action_items: string[];
  language: string;
  conversation?: SpeakerSegment[];
  /**
   * Medical/health terms found by the lexicon + flagged by the LLM.
   * Consultation template only — empty for meetings.
   */
  medical_terms?: string[];
}

export interface MeetingRecord {
  id: string;
  created_at: string;
  duration_seconds: number;
  transcript: TranscriptData;
  notes: NotesData;
  /** User-assigned display names, e.g. { 'Person 1': 'Dr. Cruz' } */
  speakerNames?: Record<string, string>;
}

/* -------------------------------------------------------------------------
 * Desktop-only
 * ---------------------------------------------------------------------- */

/** What the Rust side learned from the sidecar's stdout handshake. */
export interface SidecarInfo {
  host: string;
  port: number;
  token: string;
  pid: number;
}

export interface ProcessResponse {
  transcript: TranscriptData;
  notes: NotesData;
}

/** Engine availability, as reported by GET /backends. */
export type BackendStatus = Record<string, boolean>;

/** Language modes the backend implements. `taglish` is deliberately absent.
 *  `ceb` (Cebuano / Bisaya) routes to the Omnilingual engine. */
export type LanguageMode = "en" | "tl" | "ceb" | "auto";

/** Note templates. Mirrors `TEMPLATES` in summarizer.py. */
export type NoteTemplate = "meeting" | "consultation";

/** Export formats. Mirrors `EXPORT_FORMATS` in document.py. */
export type ExportFormat = "markdown" | "docx" | "text";

/** A row in the history list — decrypted, but without the full transcript. */
export interface MeetingSummary {
  id: string;
  created_at: string;
  duration_seconds: number;
  language: string;
  backend: string;
  title: string;
  preview: string;
}

/** A saved meeting as returned by GET /meetings/{id}. */
export interface StoredMeeting {
  id: string;
  created_at: string;
  duration_seconds: number;
  transcript: TranscriptData;
  notes: NotesData;
  speaker_names?: Record<string, string>;
}
