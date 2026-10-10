export interface SpeakerSegment {
  speaker: string;   // 'Person 1', 'Person 2', etc.
  text: string;
  t0: number;        // centiseconds
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
  /** Medical/health terms found by the lexicon + flagged by the LLM. */
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
