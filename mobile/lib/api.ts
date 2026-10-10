import AsyncStorage from '@react-native-async-storage/async-storage';
import { MeetingRecord, NotesData, TranscriptData } from './types';

const BACKEND_URL_KEY = 'backend_url';
export const DEFAULT_BACKEND_URL = 'http://localhost:8811';

export async function getBackendUrl(): Promise<string> {
  return (await AsyncStorage.getItem(BACKEND_URL_KEY)) ?? DEFAULT_BACKEND_URL;
}

export async function saveBackendUrl(url: string): Promise<void> {
  await AsyncStorage.setItem(BACKEND_URL_KEY, url.replace(/\/$/, ''));
}

export async function checkHealth(): Promise<boolean> {
  try {
    const url = await getBackendUrl();
    const res = await fetch(`${url}/health`, {
      signal: AbortSignal.timeout(4000),
    });
    return res.ok;
  } catch {
    return false;
  }
}

export async function processAudio(
  audioUri: string,
  opts: {
    whisperModel?: string;
    language?: string | null;
    useOllama?: boolean;
    ollamaModel?: string;
    onStep?: (step: string) => void;
  } = {}
): Promise<{ transcript: TranscriptData; notes: NotesData }> {
  const url = await getBackendUrl();
  opts.onStep?.('Uploading audio…');

  const form = new FormData();
  form.append('file', {
    uri: audioUri,
    name: 'recording.m4a',
    type: 'audio/m4a',
  } as unknown as Blob);
  form.append('whisper_model', opts.whisperModel ?? 'base');
  form.append('use_ollama', opts.useOllama === false ? 'false' : 'true');
  if (opts.language) form.append('language', opts.language);
  if (opts.ollamaModel) form.append('ollama_model', opts.ollamaModel);

  opts.onStep?.('Transcribing with Whisper…');
  const res = await fetch(`${url}/process`, {
    method: 'POST',
    body: form,
  });

  if (!res.ok) {
    const text = await res.text().catch(() => 'Unknown error');
    throw new Error(`Server error ${res.status}: ${text}`);
  }

  opts.onStep?.('Generating meeting notes…');
  return res.json();
}

// ---------------------------------------------------------------------------
// Local meeting history stored in AsyncStorage
// ---------------------------------------------------------------------------

const HISTORY_KEY = 'meeting_history';

export async function saveMeeting(record: MeetingRecord): Promise<void> {
  const history = await loadHistory();
  history.unshift(record);
  await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history.slice(0, 50)));
}

export async function loadHistory(): Promise<MeetingRecord[]> {
  const raw = await AsyncStorage.getItem(HISTORY_KEY);
  return raw ? JSON.parse(raw) : [];
}

export async function deleteMeeting(id: string): Promise<void> {
  const history = await loadHistory();
  await AsyncStorage.setItem(
    HISTORY_KEY,
    JSON.stringify(history.filter((m) => m.id !== id))
  );
}

/** Replace a stored meeting in place (e.g., after renaming speakers). */
export async function updateMeeting(record: MeetingRecord): Promise<void> {
  const history = await loadHistory();
  const idx = history.findIndex((m) => m.id === record.id);
  if (idx >= 0) {
    history[idx] = record;
    await AsyncStorage.setItem(HISTORY_KEY, JSON.stringify(history));
  }
}
