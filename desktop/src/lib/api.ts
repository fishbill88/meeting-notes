/**
 * Client for the Python sidecar.
 *
 * The sidecar's port and token are unknowable until it starts, so the Rust
 * side reads them from its stdout handshake and hands them over via the
 * `sidecar_info` command. Everything here goes through {@link connect} first.
 */

import { invoke } from "@tauri-apps/api/core";

import type {
  BackendStatus,
  ExportFormat,
  LanguageMode,
  MeetingSummary,
  NotesData,
  NoteTemplate,
  ProcessResponse,
  SidecarInfo,
  StoredMeeting,
  TranscriptData,
} from "./types";

let cached: SidecarInfo | null = null;

/** Fetch (and memoise) the sidecar's connection details from the Rust side. */
export async function connect(): Promise<SidecarInfo> {
  if (cached) return cached;
  try {
    cached = await invoke<SidecarInfo>("sidecar_info");
  } catch (err) {
    // Dev fallback: the frontend is running in a plain browser (vite dev
    // server, no Tauri shell), where `invoke` cannot work. Start the sidecar
    // yourself (`python -m meeting_notes.sidecar --no-parent-watch`) and put
    // its handshake JSON in localStorage under this key.
    const raw = localStorage.getItem("sidecar_info_dev");
    if (!raw) throw err;
    cached = JSON.parse(raw) as SidecarInfo;
  }
  return cached;
}

/** Drop the memoised connection — used when the sidecar is restarted. */
export function resetConnection(): void {
  cached = null;
}

function baseUrl(info: SidecarInfo): string {
  return `http://${info.host}:${info.port}`;
}

async function request<T>(
  path: string,
  init: RequestInit = {},
): Promise<T> {
  const info = await connect();
  const headers = new Headers(init.headers);
  headers.set("Authorization", `Bearer ${info.token}`);

  const response = await fetch(`${baseUrl(info)}${path}`, {
    ...init,
    headers,
  });

  if (!response.ok) {
    throw new Error(await describeError(response));
  }
  return (await response.json()) as T;
}

async function describeError(response: Response): Promise<string> {
  // FastAPI puts the useful text in `detail`; fall back to the status line so
  // an HTML error page does not surface as "[object Object]".
  try {
    const body = await response.json();
    if (body && typeof body.detail === "string") {
      return body.detail;
    }
  } catch {
    /* not JSON */
  }
  return `${response.status} ${response.statusText}`;
}

export interface HealthResponse {
  status: string;
  version: string;
  auth: boolean;
}

/**
 * Liveness check. Unauthenticated by design, and the only call safe to make
 * before the sidecar has finished starting.
 */
export async function health(): Promise<HealthResponse> {
  const info = await connect();
  const response = await fetch(`${baseUrl(info)}/health`);
  if (!response.ok) throw new Error(`Health check failed: ${response.status}`);
  return (await response.json()) as HealthResponse;
}

/**
 * Poll `/health` until the sidecar answers.
 *
 * The first model load is slow, and silence during it reads as a hang — the
 * caller is expected to show a "starting engine" state while this runs.
 */
export async function waitForHealth(
  timeoutMs = 60_000,
  intervalMs = 250,
): Promise<HealthResponse> {
  const deadline = Date.now() + timeoutMs;
  let lastError: unknown;

  while (Date.now() < deadline) {
    try {
      return await health();
    } catch (error) {
      lastError = error;
      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
  }
  throw new Error(
    `Sidecar did not become healthy within ${timeoutMs}ms: ${String(lastError)}`,
  );
}

/** Which engines are installed. Drives the startup warning. */
export async function backends(): Promise<BackendStatus> {
  const body = await request<{ backends: BackendStatus }>("/backends");
  return body.backends;
}

export interface ProcessOptions {
  language?: LanguageMode;
  whisperModel?: string;
  backend?: string;
  template?: NoteTemplate;
  useOllama?: boolean;
  /** Label segments with speakers. Defaults to true; the sidecar silently
   *  skips it when sherpa-onnx or its models are unavailable. */
  diarize?: boolean;
}

/** Transcribe a file and generate notes in one call. */
export async function processFile(
  file: File | Blob,
  filename: string,
  options: ProcessOptions = {},
): Promise<ProcessResponse> {
  const form = new FormData();
  form.append("file", file, filename);
  if (options.language) form.append("language", options.language);
  if (options.whisperModel) form.append("whisper_model", options.whisperModel);
  if (options.backend) form.append("backend", options.backend);
  if (options.template) form.append("template", options.template);
  form.append("use_ollama", String(options.useOllama ?? true));
  form.append("diarize", String(options.diarize ?? true));

  return request<ProcessResponse>("/process", { method: "POST", body: form });
}

/**
 * Regenerate notes from transcript text.
 *
 * This is what makes the transcript editor worth having: someone fixes what
 * the engine misheard, and the notes are rebuilt from the corrected text
 * rather than from the original audio.
 */
export async function summarize(
  text: string,
  language = "en",
  template: NoteTemplate = "meeting",
  useOllama = true,
): Promise<NotesData> {
  return request<NotesData>("/summarize", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      text,
      language,
      template,
      use_ollama: useOllama,
    }),
  });
}

/* -------------------------------------------------------------------------
 * Stored meetings
 * ---------------------------------------------------------------------- */

export interface SaveMeetingInput {
  transcript: TranscriptData;
  notes: NotesData;
  durationSeconds?: number;
  speakerNames?: Record<string, string>;
  /** Pass to update an existing record instead of creating one. */
  id?: string;
}

/** Save (or update) a meeting. Content is encrypted at rest. */
export async function saveMeeting(
  input: SaveMeetingInput,
): Promise<{ id: string; created_at: string }> {
  return request("/meetings", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      transcript: input.transcript,
      notes: input.notes,
      duration_seconds: input.durationSeconds ?? 0,
      speaker_names: input.speakerNames ?? {},
      ...(input.id ? { id: input.id } : {}),
    }),
  });
}

/** History list, newest first. Pass `query` to search. */
export async function listMeetings(
  query = "",
  limit = 50,
  offset = 0,
): Promise<MeetingSummary[]> {
  const params = new URLSearchParams({
    limit: String(limit),
    offset: String(offset),
  });
  if (query.trim()) params.set("q", query.trim());
  return request<MeetingSummary[]>(`/meetings?${params.toString()}`);
}

export async function getMeeting(id: string): Promise<StoredMeeting> {
  return request<StoredMeeting>(`/meetings/${encodeURIComponent(id)}`);
}

export async function deleteMeeting(id: string): Promise<void> {
  await request(`/meetings/${encodeURIComponent(id)}`, { method: "DELETE" });
}

/* -------------------------------------------------------------------------
 * Export
 * ---------------------------------------------------------------------- */

export interface ExportResult {
  blob: Blob;
  filename: string;
}

/**
 * Render notes to a document.
 *
 * The sidecar returns bytes rather than writing a file, so a meeting never
 * lands in a temp directory on the way to wherever the user actually wants
 * it.
 */
export async function exportNotes(
  notes: NotesData,
  transcript: string | null,
  format: ExportFormat,
): Promise<ExportResult> {
  const info = await connect();
  const response = await fetch(`${baseUrl(info)}/export`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${info.token}`,
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ notes, transcript, output_format: format }),
  });

  if (!response.ok) throw new Error(await describeError(response));

  return {
    blob: await response.blob(),
    filename: response.headers.get("X-Suggested-Filename") ?? "notes",
  };
}

/** Plain text for the clipboard — the fastest path into an email or ticket. */
export async function exportText(
  notes: NotesData,
  transcript: string | null,
): Promise<string> {
  const { blob } = await exportNotes(notes, transcript, "text");
  return blob.text();
}

/** Ask the sidecar to exit. Called when the window closes. */
export async function shutdown(): Promise<void> {
  await request("/shutdown", { method: "POST" });
}
