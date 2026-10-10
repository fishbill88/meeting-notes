/**
 * Export actions.
 *
 * Copy-to-clipboard is first and visually primary: pasting into an email,
 * ticket or wiki page is the fastest route from this app to wherever the
 * notes are actually going, and it is what most people will reach for. File
 * export exists for the cases where a document is genuinely wanted.
 */

import { useState } from "react";

import { CheckIcon, FileAudioIcon } from "./Icons";
import { exportNotes, exportText } from "../lib/api";
import type { ExportFormat, NotesData } from "../lib/types";

const FILE_FORMATS: { format: ExportFormat; label: string }[] = [
  { format: "docx", label: "Word (.docx)" },
  { format: "markdown", label: "Markdown (.md)" },
];

interface Props {
  notes: NotesData;
  transcript: string | null;
  /** Include the transcript in the exported document. */
  includeTranscript: boolean;
  onIncludeTranscriptChange: (include: boolean) => void;
}

export function ExportBar({
  notes,
  transcript,
  includeTranscript,
  onIncludeTranscriptChange,
}: Props) {
  const [copied, setCopied] = useState(false);
  const [busy, setBusy] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const body = includeTranscript ? transcript : null;

  const copy = async () => {
    setError(null);
    setBusy("copy");
    try {
      await navigator.clipboard.writeText(await exportText(notes, body));
      setCopied(true);
      window.setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  const download = async (format: ExportFormat) => {
    setError(null);
    setBusy(format);
    try {
      const { blob, filename } = await exportNotes(notes, body, format);

      // Object-URL download rather than a native save dialog: that would need
      // the Tauri fs plugin and a capability grant. Worth upgrading, but this
      // works in the webview today and keeps the bytes in memory.
      const url = URL.createObjectURL(blob);
      const anchor = document.createElement("a");
      anchor.href = url;
      anchor.download = filename;
      document.body.appendChild(anchor);
      anchor.click();
      anchor.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(null);
    }
  };

  return (
    <div className="export-bar">
      <div className="export-actions">
        <button
          type="button"
          className="btn btn-primary btn-sm"
          onClick={copy}
          disabled={busy !== null}
        >
          {copied ? <CheckIcon size={16} /> : <FileAudioIcon size={16} />}
          {copied ? "Copied" : "Copy notes"}
        </button>

        {FILE_FORMATS.map(({ format, label }) => (
          <button
            key={format}
            type="button"
            className="btn btn-secondary btn-sm"
            onClick={() => void download(format)}
            disabled={busy !== null}
          >
            {busy === format ? "Preparing…" : label}
          </button>
        ))}
      </div>

      <label className="checkbox">
        <input
          type="checkbox"
          checked={includeTranscript}
          onChange={(event) => onIncludeTranscriptChange(event.target.checked)}
        />
        Include the full transcript
      </label>

      {error && (
        <p className="alert alert-error" role="alert">
          {error}
        </p>
      )}
    </div>
  );
}
