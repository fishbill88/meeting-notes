/**
 * Saved meetings: list, search, open, delete.
 *
 * Search runs on the sidecar, which decrypts as it scans — so it is debounced
 * rather than fired per keystroke. Deletion is permanent and there is no
 * undo, so it asks first.
 */

import { useCallback, useEffect, useRef, useState } from "react";

import { AlertIcon, TrashIcon } from "./Icons";
import { deleteMeeting, listMeetings } from "../lib/api";
import type { MeetingSummary } from "../lib/types";

const SEARCH_DEBOUNCE_MS = 250;

interface Props {
  onOpen: (id: string) => void;
  /** Changes whenever a meeting is saved, to trigger a refresh. */
  refreshToken: number;
}

function formatWhen(iso: string): string {
  const date = new Date(iso);
  if (Number.isNaN(date.getTime())) return iso;
  return date.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatLength(seconds: number): string {
  if (!seconds) return "";
  const minutes = Math.round(seconds / 60);
  return minutes < 1 ? "under a minute" : `${minutes} min`;
}

export function HistoryPanel({ onOpen, refreshToken }: Props) {
  const [query, setQuery] = useState("");
  const [rows, setRows] = useState<MeetingSummary[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [pendingDelete, setPendingDelete] = useState<string | null>(null);
  const requestId = useRef(0);

  const load = useCallback(async (search: string) => {
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    try {
      const results = await listMeetings(search);
      // Drop responses from searches the user has already typed past.
      if (id === requestId.current) setRows(results);
    } catch (err) {
      if (id === requestId.current) {
        setError(err instanceof Error ? err.message : String(err));
      }
    } finally {
      if (id === requestId.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const handle = window.setTimeout(() => void load(query), SEARCH_DEBOUNCE_MS);
    return () => window.clearTimeout(handle);
  }, [query, load, refreshToken]);

  const confirmDelete = async (id: string) => {
    try {
      await deleteMeeting(id);
      setRows((prev) => prev.filter((row) => row.id !== id));
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setPendingDelete(null);
    }
  };

  return (
    <section className="panel" aria-labelledby="history-heading">
      <div className="panel-head">
        <h2 id="history-heading">Saved meetings</h2>
      </div>

      <div className="field">
        <label htmlFor="history-search">Search</label>
        <input
          id="history-search"
          type="search"
          value={query}
          placeholder="Search titles, notes and transcripts"
          onChange={(event) => setQuery(event.target.value)}
        />
      </div>

      {error && (
        <p className="alert alert-error" role="alert">
          <AlertIcon size={16} />
          <span>{error}</span>
        </p>
      )}

      {loading && rows.length === 0 && <p className="muted">Loading…</p>}

      {!loading && rows.length === 0 && (
        <p className="muted empty">
          {query
            ? `Nothing matches “${query}”.`
            : "No meetings saved yet. Record one and it will appear here."}
        </p>
      )}

      <ul className="history-list">
        {rows.map((row) => (
          <li key={row.id} className="history-row">
            <button
              type="button"
              className="history-open"
              onClick={() => onOpen(row.id)}
            >
              <span className="history-title">{row.title}</span>
              <span className="history-meta">
                {formatWhen(row.created_at)}
                {row.duration_seconds > 0 && <> · {formatLength(row.duration_seconds)}</>}
                {row.language && <> · {row.language}</>}
              </span>
              {row.preview && <span className="history-preview">{row.preview}</span>}
            </button>

            {pendingDelete === row.id ? (
              <span className="history-confirm">
                <button
                  type="button"
                  className="btn btn-sm btn-stop"
                  onClick={() => void confirmDelete(row.id)}
                >
                  Delete permanently
                </button>
                <button
                  type="button"
                  className="btn btn-sm btn-secondary"
                  onClick={() => setPendingDelete(null)}
                >
                  Cancel
                </button>
              </span>
            ) : (
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={() => setPendingDelete(row.id)}
                aria-label={`Delete ${row.title}`}
              >
                <TrashIcon size={16} />
              </button>
            )}
          </li>
        ))}
      </ul>
    </section>
  );
}
