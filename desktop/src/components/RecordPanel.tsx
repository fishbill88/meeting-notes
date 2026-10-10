/**
 * The primary screen: start a meeting recording.
 *
 * Record-first, not file-first. The file picker stays available because it is
 * genuinely useful for re-processing, but it is secondary — the product is
 * "press record when the meeting starts".
 */

import { useCallback, useEffect, useRef, useState } from "react";

import {
  MicIcon,
  MonitorIcon,
  PauseIcon,
  PlayIcon,
  StopIcon,
  FileAudioIcon,
} from "./Icons";
import { LevelMeter } from "./LevelMeter";
import {
  Recorder,
  RecorderError,
  formatDuration,
  listInputDevices,
  type Recording,
  type RecordingSource,
} from "../lib/recorder";
import type { LanguageMode } from "../lib/types";

const LANGUAGES: { value: LanguageMode; label: string; hint: string }[] = [
  { value: "auto", label: "Auto-detect", hint: "Whisper picks the language" },
  { value: "en", label: "English", hint: "Whisper, forced to English" },
  { value: "tl", label: "Tagalog", hint: "Dedicated Tagalog engine" },
  {
    value: "ceb",
    label: "Bisaya",
    hint: "Cebuano via Meta Omnilingual — first use downloads ~290 MB",
  },
];

interface Props {
  busy: boolean;
  language: LanguageMode;
  onLanguageChange: (language: LanguageMode) => void;
  onRecorded: (recording: Recording) => void;
  onFilePicked: (file: File) => void;
  tagalogReady: boolean;
  cebuanoReady: boolean;
}

export function RecordPanel({
  busy,
  language,
  onLanguageChange,
  onRecorded,
  onFilePicked,
  tagalogReady,
  cebuanoReady,
}: Props) {
  const recorder = useRef(new Recorder());
  const [recording, setRecording] = useState(false);
  const [paused, setPaused] = useState(false);
  const [level, setLevel] = useState(0);
  const [elapsed, setElapsed] = useState(0);
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([]);
  const [deviceId, setDeviceId] = useState<string>("");
  const [source, setSource] = useState<RecordingSource>("microphone");
  const [error, setError] = useState<string | null>(null);

  const refreshDevices = useCallback(async () => {
    try {
      setDevices(await listInputDevices());
    } catch {
      /* enumeration is best-effort; the default device still works */
    }
  }, []);

  useEffect(() => {
    void refreshDevices();
  }, [refreshDevices]);

  // Stop the capture if the component unmounts mid-recording, so the mic
  // indicator does not stay lit.
  useEffect(() => {
    const active = recorder.current;
    return () => {
      void active.cancel();
    };
  }, []);

  const start = async () => {
    setError(null);
    try {
      await recorder.current.start({
        source,
        deviceId: source === "microphone" && deviceId ? deviceId : undefined,
        onLevel: setLevel,
        onTick: setElapsed,
      });
      setRecording(true);
      setPaused(false);
      setElapsed(0);
      // Labels are only populated after permission is granted.
      void refreshDevices();
    } catch (err) {
      setError(
        err instanceof RecorderError ? err.message : "Could not start recording.",
      );
    }
  };

  const stop = async () => {
    try {
      const result = await recorder.current.stop();
      setRecording(false);
      setPaused(false);
      setLevel(0);

      if (result.durationSeconds < 1) {
        setError("That recording was too short to transcribe.");
        return;
      }
      onRecorded(result);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  };

  const togglePause = () => {
    if (paused) {
      recorder.current.resume();
      setPaused(false);
    } else {
      recorder.current.pause();
      setPaused(true);
      setLevel(0);
    }
  };

  return (
    <section className="panel" aria-labelledby="record-heading">
      <h2 id="record-heading" className="visually-hidden">
        Record a meeting
      </h2>

      <div className="record-stage">
        <div className="timer" aria-live="off">
          {formatDuration(elapsed)}
        </div>

        <LevelMeter level={level} active={recording && !paused} />

        <div className="record-actions">
          {!recording ? (
            <button
              type="button"
              className="btn btn-record"
              onClick={start}
              disabled={busy}
            >
              <MicIcon size={22} />
              Start recording
            </button>
          ) : (
            <>
              <button
                type="button"
                className="btn btn-secondary"
                onClick={togglePause}
                aria-label={paused ? "Resume recording" : "Pause recording"}
              >
                {paused ? <PlayIcon /> : <PauseIcon />}
                {paused ? "Resume" : "Pause"}
              </button>
              <button type="button" className="btn btn-stop" onClick={stop}>
                <StopIcon />
                Stop and transcribe
              </button>
            </>
          )}
        </div>

        {recording && (
          <p className="recording-badge" role="status">
            <span className="pulse" aria-hidden="true" />
            {paused ? "Paused" : "Recording"}
          </p>
        )}
      </div>

      {error && (
        <p className="alert alert-error" role="alert">
          {error}
        </p>
      )}

      <div className="settings-grid">
        <fieldset disabled={recording || busy}>
          <legend>Language</legend>
          <div className="segmented" role="radiogroup" aria-label="Language">
            {LANGUAGES.map((option) => (
              <label key={option.value} className="segment">
                <input
                  type="radio"
                  name="language"
                  value={option.value}
                  checked={language === option.value}
                  onChange={() => onLanguageChange(option.value)}
                />
                <span>{option.label}</span>
              </label>
            ))}
          </div>
          <p className="hint">
            {LANGUAGES.find((l) => l.value === language)?.hint}
            {language === "tl" && !tagalogReady && " — not installed, will use Whisper"}
            {language === "ceb" &&
              !cebuanoReady &&
              " — engine not installed, will use Whisper as Tagalog"}
          </p>
        </fieldset>

        <fieldset disabled={recording || busy}>
          <legend>Source</legend>
          <div className="segmented" role="radiogroup" aria-label="Audio source">
            <label className="segment">
              <input
                type="radio"
                name="source"
                checked={source === "microphone"}
                onChange={() => setSource("microphone")}
              />
              <span>
                <MicIcon size={16} /> Microphone
              </span>
            </label>
            <label className="segment">
              <input
                type="radio"
                name="source"
                checked={source === "system"}
                onChange={() => setSource("system")}
              />
              <span>
                <MonitorIcon size={16} /> System audio
              </span>
            </label>
          </div>
          <p className="hint">
            {source === "system"
              ? "For calls on Zoom, Teams or Meet — you will be asked which window to capture."
              : "For everyone in the same room."}
          </p>
        </fieldset>

        {source === "microphone" && devices.length > 1 && (
          <div className="field">
            <label htmlFor="device">Microphone</label>
            <select
              id="device"
              value={deviceId}
              onChange={(e) => setDeviceId(e.target.value)}
              disabled={recording || busy}
            >
              <option value="">System default</option>
              {devices.map((device, index) => (
                <option key={device.deviceId} value={device.deviceId}>
                  {device.label || `Input ${index + 1}`}
                </option>
              ))}
            </select>
          </div>
        )}
      </div>

      <details className="import">
        <summary>
          <FileAudioIcon size={16} />
          Transcribe an existing recording instead
        </summary>
        <div className="field">
          <label htmlFor="audio-file">Audio file</label>
          <input
            id="audio-file"
            type="file"
            accept="audio/*"
            disabled={busy || recording}
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) onFilePicked(file);
              event.target.value = "";
            }}
          />
        </div>
      </details>
    </section>
  );
}
