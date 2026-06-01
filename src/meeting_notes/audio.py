"""Audio recording module – captures microphone input or loads from a file."""

from __future__ import annotations

import io
import os
import queue
import tempfile
import threading
import wave
from pathlib import Path
from typing import Optional

import numpy as np

from meeting_notes.config import DEFAULT_CHANNELS, DEFAULT_SAMPLE_RATE


class AudioRecorder:
    """Records audio from the system microphone in a background thread.

    Usage::

        recorder = AudioRecorder()
        recorder.start()
        # … wait while meeting is in progress …
        audio_path = recorder.stop()
        # audio_path is a WAV file ready for Whisper transcription
    """

    def __init__(
        self,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        channels: int = DEFAULT_CHANNELS,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self._frames: list[np.ndarray] = []
        self._recording = False
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Begin recording audio from the default microphone."""
        try:
            import sounddevice as sd  # noqa: PLC0415 – deferred import
        except ImportError as exc:
            raise RuntimeError(
                "sounddevice is required for microphone recording. "
                "Install it with:  pip install sounddevice"
            ) from exc

        if self._recording:
            raise RuntimeError("Recording is already in progress.")

        self._frames = []
        self._recording = True
        self._error = None

        def _capture() -> None:
            try:
                with sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="float32",
                ) as stream:
                    while self._recording:
                        data, _ = stream.read(self.sample_rate // 10)  # 100 ms chunks
                        self._frames.append(data.copy())
            except Exception as exc:  # noqa: BLE001
                self._error = exc
                self._recording = False

        self._thread = threading.Thread(target=_capture, daemon=True)
        self._thread.start()

    def stop(self, output_path: Optional[str] = None) -> str:
        """Stop recording and write the captured audio to a WAV file.

        Args:
            output_path: Optional path for the WAV file.  If not provided a
                temporary file is created.

        Returns:
            Absolute path to the WAV file.
        """
        if not self._recording:
            raise RuntimeError("No recording is in progress.")

        self._recording = False
        if self._thread is not None:
            self._thread.join(timeout=5)

        if self._error is not None:
            raise RuntimeError(f"Audio capture failed: {self._error}") from self._error

        if not self._frames:
            raise RuntimeError("No audio was captured.")

        audio = np.concatenate(self._frames, axis=0)
        return _save_wav(audio, self.sample_rate, self.channels, output_path)

    @property
    def is_recording(self) -> bool:
        return self._recording


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _save_wav(
    audio: np.ndarray,
    sample_rate: int,
    channels: int,
    output_path: Optional[str] = None,
) -> str:
    """Convert a float32 NumPy array to a 16-bit PCM WAV file."""
    if output_path is None:
        fd, output_path = tempfile.mkstemp(suffix=".wav")
        os.close(fd)

    # Whisper expects 16-bit PCM
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)

    with wave.open(output_path, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit = 2 bytes
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

    return output_path


def load_audio_file(path: str) -> tuple[np.ndarray, int]:
    """Load an audio file and return (float32 array, sample_rate).

    Supports WAV natively; other formats require ``ffmpeg`` (used by Whisper
    internally).

    Args:
        path: Path to the audio file.

    Returns:
        Tuple of (audio array normalised to [-1, 1], sample rate in Hz).
    """
    path = str(Path(path).resolve())
    if not os.path.exists(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    if path.lower().endswith(".wav"):
        with wave.open(path, "rb") as wf:
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
            sample_width = wf.getsampwidth()
            n_channels = wf.getnchannels()
            sample_rate = wf.getframerate()

        dtype = {1: np.int8, 2: np.int16, 4: np.int32}.get(sample_width, np.int16)
        audio = np.frombuffer(raw, dtype=dtype).astype(np.float32)

        # Normalise
        audio /= float(np.iinfo(dtype).max)

        # Mix to mono if stereo
        if n_channels > 1:
            audio = audio.reshape(-1, n_channels).mean(axis=1)

        return audio, sample_rate

    # For other formats, let Whisper's load_audio handle it (requires ffmpeg)
    try:
        import whisper  # noqa: PLC0415
        audio = whisper.load_audio(path)
        return audio, DEFAULT_SAMPLE_RATE
    except ImportError as exc:
        raise RuntimeError(
            "openai-whisper is required to load non-WAV audio files. "
            "Install it with: pip install openai-whisper"
        ) from exc
