import * as FileSystem from 'expo-file-system/legacy';
import { loadSettings } from './settings';

export const MODELS_DIR = FileSystem.documentDirectory + 'models/';

const HF_WHISPER = 'https://huggingface.co/ggerganov/whisper.cpp/resolve/main';

export interface WhisperModelInfo {
  key: string;
  name: string;
  filename: string;
  url: string;
  sizeMB: number;
  multilingual: boolean;
  /** Short hint shown in the model picker. */
  hint?: string;
}

export interface LlmModelInfo {
  key: string;
  name: string;
  filename: string;
  url: string;
  sizeMB: number;
  stopTokens: string[];
  chatTemplate: string;
  hint?: string;
  /** Appended to the user prompt (e.g. Qwen3's ' /no_think' soft switch). */
  promptSuffix?: string;
}

export type ModelInfo = WhisperModelInfo | LlmModelInfo;

// ---------------------------------------------------------------------------
// Whisper model catalogue
// ---------------------------------------------------------------------------
export const WHISPER_MODELS: Record<string, WhisperModelInfo> = {
  TINY:     { key: 'TINY',     name: 'Whisper tiny',           filename: 'ggml-tiny.bin',     url: `${HF_WHISPER}/ggml-tiny.bin`,     sizeMB: 77,   multilingual: true,  hint: 'Fastest, basic accuracy' },
  TINY_EN:  { key: 'TINY_EN',  name: 'Whisper tiny (EN only)', filename: 'ggml-tiny.en.bin',  url: `${HF_WHISPER}/ggml-tiny.en.bin`,  sizeMB: 77,   multilingual: false },
  BASE:     { key: 'BASE',     name: 'Whisper base',           filename: 'ggml-base.bin',     url: `${HF_WHISPER}/ggml-base.bin`,     sizeMB: 142,  multilingual: true,  hint: 'Fast, good for clear English' },
  BASE_EN:  { key: 'BASE_EN',  name: 'Whisper base (EN only)', filename: 'ggml-base.en.bin',  url: `${HF_WHISPER}/ggml-base.en.bin`,  sizeMB: 142,  multilingual: false },
  SMALL:    { key: 'SMALL',    name: 'Whisper small',          filename: 'ggml-small.bin',    url: `${HF_WHISPER}/ggml-small.bin`,    sizeMB: 466,  multilingual: true,  hint: 'Recommended for Taglish + medical terms' },
  SMALL_EN: { key: 'SMALL_EN', name: 'Whisper small (EN only)',filename: 'ggml-small.en.bin', url: `${HF_WHISPER}/ggml-small.en.bin`, sizeMB: 466,  multilingual: false },
  MEDIUM:   { key: 'MEDIUM',   name: 'Whisper medium',         filename: 'ggml-medium.bin',   url: `${HF_WHISPER}/ggml-medium.bin`,   sizeMB: 1500, multilingual: true,  hint: 'Superseded by large-v3-turbo' },
  LARGE_V3: { key: 'LARGE_V3', name: 'Whisper large-v3',       filename: 'ggml-large-v3.bin', url: `${HF_WHISPER}/ggml-large-v3.bin`, sizeMB: 3100, multilingual: true },
  // large-v3-turbo: full large-v3 encoder, decoder pruned 32→4 layers —
  // near large-v3 accuracy at several times the speed. Best Taglish option.
  LARGE_V3_TURBO_Q5: { key: 'LARGE_V3_TURBO_Q5', name: 'Whisper large-v3-turbo (q5)', filename: 'ggml-large-v3-turbo-q5_0.bin', url: `${HF_WHISPER}/ggml-large-v3-turbo-q5_0.bin`, sizeMB: 547, multilingual: true, hint: 'BEST for Taglish — near large-v3 accuracy, small download' },
  LARGE_V3_TURBO_Q8: { key: 'LARGE_V3_TURBO_Q8', name: 'Whisper large-v3-turbo (q8)', filename: 'ggml-large-v3-turbo-q8_0.bin', url: `${HF_WHISPER}/ggml-large-v3-turbo-q8_0.bin`, sizeMB: 833, multilingual: true, hint: 'Same as q5 with slightly higher fidelity' },
};

/** Models offered in the Settings picker (multilingual only — Taglish needs them). */
export const WHISPER_CHOICES = ['TINY', 'BASE', 'SMALL', 'LARGE_V3_TURBO_Q5', 'LARGE_V3_TURBO_Q8'] as const;

// ---------------------------------------------------------------------------
// Silero VAD model (hidden asset — not in any picker)
// ---------------------------------------------------------------------------
// Tiny speech/silence detector. Transcription uses it to skip silent stretches
// (meetings are often 20–40% silence), so the Whisper pass scales with speech
// time instead of recording time. Version matches whisper.rn 0.6.0's docs.
export const VAD_MODEL: WhisperModelInfo = {
  key: 'VAD',
  name: 'Silero VAD v6.2.0',
  filename: 'ggml-silero-v6.2.0.bin',
  url: 'https://huggingface.co/ggml-org/whisper-vad/resolve/main/ggml-silero-v6.2.0.bin',
  sizeMB: 0.8, // actual ~0.85 MB; kept below so the 95% size check passes
  multilingual: true,
};

/**
 * Returns the local path of the VAD model, downloading it on first use
 * (<1 MB). Returns null when unavailable (offline + never downloaded) —
 * callers fall back to full-file transcription.
 */
export async function ensureVadModel(): Promise<string | null> {
  return ensureHiddenModel(VAD_MODEL);
}

// ---------------------------------------------------------------------------
// Speaker diarization models (hidden assets — not in any picker)
// ---------------------------------------------------------------------------
// Used by DiarizationModule.kt (sherpa-onnx). pyannote finds speaker change
// points acoustically; CAM++ embeddings identify who is who. Both rehosted
// ungated by the sherpa-onnx project (pyannote segmentation-3.0 is MIT).
export const DIARIZE_SEGMENTATION_MODEL: WhisperModelInfo = {
  key: 'DIARIZE_SEG',
  name: 'pyannote segmentation-3.0',
  filename: 'pyannote-segmentation-3-0.onnx',
  url: 'https://huggingface.co/csukuangfj/sherpa-onnx-pyannote-segmentation-3-0/resolve/main/model.onnx',
  sizeMB: 5.7,
  multilingual: true,
};

export const DIARIZE_EMBEDDING_MODEL: WhisperModelInfo = {
  key: 'DIARIZE_EMB',
  name: '3D-Speaker CAM++ (zh-en advanced) speaker embedding',
  // Same size/architecture as the earlier VoxCeleb-EN CAM++, but benchmarked
  // far more robust on processed/noisy audio (see diarize.py CLUSTER_THRESHOLD
  // note). Must stay in sync with the desktop EMBEDDING_URL in diarize.py.
  filename: '3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx',
  url: 'https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-recongition-models/3dspeaker_speech_campplus_sv_zh_en_16k-common_advanced.onnx',
  sizeMB: 27,
  multilingual: true,
};

/**
 * Returns local paths of both diarization models, downloading on first use
 * (~34 MB total). Returns null when unavailable — diarize.ts falls back to
 * the pitch-clustering heuristic.
 */
export async function ensureDiarizationModels(): Promise<{ segmentation: string; embedding: string } | null> {
  const [segmentation, embedding] = await Promise.all([
    ensureHiddenModel(DIARIZE_SEGMENTATION_MODEL),
    ensureHiddenModel(DIARIZE_EMBEDDING_MODEL),
  ]);
  return segmentation && embedding ? { segmentation, embedding } : null;
}

/** Download-if-missing for models that never appear in a picker. */
async function ensureHiddenModel(model: ModelInfo): Promise<string | null> {
  if (await isModelDownloaded(model.filename, model.sizeMB)) {
    return modelPath(model.filename);
  }
  try {
    await downloadModel(model, () => {});
    return modelPath(model.filename);
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// LLM model catalogue
// ---------------------------------------------------------------------------
export const LLM_MODELS: Record<string, LlmModelInfo> = {
  TINYLLAMA: {
    key: 'TINYLLAMA',
    name: 'TinyLlama 1.1B Q4_K_M',
    filename: 'tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    url: 'https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf',
    sizeMB: 600,
    stopTokens: ['</s>', '<|im_end|>'],
    chatTemplate: 'chatml',
    hint: 'Fastest, weakest summaries',
  },
  QWEN2_1_5B: {
    key: 'QWEN2_1_5B',
    name: 'Qwen2 1.5B Instruct Q4_K_M',
    filename: 'qwen2-1.5b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2-1.5B-Instruct-GGUF/resolve/main/qwen2-1.5b-instruct-q4_k_m.gguf',
    sizeMB: 954,
    stopTokens: ['<|im_end|>', '<|endoftext|>'],
    chatTemplate: 'chatml',
  },
  QWEN2_5_1_5B: {
    key: 'QWEN2_5_1_5B',
    name: 'Qwen2.5 1.5B Instruct Q4_K_M',
    filename: 'qwen2.5-1.5b-instruct-q4_k_m.gguf',
    url: 'https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf',
    sizeMB: 1000,
    stopTokens: ['<|im_end|>', '<|endoftext|>'],
    chatTemplate: 'chatml',
    hint: 'Recommended — best quality/speed balance',
  },
  GEMMA2_2B: {
    key: 'GEMMA2_2B',
    name: 'Gemma 2 2B Instruct Q4_K_M',
    filename: 'gemma-2-2b-it-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/gemma-2-2b-it-GGUF/resolve/main/gemma-2-2b-it-Q4_K_M.gguf',
    sizeMB: 1600,
    stopTokens: ['<end_of_turn>', '<eos>'],
    chatTemplate: 'gemma',
    hint: 'Very good summaries, slower',
  },
  PHI3_MINI: {
    key: 'PHI3_MINI',
    name: 'Phi-3.5 Mini Instruct Q4_K_M',
    filename: 'Phi-3.5-mini-instruct-Q4_K_M.gguf',
    url: 'https://huggingface.co/bartowski/Phi-3.5-mini-instruct-GGUF/resolve/main/Phi-3.5-mini-instruct-Q4_K_M.gguf',
    sizeMB: 2200,
    stopTokens: ['<|end|>', '<|endoftext|>'],
    chatTemplate: 'phi3',
    hint: 'Superseded by Qwen3 / Gemma 3',
  },
  QWEN3_1_7B: {
    key: 'QWEN3_1_7B',
    name: 'Qwen3 1.7B Q4_K_M',
    filename: 'Qwen3-1.7B-Q4_K_M.gguf',
    url: 'https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_K_M.gguf',
    sizeMB: 1056,
    stopTokens: ['<|im_end|>', '<|endoftext|>'],
    chatTemplate: 'chatml',
    hint: 'Recommended — strong multilingual, fast',
    // Qwen3 defaults to "thinking" mode, which floods the output with
    // reasoning before the JSON; the soft switch disables it per prompt.
    promptSuffix: ' /no_think',
  },
  QWEN3_4B: {
    key: 'QWEN3_4B',
    name: 'Qwen3 4B Instruct 2507 Q4_K_M',
    filename: 'Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
    url: 'https://huggingface.co/unsloth/Qwen3-4B-Instruct-2507-GGUF/resolve/main/Qwen3-4B-Instruct-2507-Q4_K_M.gguf',
    sizeMB: 2381,
    stopTokens: ['<|im_end|>', '<|endoftext|>'],
    chatTemplate: 'chatml',
    hint: 'Best notes quality — flagship phones',
  },
  GEMMA3_4B: {
    key: 'GEMMA3_4B',
    name: 'Gemma 3 4B Instruct Q4_K_M',
    filename: 'gemma-3-4b-it-Q4_K_M.gguf',
    url: 'https://huggingface.co/ggml-org/gemma-3-4b-it-GGUF/resolve/main/gemma-3-4b-it-Q4_K_M.gguf',
    sizeMB: 2374,
    stopTokens: ['<end_of_turn>', '<eos>'],
    chatTemplate: 'gemma',
    hint: 'Strongest Filipino/Tagalog (140+ languages)',
  },
  // Qwen3.5 small series (Mar 2026, Apache 2.0): 201 languages, much stronger
  // structured/JSON output than the Qwen3 + Gemma 3 generation.
  QWEN3_5_2B: {
    key: 'QWEN3_5_2B',
    name: 'Qwen3.5 2B Q4_K_M',
    filename: 'Qwen3.5-2B-Q4_K_M.gguf',
    url: 'https://huggingface.co/unsloth/Qwen3.5-2B-GGUF/resolve/main/Qwen3.5-2B-Q4_K_M.gguf',
    sizeMB: 1221,
    stopTokens: ['<|im_end|>', '<|endoftext|>'],
    chatTemplate: 'chatml',
    hint: 'Recommended — big quality jump at Qwen3-1.7B speed',
    // Harmless if Qwen3.5 ignores the Qwen3 soft switch; without it a
    // thinking-mode default would fight the JSON grammar constraint.
    promptSuffix: ' /no_think',
  },
  QWEN3_5_4B: {
    key: 'QWEN3_5_4B',
    name: 'Qwen3.5 4B Q4_K_M',
    filename: 'Qwen3.5-4B-Q4_K_M.gguf',
    url: 'https://huggingface.co/unsloth/Qwen3.5-4B-GGUF/resolve/main/Qwen3.5-4B-Q4_K_M.gguf',
    sizeMB: 2613,
    stopTokens: ['<|im_end|>', '<|endoftext|>'],
    chatTemplate: 'chatml',
    hint: 'Best notes quality — flagship phones',
    promptSuffix: ' /no_think',
  },
};

export const LLM_CHOICES = ['TINYLLAMA', 'QWEN3_5_2B', 'QWEN3_5_4B', 'GEMMA3_4B'] as const;

// ---------------------------------------------------------------------------
// Active model resolution (user-selectable in Settings)
// ---------------------------------------------------------------------------

export async function getActiveWhisperModel(): Promise<WhisperModelInfo> {
  const { whisperModelKey } = await loadSettings();
  return WHISPER_MODELS[whisperModelKey] ?? WHISPER_MODELS.BASE;
}

export async function getActiveLlmModel(): Promise<LlmModelInfo> {
  const { llmModelKey } = await loadSettings();
  return LLM_MODELS[llmModelKey] ?? LLM_MODELS.TINYLLAMA;
}

// ---------------------------------------------------------------------------
// Download helpers
// ---------------------------------------------------------------------------

export function modelPath(filename: string): string {
  return MODELS_DIR + filename;
}

export async function isModelDownloaded(filename: string, expectedSizeMB?: number): Promise<boolean> {
  const info = await FileSystem.getInfoAsync(modelPath(filename));
  if (!info.exists) return false;
  const minBytes = expectedSizeMB
    ? expectedSizeMB * 1024 * 1024 * 0.95
    : 1024 * 1024;
  return (info as any).size >= minBytes;
}

export async function deleteModel(filename: string): Promise<void> {
  const path = modelPath(filename);
  const info = await FileSystem.getInfoAsync(path);
  if (info.exists) await FileSystem.deleteAsync(path, { idempotent: true });
}

export async function downloadModel(
  model: ModelInfo,
  onProgress: (ratio: number) => void,
): Promise<string> {
  await FileSystem.makeDirectoryAsync(MODELS_DIR, { intermediates: true });
  const dest = modelPath(model.filename);

  const dl = FileSystem.createDownloadResumable(
    model.url,
    dest,
    {},
    ({ totalBytesWritten, totalBytesExpectedToWrite }) => {
      if (totalBytesExpectedToWrite > 0)
        onProgress(totalBytesWritten / totalBytesExpectedToWrite);
    },
  );

  const result = await dl.downloadAsync();
  if (!result?.uri) {
    await FileSystem.deleteAsync(dest, { idempotent: true });
    throw new Error(`Download failed for ${model.name}`);
  }

  // Guard: if the server returned a redirect page or error HTML instead of
  // the actual model file, the size will be tiny. Catch it here.
  const ok = await isModelDownloaded(model.filename, model.sizeMB);
  if (!ok) {
    const info = await FileSystem.getInfoAsync(dest);
    const actualMB = Math.round(((info as any).size ?? 0) / 1024 / 1024);
    await FileSystem.deleteAsync(dest, { idempotent: true });
    throw new Error(
      `${model.name} download incomplete: got ${actualMB} MB, expected ~${model.sizeMB} MB.\n` +
      `The server may have returned an error instead of the file. Check your connection and try again.`,
    );
  }

  return result.uri;
}

export async function allModelsReady(): Promise<boolean> {
  const [whisperModel, llmModel] = await Promise.all([
    getActiveWhisperModel(),
    getActiveLlmModel(),
  ]);
  const [w, l] = await Promise.all([
    isModelDownloaded(whisperModel.filename, whisperModel.sizeMB),
    isModelDownloaded(llmModel.filename, llmModel.sizeMB),
  ]);
  return w && l;
}
