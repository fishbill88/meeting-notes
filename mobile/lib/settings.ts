import AsyncStorage from '@react-native-async-storage/async-storage';

export type LanguageMode = 'auto' | 'en' | 'tl';

export interface AppSettings {
  /** Language preset for Whisper. 'tl' keeps Taglish code-switching intact. */
  languageMode: LanguageMode;
  /** Show live captions while recording (transcribes in the background). */
  liveTranscription: boolean;
  /** Delete the raw WAV after processing — keeps sensitive audio off the device. */
  autoDeleteAudio: boolean;
  /** Key into WHISPER_MODELS in models.ts */
  whisperModelKey: string;
  /** Key into LLM_MODELS in models.ts */
  llmModelKey: string;
}

const SETTINGS_KEY = 'app_settings';

export const DEFAULT_SETTINGS: AppSettings = {
  languageMode: 'auto',
  liveTranscription: true,
  autoDeleteAudio: true,
  whisperModelKey: 'BASE',
  llmModelKey: 'TINYLLAMA',
};

export async function loadSettings(): Promise<AppSettings> {
  try {
    const raw = await AsyncStorage.getItem(SETTINGS_KEY);
    return raw ? { ...DEFAULT_SETTINGS, ...JSON.parse(raw) } : { ...DEFAULT_SETTINGS };
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

export async function saveSettings(patch: Partial<AppSettings>): Promise<AppSettings> {
  const next = { ...(await loadSettings()), ...patch };
  await AsyncStorage.setItem(SETTINGS_KEY, JSON.stringify(next));
  return next;
}

/** Maps the language preset to the value Whisper expects (undefined = auto-detect). */
export function whisperLanguage(mode: LanguageMode): string | undefined {
  return mode === 'auto' ? undefined : mode;
}

export const LANGUAGE_LABELS: Record<LanguageMode, string> = {
  auto: 'Auto-detect',
  en: 'English',
  tl: 'Filipino (Taglish)',
};
