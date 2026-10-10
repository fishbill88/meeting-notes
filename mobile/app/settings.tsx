import { useEffect, useState } from 'react';
import {
  Alert,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Switch,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import {
  LLM_CHOICES,
  LLM_MODELS,
  ModelInfo,
  WHISPER_CHOICES,
  WHISPER_MODELS,
  deleteModel,
  downloadModel,
  isModelDownloaded,
} from '../lib/models';
import {
  AppSettings,
  DEFAULT_SETTINGS,
  LANGUAGE_LABELS,
  LanguageMode,
  loadSettings,
  saveSettings,
} from '../lib/settings';

type DlState = 'checking' | 'idle' | 'downloading' | 'done';
interface ModelStatus { state: DlState; progress: number; }

const LANGUAGE_MODES: LanguageMode[] = ['auto', 'en', 'tl'];

export default function SettingsScreen() {
  const [settings, setSettings] = useState<AppSettings>(DEFAULT_SETTINGS);
  const [statuses, setStatuses] = useState<Record<string, ModelStatus>>({});

  const whisperChoices = WHISPER_CHOICES.map((k) => WHISPER_MODELS[k]);
  const llmChoices = LLM_CHOICES.map((k) => LLM_MODELS[k]);

  useEffect(() => {
    loadSettings().then(setSettings);
    checkAll();
  }, []);

  async function checkAll() {
    const all = [...whisperChoices, ...llmChoices];
    const entries = await Promise.all(
      all.map(async (m) => {
        const done = await isModelDownloaded(m.filename, m.sizeMB);
        return [m.filename, { state: done ? 'done' : 'idle', progress: 0 } as ModelStatus] as const;
      }),
    );
    setStatuses(Object.fromEntries(entries));
  }

  function setStatus(filename: string, status: ModelStatus) {
    setStatuses((prev) => ({ ...prev, [filename]: status }));
  }

  async function handleDownload(info: ModelInfo) {
    try {
      await deleteModel(info.filename);
      setStatus(info.filename, { state: 'downloading', progress: 0 });
      await downloadModel(info, (p) => setStatus(info.filename, { state: 'downloading', progress: p }));
      setStatus(info.filename, { state: 'done', progress: 1 });
    } catch (err) {
      Alert.alert('Download failed', err instanceof Error ? err.message : String(err));
      setStatus(info.filename, { state: 'idle', progress: 0 });
    }
  }

  function handleDelete(info: ModelInfo) {
    Alert.alert(
      'Delete model?',
      `Remove ${info.name} (~${info.sizeMB} MB) from your device? You can re-download it here.`,
      [
        { text: 'Cancel', style: 'cancel' },
        {
          text: 'Delete',
          style: 'destructive',
          onPress: async () => {
            await deleteModel(info.filename);
            setStatus(info.filename, { state: 'idle', progress: 0 });
          },
        },
      ],
    );
  }

  async function selectModel(kind: 'whisper' | 'llm', info: ModelInfo) {
    const patch = kind === 'whisper'
      ? { whisperModelKey: info.key }
      : { llmModelKey: info.key };
    setSettings(await saveSettings(patch));
    if (statuses[info.filename]?.state === 'idle') {
      // Selected a model that isn't on the device yet — fetch it right away
      handleDownload(info);
    }
  }

  async function patchSettings(patch: Partial<AppSettings>) {
    setSettings(await saveSettings(patch));
  }

  function ModelRow({
    info,
    selected,
    onSelect,
  }: {
    info: ModelInfo;
    selected: boolean;
    onSelect: () => void;
  }) {
    const status = statuses[info.filename] ?? { state: 'checking', progress: 0 };
    return (
      <TouchableOpacity
        style={[styles.modelRow, selected && styles.modelRowSelected]}
        onPress={onSelect}
        activeOpacity={0.75}
      >
        <View style={styles.radioOuter}>
          {selected && <View style={styles.radioInner} />}
        </View>
        <View style={styles.modelInfo}>
          <Text style={styles.modelName}>{info.name}</Text>
          <Text style={styles.modelMeta}>
            ~{info.sizeMB} MB{info.hint ? ` · ${info.hint}` : ''}
          </Text>
          {status.state === 'downloading' && (
            <View style={styles.progressBg}>
              <View style={[styles.progressFill, { width: `${Math.round(status.progress * 100)}%` }]} />
            </View>
          )}
        </View>
        <View style={styles.modelActions}>
          {status.state === 'done' && <Text style={styles.downloadedMark}>✓</Text>}
          {status.state === 'downloading' && (
            <Text style={styles.downloadingPct}>{Math.round(status.progress * 100)}%</Text>
          )}
          {status.state === 'idle' && (
            <TouchableOpacity style={styles.smallBtn} onPress={() => handleDownload(info)}>
              <Text style={styles.smallBtnText}>Get</Text>
            </TouchableOpacity>
          )}
          {status.state === 'done' && !selected && (
            <TouchableOpacity onPress={() => handleDelete(info)} hitSlop={{ top: 8, bottom: 8, left: 8, right: 8 }}>
              <Text style={styles.deleteText}>Delete</Text>
            </TouchableOpacity>
          )}
        </View>
      </TouchableOpacity>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0F172A" />
      <ScrollView style={styles.scroll} contentContainerStyle={styles.inner}>

        {/* ── Language ── */}
        <Text style={styles.heading}>Language</Text>
        <Text style={styles.hint}>
          Filipino (Taglish) keeps mixed Tagalog–English speech intact. Auto-detect can flip
          mid-meeting on mixed speech.
        </Text>
        <View style={styles.segmentRow}>
          {LANGUAGE_MODES.map((mode) => (
            <TouchableOpacity
              key={mode}
              style={[styles.segment, settings.languageMode === mode && styles.segmentActive]}
              onPress={() => patchSettings({ languageMode: mode })}
            >
              <Text style={[styles.segmentText, settings.languageMode === mode && styles.segmentTextActive]}>
                {LANGUAGE_LABELS[mode]}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        {/* ── Transcription model ── */}
        <Text style={styles.heading}>Transcription Model</Text>
        <Text style={styles.hint}>
          Large-v3-turbo (q5) is the best pick for Taglish and medical terms — near
          top-tier accuracy in a 547 MB download. Smaller models are faster but mishear
          mixed languages.
        </Text>
        {whisperChoices.map((m) => (
          <ModelRow
            key={m.key}
            info={m}
            selected={settings.whisperModelKey === m.key}
            onSelect={() => selectModel('whisper', m)}
          />
        ))}

        {/* ── AI notes model ── */}
        <Text style={styles.heading}>AI Notes Model</Text>
        <Text style={styles.hint}>
          Generates the summary, decisions, and action items. Qwen3 1.7B is the best
          balance; the 4B models give the best notes if your phone has the headroom.
        </Text>
        {llmChoices.map((m) => (
          <ModelRow
            key={m.key}
            info={m}
            selected={settings.llmModelKey === m.key}
            onSelect={() => selectModel('llm', m)}
          />
        ))}

        {/* ── Recording ── */}
        <Text style={styles.heading}>Recording</Text>
        <View style={styles.switchRow}>
          <View style={styles.switchLabelBox}>
            <Text style={styles.switchLabel}>Live transcript while recording</Text>
            <Text style={styles.switchHint}>Shows captions as you speak. Uses more battery.</Text>
          </View>
          <Switch
            value={settings.liveTranscription}
            onValueChange={(v) => patchSettings({ liveTranscription: v })}
            trackColor={{ false: '#334155', true: '#1D4ED8' }}
            thumbColor="#F1F5F9"
          />
        </View>
        <View style={styles.switchRow}>
          <View style={styles.switchLabelBox}>
            <Text style={styles.switchLabel}>Auto-delete audio after processing</Text>
            <Text style={styles.switchHint}>
              Keeps sensitive recordings off the device — only the transcript and notes are stored.
            </Text>
          </View>
          <Switch
            value={settings.autoDeleteAudio}
            onValueChange={(v) => patchSettings({ autoDeleteAudio: v })}
            trackColor={{ false: '#334155', true: '#1D4ED8' }}
            thumbColor="#F1F5F9"
          />
        </View>

        <Text style={styles.footer}>
          Models run fully offline on your device. Download once, use forever.
        </Text>
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A' },
  scroll: { flex: 1 },
  inner: { padding: 20, paddingBottom: 48, gap: 10 },
  heading: { fontSize: 13, fontWeight: '700', color: '#3B82F6', textTransform: 'uppercase', letterSpacing: 0.8, marginTop: 16 },
  hint: { fontSize: 13, color: '#64748B', lineHeight: 18, marginBottom: 4 },
  segmentRow: { flexDirection: 'row', backgroundColor: '#1E293B', borderRadius: 10, padding: 4, gap: 4 },
  segment: { flex: 1, paddingVertical: 10, borderRadius: 8, alignItems: 'center', minHeight: 44, justifyContent: 'center' },
  segmentActive: { backgroundColor: '#1D4ED8' },
  segmentText: { fontSize: 13, color: '#94A3B8', fontWeight: '600', textAlign: 'center' },
  segmentTextActive: { color: '#fff' },
  modelRow: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: '#1E293B', borderRadius: 12, padding: 14,
    borderWidth: 1, borderColor: '#334155', minHeight: 44,
  },
  modelRowSelected: { borderColor: '#3B82F6' },
  radioOuter: {
    width: 20, height: 20, borderRadius: 10, borderWidth: 2, borderColor: '#3B82F6',
    alignItems: 'center', justifyContent: 'center',
  },
  radioInner: { width: 10, height: 10, borderRadius: 5, backgroundColor: '#3B82F6' },
  modelInfo: { flex: 1, gap: 2 },
  modelName: { fontSize: 14, fontWeight: '600', color: '#F1F5F9' },
  modelMeta: { fontSize: 12, color: '#64748B' },
  modelActions: { alignItems: 'flex-end', gap: 6 },
  downloadedMark: { fontSize: 16, color: '#4ADE80', fontWeight: '700' },
  downloadingPct: { fontSize: 13, color: '#60A5FA', fontWeight: '600' },
  smallBtn: { backgroundColor: '#1D4ED8', borderRadius: 8, paddingVertical: 8, paddingHorizontal: 16 },
  smallBtnText: { color: '#fff', fontWeight: '700', fontSize: 13 },
  deleteText: { color: '#EF4444', fontSize: 12 },
  progressBg: { height: 4, backgroundColor: '#0F172A', borderRadius: 2, marginTop: 6 },
  progressFill: { height: 4, backgroundColor: '#3B82F6', borderRadius: 2 },
  switchRow: {
    flexDirection: 'row', alignItems: 'center', gap: 12,
    backgroundColor: '#1E293B', borderRadius: 12, padding: 14,
    borderWidth: 1, borderColor: '#334155',
  },
  switchLabelBox: { flex: 1, gap: 2 },
  switchLabel: { fontSize: 14, fontWeight: '600', color: '#F1F5F9' },
  switchHint: { fontSize: 12, color: '#64748B', lineHeight: 16 },
  footer: { fontSize: 12, color: '#475569', textAlign: 'center', marginTop: 20 },
});
