import { useNavigation, useRouter } from 'expo-router';
import { useCallback, useEffect, useRef, useState } from 'react';
import * as FileSystem from 'expo-file-system/legacy';
import {
  Alert,
  Animated,
  Easing,
  SafeAreaView,
  ScrollView,
  StatusBar,
  StyleSheet,
  Text,
  TouchableOpacity,
  View,
} from 'react-native';
import { ConversationView } from '../components/ConversationView';
import { saveMeeting, updateMeeting } from '../lib/api';
import { assignSpeakers } from '../lib/diarize';
import { emailNotes, sharePdf } from '../lib/export';
import { startLiveTranscription, LiveSession } from '../lib/live';
import { generateNotes } from '../lib/llm';
import { findMedicalTerms } from '../lib/medicalTerms';
import { allModelsReady } from '../lib/models';
import { AppSettings, loadSettings, whisperLanguage } from '../lib/settings';
import { startWavRecording, stopWavRecording } from '../lib/wavRecorder';
import { transcribeAudio } from '../lib/whisper';
import type { MeetingRecord } from '../lib/types';

type AppState = 'idle' | 'recording' | 'transcribing' | 'generating' | 'done' | 'error';

export default function RecordScreen() {
  const router = useRouter();
  const [appState, setAppState] = useState<AppState>('idle');
  const [elapsed, setElapsed] = useState(0);
  const [stepText, setStepText] = useState('');
  const [transcribeProgress, setTranscribeProgress] = useState(0);
  const [liveText, setLiveText] = useState('');
  const [errorMsg, setErrorMsg] = useState('');
  const [lastMeeting, setLastMeeting] = useState<MeetingRecord | null>(null);
  const [modelsReady, setModelsReady] = useState<boolean | null>(null);

  const isRecordingRef = useRef(false);
  const liveSessionRef = useRef<LiveSession | null>(null);
  const settingsRef = useRef<AppSettings | null>(null);
  const timerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const pulseAnim = useRef(new Animated.Value(1)).current;
  const captionScrollRef = useRef<ScrollView | null>(null);

  const navigation = useNavigation();
  useEffect(() => {
    allModelsReady().then(setModelsReady);
    const unsub = navigation.addListener('focus', () => allModelsReady().then(setModelsReady));
    return unsub;
  }, [navigation]);

  useEffect(() => {
    if (appState === 'recording') {
      Animated.loop(
        Animated.sequence([
          Animated.timing(pulseAnim, { toValue: 1.15, duration: 700, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
          Animated.timing(pulseAnim, { toValue: 1, duration: 700, easing: Easing.inOut(Easing.ease), useNativeDriver: true }),
        ]),
      ).start();
    } else {
      pulseAnim.stopAnimation();
      pulseAnim.setValue(1);
    }
  }, [appState]);

  const fmt = (s: number) =>
    `${Math.floor(s / 60).toString().padStart(2, '0')}:${(s % 60).toString().padStart(2, '0')}`;

  const startRecording = useCallback(async () => {
    try {
      const settings = await loadSettings();
      settingsRef.current = settings;
      setLiveText('');
      await startWavRecording(settings.liveTranscription);
      isRecordingRef.current = true;
      setElapsed(0);
      setAppState('recording');
      timerRef.current = setInterval(() => setElapsed((e) => e + 1), 1000);

      if (settings.liveTranscription) {
        try {
          liveSessionRef.current = await startLiveTranscription(setLiveText, {
            language: whisperLanguage(settings.languageMode),
          });
        } catch {
          // Captions are best-effort — recording continues without them
          liveSessionRef.current = null;
        }
      }
    } catch (err) {
      Alert.alert('Permission denied', err instanceof Error ? err.message : 'Microphone access is required.');
    }
  }, []);

  const stopAndProcess = useCallback(async () => {
    if (!isRecordingRef.current) return;
    if (timerRef.current) clearInterval(timerRef.current);
    const durationSeconds = elapsed;
    const settings = settingsRef.current ?? (await loadSettings());

    setAppState('transcribing');
    setStepText('Stopping recording…');

    if (liveSessionRef.current) {
      await liveSessionRef.current.stop().catch(() => {});
      liveSessionRef.current = null;
    }
    const uri = await stopWavRecording();
    isRecordingRef.current = false;

    if (!uri) { setErrorMsg('No audio captured.'); setAppState('error'); return; }

    try {
      // Step 1: Transcribe with Whisper (on-device)
      setStepText('Transcribing with Whisper…');
      setTranscribeProgress(0);
      const { text, language, segments, speechRegions } = await transcribeAudio(uri, {
        language: whisperLanguage(settings.languageMode),
        onProgress: setTranscribeProgress,
      });

      // Step 2: Who said what — voice-based speaker detection
      setAppState('generating');
      setStepText('Identifying speakers…');
      const speakers = await assignSpeakers(segments, uri, speechRegions);

      // Step 3: Generate notes with LLM (on-device)
      setStepText('Generating meeting notes…');
      const notes = await generateNotes(text, language, speakers, { onStep: setStepText });

      // Merge lexicon hits with LLM-flagged terms (dedupe, keep order)
      const seen = new Set<string>();
      const medicalTerms = [...findMedicalTerms(text), ...(notes.medical_terms ?? [])]
        .filter((t) => {
          const k = t.toLowerCase();
          if (seen.has(k)) return false;
          seen.add(k);
          return true;
        });

      const record: MeetingRecord = {
        id: Date.now().toString(),
        created_at: new Date().toISOString(),
        duration_seconds: durationSeconds,
        transcript: { text, language, language_name: language, timed_transcript: text, speakers },
        notes: {
          ...notes,
          medical_terms: medicalTerms,
          conversation: speakers.length ? speakers : undefined,
        },
      };
      await saveMeeting(record);

      if (settings.autoDeleteAudio) {
        const fileUri = uri.startsWith('file://') ? uri : 'file://' + uri;
        await FileSystem.deleteAsync(fileUri, { idempotent: true }).catch(() => {});
      }

      setLastMeeting(record);
      setAppState('done');
    } catch (err: unknown) {
      console.error('Processing error:', JSON.stringify(err));
      let msg = 'Unknown error';
      if (err instanceof Error) {
        msg = err.message;
      } else if (err && typeof err === 'object') {
        const e = err as Record<string, unknown>;
        msg = String(e.message ?? e.error ?? e.code ?? JSON.stringify(err));
      } else {
        msg = String(err);
      }
      setErrorMsg(msg);
      setAppState('error');
    }
  }, [elapsed]);

  const handleRename = useCallback((speaker: string, name: string) => {
    setLastMeeting((prev) => {
      if (!prev) return prev;
      const next = { ...prev, speakerNames: { ...prev.speakerNames, [speaker]: name } };
      updateMeeting(next).catch(() => {});
      return next;
    });
  }, []);

  const handleEmail = useCallback(async () => {
    if (!lastMeeting) return;
    try { await emailNotes(lastMeeting); }
    catch (err) { Alert.alert('Email failed', err instanceof Error ? err.message : String(err)); }
  }, [lastMeeting]);

  const handleShare = useCallback(async () => {
    if (!lastMeeting) return;
    try { await sharePdf(lastMeeting); }
    catch (err) { Alert.alert('Share failed', err instanceof Error ? err.message : String(err)); }
  }, [lastMeeting]);

  const handlePress = () => {
    if (appState === 'recording') stopAndProcess();
    else if (['idle', 'done', 'error'].includes(appState)) startRecording();
  };

  const isRecording = appState === 'recording';
  const isProcessing = appState === 'transcribing' || appState === 'generating';

  if (modelsReady === false) {
    return (
      <SafeAreaView style={styles.container}>
        <View style={styles.setupPrompt}>
          <Text style={styles.setupIcon}>⚙️</Text>
          <Text style={styles.setupTitle}>Models not downloaded</Text>
          <Text style={styles.setupBody}>Download the AI models in Settings to enable offline transcription.</Text>
          <TouchableOpacity style={styles.setupBtn} onPress={() => router.push('/settings')}>
            <Text style={styles.setupBtnText}>Go to Settings →</Text>
          </TouchableOpacity>
        </View>
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0F172A" />
      <ScrollView contentContainerStyle={styles.content} bounces={false}>

        {/* Offline badge */}
        <View style={styles.offlineBadge}>
          <View style={styles.offlineDot} />
          <Text style={styles.offlineText}>Offline · No server needed</Text>
        </View>

        <Text style={styles.timer}>{fmt(elapsed)}</Text>

        <Text style={styles.stateLabel}>
          {appState === 'idle' && 'Tap to start recording'}
          {appState === 'recording' && 'Recording…  tap to stop'}
          {appState === 'transcribing' && `Transcribing… ${Math.round(transcribeProgress * 100)}%`}
          {appState === 'generating' && stepText}
          {appState === 'done' && 'Notes ready!'}
          {appState === 'error' && `Error: ${errorMsg}`}
        </Text>

        <Animated.View style={{ transform: [{ scale: pulseAnim }] }}>
          <TouchableOpacity
            style={[styles.recordBtn, isRecording && styles.recordBtnActive, isProcessing && styles.recordBtnProcessing]}
            onPress={handlePress}
            disabled={isProcessing || modelsReady === null}
            activeOpacity={0.8}
          >
            <Text style={styles.recordBtnIcon}>
              {isRecording ? '■' : isProcessing ? '…' : '●'}
            </Text>
          </TouchableOpacity>
        </Animated.View>

        {/* Live captions while recording */}
        {isRecording && liveText !== '' && (
          <View style={styles.captionBox}>
            <Text style={styles.captionLabel}>LIVE TRANSCRIPT</Text>
            <ScrollView
              ref={captionScrollRef}
              style={styles.captionScroll}
              nestedScrollEnabled
              onContentSizeChange={() => captionScrollRef.current?.scrollToEnd({ animated: true })}
            >
              <Text style={styles.captionText}>{liveText}</Text>
            </ScrollView>
          </View>
        )}

        {isProcessing && (
          <View style={styles.stepsBox}>
            <Text style={styles.stepsText}>{stepText}</Text>
            {appState === 'transcribing' && (
              <View style={styles.progressBg}>
                <View style={[styles.progressFill, { width: `${Math.round(transcribeProgress * 100)}%` }]} />
              </View>
            )}
            <Text style={styles.stepsHint}>Running on-device · no internet needed</Text>
          </View>
        )}

        {appState === 'done' && lastMeeting && (
          <View style={styles.previewBox}>
            <Text style={styles.previewTitle}>{lastMeeting.notes.title}</Text>
            <Text style={styles.previewSummary}>{lastMeeting.notes.summary}</Text>

            {(lastMeeting.notes.medical_terms?.length ?? 0) > 0 && (
              <>
                <Text style={styles.previewSection}>Medical Terms</Text>
                <View style={styles.chipRow}>
                  {lastMeeting.notes.medical_terms!.slice(0, 8).map((term, i) => (
                    <View key={i} style={styles.chip}>
                      <Text style={styles.chipText}>{term}</Text>
                    </View>
                  ))}
                </View>
              </>
            )}

            {lastMeeting.notes.conversation && lastMeeting.notes.conversation.length > 0 && (
              <>
                <Text style={styles.previewSection}>Conversation · tap a name to rename</Text>
                <ConversationView
                  turns={lastMeeting.notes.conversation}
                  names={lastMeeting.speakerNames}
                  medicalTerms={lastMeeting.notes.medical_terms}
                  onRename={handleRename}
                  maxTurns={6}
                />
              </>
            )}

            {lastMeeting.notes.action_items.length > 0 && (
              <>
                <Text style={styles.previewSection}>Action Items</Text>
                {lastMeeting.notes.action_items.slice(0, 3).map((item, i) => (
                  <Text key={i} style={styles.previewItem}>☐ {item}</Text>
                ))}
              </>
            )}

            <View style={styles.actionRow}>
              <TouchableOpacity style={styles.actionBtn} onPress={handleEmail}>
                <Text style={styles.actionBtnText}>Email Notes</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.actionBtnAlt} onPress={handleShare}>
                <Text style={styles.actionBtnAltText}>Share PDF</Text>
              </TouchableOpacity>
            </View>

            <TouchableOpacity style={styles.viewBtn} onPress={() => router.push('/history')}>
              <Text style={styles.viewBtnText}>View Full Notes →</Text>
            </TouchableOpacity>
          </View>
        )}
      </ScrollView>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A' },
  content: { alignItems: 'center', paddingHorizontal: 24, paddingTop: 24, paddingBottom: 48, gap: 24 },
  offlineBadge: { flexDirection: 'row', alignItems: 'center', gap: 6, backgroundColor: '#0D2011', paddingHorizontal: 12, paddingVertical: 5, borderRadius: 20, borderWidth: 1, borderColor: '#166534' },
  offlineDot: { width: 7, height: 7, borderRadius: 4, backgroundColor: '#4ADE80' },
  offlineText: { fontSize: 12, color: '#4ADE80', fontWeight: '600' },
  timer: { fontSize: 64, fontWeight: '200', color: '#F1F5F9', letterSpacing: 4 },
  stateLabel: { fontSize: 15, color: '#94A3B8', textAlign: 'center', minHeight: 20 },
  recordBtn: { width: 120, height: 120, borderRadius: 60, backgroundColor: '#1E3A5F', borderWidth: 3, borderColor: '#3B82F6', alignItems: 'center', justifyContent: 'center', marginVertical: 8 },
  recordBtnActive: { backgroundColor: '#7F1D1D', borderColor: '#EF4444' },
  recordBtnProcessing: { backgroundColor: '#1E293B', borderColor: '#64748B' },
  recordBtnIcon: { fontSize: 40, color: '#F1F5F9' },
  captionBox: { width: '100%', backgroundColor: '#1E293B', borderRadius: 12, padding: 14, gap: 8, borderWidth: 1, borderColor: '#334155', maxHeight: 180 },
  captionLabel: { fontSize: 11, fontWeight: '700', color: '#F87171', letterSpacing: 1 },
  captionScroll: { maxHeight: 130 },
  captionText: { fontSize: 14, color: '#CBD5E1', lineHeight: 20 },
  stepsBox: { alignItems: 'center', gap: 8, width: '100%' },
  stepsText: { fontSize: 15, color: '#60A5FA', fontWeight: '500' },
  stepsHint: { fontSize: 12, color: '#475569' },
  progressBg: { width: '80%', height: 4, backgroundColor: '#1E293B', borderRadius: 2 },
  progressFill: { height: 4, backgroundColor: '#3B82F6', borderRadius: 2 },
  previewBox: { width: '100%', backgroundColor: '#1E293B', borderRadius: 16, padding: 20, gap: 10, borderWidth: 1, borderColor: '#334155' },
  previewTitle: { fontSize: 18, fontWeight: '700', color: '#F1F5F9' },
  previewSummary: { fontSize: 14, color: '#94A3B8', lineHeight: 20 },
  previewSection: { fontSize: 13, fontWeight: '600', color: '#60A5FA', marginTop: 4 },
  previewItem: { fontSize: 13, color: '#CBD5E1', paddingLeft: 4 },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  chip: { backgroundColor: 'rgba(250, 204, 21, 0.15)', borderRadius: 6, paddingHorizontal: 8, paddingVertical: 3, borderWidth: 1, borderColor: 'rgba(250, 204, 21, 0.35)' },
  chipText: { fontSize: 12, color: '#FDE047', fontWeight: '600' },
  actionRow: { flexDirection: 'row', gap: 10, marginTop: 8 },
  actionBtn: { flex: 1, paddingVertical: 12, backgroundColor: '#1D4ED8', borderRadius: 10, alignItems: 'center', minHeight: 44, justifyContent: 'center' },
  actionBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  actionBtnAlt: { flex: 1, paddingVertical: 12, backgroundColor: '#0F172A', borderRadius: 10, alignItems: 'center', borderWidth: 1, borderColor: '#3B82F6', minHeight: 44, justifyContent: 'center' },
  actionBtnAltText: { color: '#60A5FA', fontWeight: '700', fontSize: 14 },
  viewBtn: { marginTop: 4, paddingVertical: 10, paddingHorizontal: 20, borderRadius: 8, alignSelf: 'flex-start' },
  viewBtnText: { color: '#60A5FA', fontWeight: '600', fontSize: 14 },
  setupPrompt: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 16, padding: 32 },
  setupIcon: { fontSize: 48 },
  setupTitle: { fontSize: 20, fontWeight: '700', color: '#F1F5F9' },
  setupBody: { fontSize: 14, color: '#64748B', textAlign: 'center', lineHeight: 20 },
  setupBtn: { backgroundColor: '#1D4ED8', paddingVertical: 12, paddingHorizontal: 28, borderRadius: 12 },
  setupBtnText: { color: '#fff', fontWeight: '700', fontSize: 15 },
});
