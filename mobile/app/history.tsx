import { useFocusEffect } from 'expo-router';
import { useCallback, useMemo, useState } from 'react';
import {
  Alert,
  FlatList,
  Platform,
  SafeAreaView,
  StatusBar,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import { ConversationView } from '../components/ConversationView';
import { HighlightedText } from '../components/HighlightedText';
import { deleteMeeting, loadHistory, updateMeeting } from '../lib/api';
import { applySpeakerNames, emailNotes, sharePdf } from '../lib/export';
import type { MeetingRecord } from '../lib/types';

export default function HistoryScreen() {
  const [meetings, setMeetings] = useState<MeetingRecord[]>([]);
  const [selected, setSelected] = useState<MeetingRecord | null>(null);
  const [query, setQuery] = useState('');

  useFocusEffect(
    useCallback(() => {
      loadHistory().then(setMeetings);
    }, [])
  );

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    if (!q) return meetings;
    return meetings.filter((m) =>
      m.notes.title.toLowerCase().includes(q) ||
      m.notes.summary.toLowerCase().includes(q) ||
      m.transcript.text.toLowerCase().includes(q) ||
      (m.notes.medical_terms ?? []).some((t) => t.toLowerCase().includes(q))
    );
  }, [meetings, query]);

  const handleDelete = (id: string) => {
    Alert.alert('Delete meeting', 'Remove this meeting from history?', [
      { text: 'Cancel', style: 'cancel' },
      {
        text: 'Delete',
        style: 'destructive',
        onPress: async () => {
          await deleteMeeting(id);
          setMeetings((prev) => prev.filter((m) => m.id !== id));
          if (selected?.id === id) setSelected(null);
        },
      },
    ]);
  };

  const handleUpdate = (record: MeetingRecord) => {
    setSelected(record);
    setMeetings((prev) => prev.map((m) => (m.id === record.id ? record : m)));
    updateMeeting(record).catch(() => {});
  };

  if (selected) {
    return (
      <NotesDetail
        meeting={selected}
        onBack={() => setSelected(null)}
        onDelete={handleDelete}
        onUpdate={handleUpdate}
      />
    );
  }

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0F172A" />

      <View style={styles.searchBox}>
        <TextInput
          style={styles.searchInput}
          value={query}
          onChangeText={setQuery}
          placeholder="Search meetings, topics, medical terms…"
          placeholderTextColor="#64748B"
          returnKeyType="search"
          clearButtonMode="while-editing"
        />
        {query !== '' && (
          <TouchableOpacity onPress={() => setQuery('')} hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}>
            <Text style={styles.searchClear}>✕</Text>
          </TouchableOpacity>
        )}
      </View>

      {filtered.length === 0 ? (
        <View style={styles.empty}>
          <Text style={styles.emptyIcon}>📋</Text>
          <Text style={styles.emptyTitle}>{query ? 'No matches' : 'No meetings yet'}</Text>
          <Text style={styles.emptyHint}>
            {query ? 'Try a different search term.' : 'Record your first meeting on the Record tab.'}
          </Text>
        </View>
      ) : (
        <FlatList
          data={filtered}
          keyExtractor={(m) => m.id}
          contentContainerStyle={styles.list}
          renderItem={({ item }) => (
            <TouchableOpacity
              style={styles.card}
              onPress={() => setSelected(item)}
              activeOpacity={0.75}
            >
              <View style={styles.cardHeader}>
                <Text style={styles.cardTitle} numberOfLines={1}>
                  {item.notes.title}
                </Text>
                <Text style={styles.cardDuration}>{formatDuration(item.duration_seconds)}</Text>
              </View>
              <Text style={styles.cardDate}>{formatDate(item.created_at)}</Text>
              <Text style={styles.cardSummary} numberOfLines={2}>
                {applySpeakerNames(item.notes.summary, item.speakerNames)}
              </Text>
              <View style={styles.badgeRow}>
                {item.notes.action_items.length > 0 && (
                  <View style={styles.badge}>
                    <Text style={styles.badgeText}>
                      {item.notes.action_items.length} action items
                    </Text>
                  </View>
                )}
                {(item.notes.conversation?.length ?? 0) > 0 && (
                  <View style={styles.badge}>
                    <Text style={styles.badgeText}>
                      {new Set(item.notes.conversation!.map((t) => t.speaker)).size} speakers
                    </Text>
                  </View>
                )}
                {(item.notes.medical_terms?.length ?? 0) > 0 && (
                  <View style={[styles.badge, styles.badgeAmber]}>
                    <Text style={styles.badgeTextAmber}>
                      {item.notes.medical_terms!.length} medical terms
                    </Text>
                  </View>
                )}
              </View>
            </TouchableOpacity>
          )}
        />
      )}
    </SafeAreaView>
  );
}

function NotesDetail({
  meeting,
  onBack,
  onDelete,
  onUpdate,
}: {
  meeting: MeetingRecord;
  onBack: () => void;
  onDelete: (id: string) => void;
  onUpdate: (record: MeetingRecord) => void;
}) {
  const { notes } = meeting;
  const [showTranscript, setShowTranscript] = useState(false);
  const terms = notes.medical_terms ?? [];

  const handleRename = (speaker: string, name: string) => {
    onUpdate({ ...meeting, speakerNames: { ...meeting.speakerNames, [speaker]: name } });
  };

  const handleEmail = async () => {
    try { await emailNotes(meeting); }
    catch (err) { Alert.alert('Email failed', err instanceof Error ? err.message : String(err)); }
  };

  const handleShare = async () => {
    try { await sharePdf(meeting); }
    catch (err) { Alert.alert('Share failed', err instanceof Error ? err.message : String(err)); }
  };

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor="#0F172A" />

      {/* Top bar */}
      <View style={styles.detailBar}>
        <TouchableOpacity onPress={onBack} style={styles.backBtn}>
          <Text style={styles.backBtnText}>← Back</Text>
        </TouchableOpacity>
        <TouchableOpacity onPress={() => onDelete(meeting.id)} style={styles.deleteBtn}>
          <Text style={styles.deleteBtnText}>Delete</Text>
        </TouchableOpacity>
      </View>

      <FlatList
        data={[]}
        renderItem={null}
        ListHeaderComponent={
          <View style={styles.detailContent}>
            <Text style={styles.detailTitle}>{notes.title}</Text>
            <Text style={styles.detailMeta}>
              {formatDate(meeting.created_at)} · {formatDuration(meeting.duration_seconds)} · {notes.language.toUpperCase()}
            </Text>

            <View style={styles.actionRow}>
              <TouchableOpacity style={styles.actionBtn} onPress={handleEmail}>
                <Text style={styles.actionBtnText}>Email Notes</Text>
              </TouchableOpacity>
              <TouchableOpacity style={styles.actionBtnAlt} onPress={handleShare}>
                <Text style={styles.actionBtnAltText}>Share PDF</Text>
              </TouchableOpacity>
            </View>

            <Section title="Summary">
              <HighlightedText
                text={applySpeakerNames(notes.summary, meeting.speakerNames)}
                extraTerms={terms}
                style={styles.bodyText}
              />
            </Section>

            {terms.length > 0 && (
              <Section title="Medical Terms">
                <View style={styles.chipRow}>
                  {terms.map((t, i) => (
                    <View key={i} style={styles.chip}>
                      <Text style={styles.chipText}>{t}</Text>
                    </View>
                  ))}
                </View>
              </Section>
            )}

            {notes.key_topics.length > 0 && (
              <Section title="Key Topics">
                {notes.key_topics.map((t, i) => (
                  <Text key={i} style={styles.bullet}>• {applySpeakerNames(t, meeting.speakerNames)}</Text>
                ))}
              </Section>
            )}

            {notes.decisions.length > 0 && (
              <Section title="Decisions">
                {notes.decisions.map((d, i) => (
                  <Text key={i} style={styles.bullet}>• {applySpeakerNames(d, meeting.speakerNames)}</Text>
                ))}
              </Section>
            )}

            {notes.action_items.length > 0 && (
              <Section title="Action Items">
                {notes.action_items.map((a, i) => (
                  <ActionItem key={i} text={applySpeakerNames(a, meeting.speakerNames)} />
                ))}
              </Section>
            )}

            {(notes.conversation?.length ?? 0) > 0 && (
              <Section title="Conversation · tap a name to rename">
                <ConversationView
                  turns={notes.conversation!}
                  names={meeting.speakerNames}
                  medicalTerms={terms}
                  onRename={handleRename}
                />
              </Section>
            )}

            <TouchableOpacity
              style={styles.transcriptToggle}
              onPress={() => setShowTranscript((v) => !v)}
            >
              <Text style={styles.transcriptToggleText}>
                {showTranscript ? '▲ Hide transcript' : '▼ Show full transcript'}
              </Text>
            </TouchableOpacity>

            {showTranscript && (
              <Section title="Full Transcript">
                <HighlightedText
                  text={meeting.transcript.text}
                  extraTerms={terms}
                  style={styles.transcriptText}
                />
              </Section>
            )}
          </View>
        }
      />
    </SafeAreaView>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <View style={styles.section}>
      <Text style={styles.sectionTitle}>{title}</Text>
      {children}
    </View>
  );
}

function ActionItem({ text }: { text: string }) {
  const [checked, setChecked] = useState(false);
  return (
    <TouchableOpacity
      style={styles.actionItem}
      onPress={() => setChecked((v) => !v)}
      activeOpacity={0.7}
    >
      <Text style={[styles.checkbox, checked && styles.checkboxDone]}>
        {checked ? '☑' : '☐'}
      </Text>
      <Text style={[styles.actionText, checked && styles.actionTextDone]}>{text}</Text>
    </TouchableOpacity>
  );
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString(undefined, {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function formatDuration(seconds: number) {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  return m > 0 ? `${m}m ${s}s` : `${s}s`;
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: '#0F172A' },
  searchBox: {
    flexDirection: 'row', alignItems: 'center', gap: 8,
    margin: 16, marginBottom: 4,
    backgroundColor: '#1E293B', borderRadius: 12, paddingHorizontal: 14,
    borderWidth: 1, borderColor: '#334155',
  },
  searchInput: { flex: 1, color: '#F1F5F9', fontSize: 14, paddingVertical: 12 },
  searchClear: { color: '#64748B', fontSize: 14, padding: 4 },
  empty: { flex: 1, alignItems: 'center', justifyContent: 'center', gap: 12 },
  emptyIcon: { fontSize: 48 },
  emptyTitle: { fontSize: 20, fontWeight: '600', color: '#F1F5F9' },
  emptyHint: { fontSize: 14, color: '#64748B', textAlign: 'center', maxWidth: 260 },
  list: { padding: 16, gap: 12 },
  card: {
    backgroundColor: '#1E293B',
    borderRadius: 14,
    padding: 16,
    gap: 6,
    borderWidth: 1,
    borderColor: '#334155',
  },
  cardHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'flex-start' },
  cardTitle: { fontSize: 16, fontWeight: '700', color: '#F1F5F9', flex: 1, marginRight: 8 },
  cardDuration: { fontSize: 12, color: '#64748B' },
  cardDate: { fontSize: 12, color: '#64748B' },
  cardSummary: { fontSize: 13, color: '#94A3B8', lineHeight: 18 },
  badgeRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6, marginTop: 2 },
  badge: {
    alignSelf: 'flex-start',
    backgroundColor: '#1E3A5F',
    borderRadius: 6,
    paddingHorizontal: 8,
    paddingVertical: 3,
  },
  badgeText: { fontSize: 11, color: '#60A5FA', fontWeight: '600' },
  badgeAmber: { backgroundColor: 'rgba(250, 204, 21, 0.12)' },
  badgeTextAmber: { fontSize: 11, color: '#FDE047', fontWeight: '600' },
  // Detail view
  detailBar: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    paddingHorizontal: 16,
    paddingVertical: 10,
    borderBottomWidth: 1,
    borderBottomColor: '#1E293B',
  },
  backBtn: { padding: 4, minHeight: 44, justifyContent: 'center' },
  backBtnText: { color: '#60A5FA', fontSize: 15, fontWeight: '600' },
  deleteBtn: { padding: 4, minHeight: 44, justifyContent: 'center' },
  deleteBtnText: { color: '#EF4444', fontSize: 15 },
  detailContent: { padding: 20, gap: 4 },
  detailTitle: { fontSize: 22, fontWeight: '800', color: '#F1F5F9', lineHeight: 28 },
  detailMeta: { fontSize: 12, color: '#64748B', marginBottom: 8 },
  actionRow: { flexDirection: 'row', gap: 10, marginTop: 4, marginBottom: 8 },
  actionBtn: { flex: 1, paddingVertical: 12, backgroundColor: '#1D4ED8', borderRadius: 10, alignItems: 'center', minHeight: 44, justifyContent: 'center' },
  actionBtnText: { color: '#fff', fontWeight: '700', fontSize: 14 },
  actionBtnAlt: { flex: 1, paddingVertical: 12, backgroundColor: '#0F172A', borderRadius: 10, alignItems: 'center', borderWidth: 1, borderColor: '#3B82F6', minHeight: 44, justifyContent: 'center' },
  actionBtnAltText: { color: '#60A5FA', fontWeight: '700', fontSize: 14 },
  section: { marginTop: 16, gap: 8 },
  sectionTitle: { fontSize: 13, fontWeight: '700', color: '#3B82F6', textTransform: 'uppercase', letterSpacing: 0.8 },
  bodyText: { fontSize: 15, color: '#CBD5E1', lineHeight: 22 },
  bullet: { fontSize: 14, color: '#94A3B8', lineHeight: 20, paddingLeft: 4 },
  chipRow: { flexDirection: 'row', flexWrap: 'wrap', gap: 6 },
  chip: { backgroundColor: 'rgba(250, 204, 21, 0.15)', borderRadius: 6, paddingHorizontal: 8, paddingVertical: 3, borderWidth: 1, borderColor: 'rgba(250, 204, 21, 0.35)' },
  chipText: { fontSize: 12, color: '#FDE047', fontWeight: '600' },
  actionItem: { flexDirection: 'row', gap: 10, alignItems: 'flex-start', paddingVertical: 4 },
  checkbox: { fontSize: 18, color: '#3B82F6' },
  checkboxDone: { color: '#22C55E' },
  actionText: { fontSize: 14, color: '#CBD5E1', flex: 1, lineHeight: 20 },
  actionTextDone: { textDecorationLine: 'line-through', color: '#475569' },
  transcriptToggle: { marginTop: 20, paddingVertical: 10, minHeight: 44, justifyContent: 'center' },
  transcriptToggleText: { color: '#475569', fontSize: 13 },
  transcriptText: { fontSize: 13, color: '#475569', lineHeight: 20, fontFamily: Platform.OS === 'ios' ? 'Menlo' : 'monospace' },
});
