import { memo, useState } from 'react';
import {
  Modal,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View,
} from 'react-native';
import type { SpeakerSegment } from '../lib/types';
import { HighlightedText } from './HighlightedText';

/** Distinct speaker colors tuned for the dark slate background. */
const SPEAKER_COLORS_DARK = ['#60A5FA', '#C4B5FD', '#5EEAD4', '#FCD34D'];

export function speakerColorDark(speaker: string): string {
  const n = parseInt(speaker.replace(/\D/g, ''), 10);
  const idx = (isNaN(n) ? 1 : n) - 1;
  return SPEAKER_COLORS_DARK[((idx % SPEAKER_COLORS_DARK.length) + SPEAKER_COLORS_DARK.length) % SPEAKER_COLORS_DARK.length];
}

function fmtTime(cs: number): string {
  const totalSec = Math.floor(cs / 100);
  const m = Math.floor(totalSec / 60);
  const s = totalSec % 60;
  return `${m}:${s.toString().padStart(2, '0')}`;
}

interface Props {
  turns: SpeakerSegment[];
  /** User-assigned display names, e.g. { 'Person 1': 'Dr. Cruz' } */
  names?: Record<string, string>;
  /** Terms to highlight on top of the built-in medical lexicon. */
  medicalTerms?: string[];
  /** When provided, tapping a speaker name opens the rename dialog. */
  onRename?: (speaker: string, name: string) => void;
  /** Show only the first N turns (e.g., for the post-recording preview). */
  maxTurns?: number;
}

function ConversationViewInner({ turns, names, medicalTerms = [], onRename, maxTurns }: Props) {
  const [renaming, setRenaming] = useState<string | null>(null);
  const [draftName, setDraftName] = useState('');

  const visible = maxTurns ? turns.slice(0, maxTurns) : turns;
  const hiddenCount = turns.length - visible.length;

  const openRename = (speaker: string) => {
    if (!onRename) return;
    setDraftName(names?.[speaker] ?? '');
    setRenaming(speaker);
  };

  const commitRename = () => {
    if (renaming && draftName.trim()) onRename?.(renaming, draftName.trim());
    setRenaming(null);
  };

  return (
    <View style={styles.container}>
      {visible.map((turn, i) => {
        const color = speakerColorDark(turn.speaker);
        const display = names?.[turn.speaker] ?? turn.speaker;
        return (
          <View key={`${turn.t0}-${i}`} style={[styles.turn, { borderLeftColor: color }]}>
            <View style={styles.turnHeader}>
              <TouchableOpacity
                onPress={() => openRename(turn.speaker)}
                disabled={!onRename}
                hitSlop={{ top: 10, bottom: 10, left: 10, right: 10 }}
              >
                <Text style={[styles.speaker, { color }]}>
                  {display}{onRename ? '  ✎' : ''}
                </Text>
              </TouchableOpacity>
              <Text style={styles.time}>{fmtTime(turn.t0)}</Text>
            </View>
            <HighlightedText text={turn.text} extraTerms={medicalTerms} style={styles.text} />
          </View>
        );
      })}

      {hiddenCount > 0 && (
        <Text style={styles.more}>+ {hiddenCount} more turns…</Text>
      )}

      <Modal visible={renaming !== null} transparent animationType="fade" onRequestClose={() => setRenaming(null)}>
        <View style={styles.modalBackdrop}>
          <View style={styles.modalCard}>
            <Text style={styles.modalTitle}>Rename {renaming}</Text>
            <TextInput
              style={styles.modalInput}
              value={draftName}
              onChangeText={setDraftName}
              placeholder="e.g. Dr. Cruz"
              placeholderTextColor="#64748B"
              autoFocus
              onSubmitEditing={commitRename}
              returnKeyType="done"
            />
            <View style={styles.modalActions}>
              <TouchableOpacity style={styles.modalBtn} onPress={() => setRenaming(null)}>
                <Text style={styles.modalBtnCancel}>Cancel</Text>
              </TouchableOpacity>
              <TouchableOpacity
                style={[styles.modalBtn, styles.modalBtnPrimary]}
                onPress={commitRename}
                disabled={!draftName.trim()}
              >
                <Text style={styles.modalBtnSave}>Save</Text>
              </TouchableOpacity>
            </View>
          </View>
        </View>
      </Modal>
    </View>
  );
}

export const ConversationView = memo(ConversationViewInner);

const styles = StyleSheet.create({
  container: { gap: 10 },
  turn: {
    borderLeftWidth: 3,
    paddingLeft: 10,
    gap: 2,
  },
  turnHeader: { flexDirection: 'row', justifyContent: 'space-between', alignItems: 'center' },
  speaker: { fontSize: 12, fontWeight: '700', textTransform: 'uppercase', letterSpacing: 0.5, paddingVertical: 4 },
  time: { fontSize: 11, color: '#475569' },
  text: { fontSize: 14, color: '#CBD5E1', lineHeight: 20 },
  more: { fontSize: 12, color: '#64748B', fontStyle: 'italic' },
  modalBackdrop: {
    flex: 1, backgroundColor: 'rgba(0,0,0,0.6)',
    alignItems: 'center', justifyContent: 'center', padding: 32,
  },
  modalCard: {
    width: '100%', backgroundColor: '#1E293B', borderRadius: 16, padding: 20, gap: 14,
    borderWidth: 1, borderColor: '#334155',
  },
  modalTitle: { fontSize: 16, fontWeight: '700', color: '#F1F5F9' },
  modalInput: {
    backgroundColor: '#0F172A', borderRadius: 10, borderWidth: 1, borderColor: '#334155',
    color: '#F1F5F9', paddingHorizontal: 14, paddingVertical: 10, fontSize: 15,
  },
  modalActions: { flexDirection: 'row', justifyContent: 'flex-end', gap: 10 },
  modalBtn: { paddingVertical: 10, paddingHorizontal: 18, borderRadius: 8, minHeight: 44, justifyContent: 'center' },
  modalBtnPrimary: { backgroundColor: '#1D4ED8' },
  modalBtnCancel: { color: '#94A3B8', fontWeight: '600', fontSize: 14 },
  modalBtnSave: { color: '#fff', fontWeight: '700', fontSize: 14 },
});
