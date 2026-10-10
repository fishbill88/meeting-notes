/**
 * Turns a MeetingRecord into a polished HTML document, then emails it
 * (opens the phone's mail app pre-filled, PDF attached) or shares the PDF
 * via the system share sheet.
 */
import * as MailComposer from 'expo-mail-composer';
import * as Print from 'expo-print';
import * as Sharing from 'expo-sharing';
import { highlightSpans } from './medicalTerms';
import type { MeetingRecord, SpeakerSegment } from './types';

const SPEAKER_COLORS = ['#1D4ED8', '#7C3AED', '#0F766E', '#B45309'];

export function speakerColor(speaker: string): string {
  const n = parseInt(speaker.replace(/\D/g, ''), 10);
  const idx = (isNaN(n) ? 1 : n) - 1;
  return SPEAKER_COLORS[((idx % SPEAKER_COLORS.length) + SPEAKER_COLORS.length) % SPEAKER_COLORS.length];
}

/** Display name for a speaker label, honoring user renames. */
export function speakerName(record: MeetingRecord, speaker: string): string {
  return record.speakerNames?.[speaker] ?? speaker;
}

/** Replaces "Person N" mentions (e.g., in the LLM summary) with renamed speakers. */
export function applySpeakerNames(text: string, names?: Record<string, string>): string {
  if (!names) return text;
  let out = text;
  for (const [label, name] of Object.entries(names)) {
    if (name && name !== label) out = out.split(label).join(name);
  }
  return out;
}

function esc(s: string): string {
  return s
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;');
}

function highlightHtml(text: string, extraTerms: string[]): string {
  return highlightSpans(text, extraTerms)
    .map((span) =>
      span.highlighted
        ? `<b style="background:#FEF08A;padding:0 2px;border-radius:3px">${esc(span.text)}</b>`
        : esc(span.text),
    )
    .join('');
}

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    weekday: 'short', month: 'short', day: 'numeric', year: 'numeric',
    hour: '2-digit', minute: '2-digit',
  });
}

function conversationHtml(record: MeetingRecord, terms: string[]): string {
  const turns: SpeakerSegment[] = record.notes.conversation ?? [];
  if (!turns.length) return '';
  const rows = turns
    .map((t) => {
      const name = speakerName(record, t.speaker);
      return `<p style="margin:6px 0">
        <span style="color:${speakerColor(t.speaker)};font-weight:700">${esc(name)}:</span>
        ${highlightHtml(t.text, terms)}
      </p>`;
    })
    .join('\n');
  return `<h2 style="font-size:15px;color:#1D4ED8;margin:18px 0 6px">Conversation</h2>${rows}`;
}

function listHtml(title: string, items: string[], names?: Record<string, string>, prefix = '•'): string {
  if (!items.length) return '';
  const lis = items
    .map((i) => `<li style="margin:3px 0">${prefix === '•' ? '' : `${prefix} `}${esc(applySpeakerNames(i, names))}</li>`)
    .join('');
  return `<h2 style="font-size:15px;color:#1D4ED8;margin:18px 0 6px">${esc(title)}</h2>
    <ul style="margin:0;padding-left:20px">${lis}</ul>`;
}

export function notesToHtml(record: MeetingRecord): string {
  const { notes, transcript } = record;
  const names = record.speakerNames;
  const terms = notes.medical_terms ?? [];
  const durationMin = Math.round(record.duration_seconds / 60);

  const termChips = terms.length
    ? `<h2 style="font-size:15px;color:#1D4ED8;margin:18px 0 6px">Medical Terms Mentioned</h2>
       <p>${terms.map((t) => `<span style="background:#FEF08A;border-radius:4px;padding:2px 6px;margin-right:4px;display:inline-block">${esc(t)}</span>`).join(' ')}</p>`
    : '';

  return `<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="font-family:-apple-system,Segoe UI,Roboto,sans-serif;color:#0F172A;max-width:720px;margin:0 auto;padding:24px;font-size:14px;line-height:1.55">
  <h1 style="font-size:22px;margin:0 0 4px">${esc(notes.title)}</h1>
  <p style="color:#64748B;margin:0 0 16px">${esc(formatDate(record.created_at))} · ${durationMin > 0 ? `${durationMin} min` : `${record.duration_seconds}s`} · ${esc(notes.language.toUpperCase())}</p>

  <h2 style="font-size:15px;color:#1D4ED8;margin:18px 0 6px">Summary</h2>
  <p style="margin:0">${highlightHtml(applySpeakerNames(notes.summary, names), terms)}</p>

  ${listHtml('Key Topics', notes.key_topics, names)}
  ${listHtml('Decisions', notes.decisions, names)}
  ${listHtml('Action Items', notes.action_items, names, '☐')}
  ${termChips}
  ${conversationHtml(record, terms)}

  <h2 style="font-size:15px;color:#1D4ED8;margin:18px 0 6px">Full Transcript</h2>
  <p style="color:#475569;white-space:pre-wrap">${highlightHtml(transcript.text, terms)}</p>

  <p style="color:#94A3B8;font-size:11px;margin-top:24px">Generated offline by Meeting Notes AI</p>
</body>
</html>`;
}

export async function exportPdf(record: MeetingRecord): Promise<string> {
  const { uri } = await Print.printToFileAsync({ html: notesToHtml(record) });
  return uri;
}

export async function emailNotes(record: MeetingRecord): Promise<void> {
  const available = await MailComposer.isAvailableAsync();
  if (!available) {
    throw new Error('No mail app is configured on this device.');
  }
  const pdfUri = await exportPdf(record);
  await MailComposer.composeAsync({
    subject: `Meeting Notes: ${record.notes.title} — ${new Date(record.created_at).toLocaleDateString()}`,
    body: notesToHtml(record),
    isHtml: true,
    attachments: [pdfUri],
  });
}

export async function sharePdf(record: MeetingRecord): Promise<void> {
  const available = await Sharing.isAvailableAsync();
  if (!available) throw new Error('Sharing is not available on this device.');
  const pdfUri = await exportPdf(record);
  await Sharing.shareAsync(pdfUri, {
    mimeType: 'application/pdf',
    dialogTitle: record.notes.title,
  });
}
