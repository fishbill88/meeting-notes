import { initLlama, LlamaContext } from 'llama.rn';
import { getActiveLlmModel, modelPath, LlmModelInfo } from './models';
import { formatConversation } from './diarize';
import type { NotesData, SpeakerSegment } from './types';

let _ctx: LlamaContext | null = null;
let _ctxFilename: string | null = null;

// n_ctx 4096 ≈ 117 MB KV cache for a 1.5B GQA model — fits phones fine and
// lets a single pass cover ~6k chars of transcript.
const N_CTX = 4096;
/** Transcripts longer than this go through chunked map-reduce summarization. */
const SINGLE_PASS_CHAR_LIMIT = 6000;
const CHUNK_CHARS = 5000;

export async function getLlamaContext(): Promise<{ ctx: LlamaContext; model: LlmModelInfo }> {
  const model = await getActiveLlmModel();
  if (_ctx && _ctxFilename === model.filename) return { ctx: _ctx, model };

  if (_ctx) { await _ctx.release().catch(() => {}); _ctx = null; _ctxFilename = null; }
  try {
    _ctx = await initLlama({
      model: modelPath(model.filename),
      n_ctx: N_CTX,
      n_batch: 128,
      // 6 threads engages the big cores on modern 6–8 core phones (matches
      // the Whisper full pass in whisper.ts)
      n_threads: 6,
    });
    _ctxFilename = model.filename;
  } catch (err) {
    if (err instanceof Error) throw err;
    const e = err as Record<string, unknown>;
    throw new Error(String(e?.message ?? e?.error ?? JSON.stringify(err)));
  }
  return { ctx: _ctx, model };
}

export function releaseLlama(): void {
  _ctx?.release();
  _ctx = null;
  _ctxFilename = null;
}

export async function generateNotes(
  transcript: string,
  language = 'en',
  speakers: SpeakerSegment[] = [],
  opts: { onStep?: (step: string) => void } = {},
): Promise<NotesData> {
  if (!transcript.trim()) {
    return extractiveFallback(transcript, language);
  }

  const body = speakers.length ? formatConversation(speakers) : transcript;

  try {
    const { ctx, model } = await getLlamaContext();

    if (body.length <= SINGLE_PASS_CHAR_LIMIT) {
      const raw = await complete(ctx, model, buildPrompt(body, language, speakers.length > 0));
      return parseNotes(raw, language);
    }

    // Map-reduce: summarize each part, then combine part notes into the final set
    const chunks = splitChunks(body, CHUNK_CHARS);
    const partNotes: string[] = [];
    for (let i = 0; i < chunks.length; i++) {
      opts.onStep?.(`Summarizing part ${i + 1} of ${chunks.length}…`);
      const raw = await complete(ctx, model, buildChunkPrompt(chunks[i], language, i + 1, chunks.length), CHUNK_SCHEMA);
      partNotes.push(raw);
    }
    opts.onStep?.('Combining notes…');
    const combined = partNotes
      .map((raw, i) => `PART ${i + 1}: ${compactPartJson(raw)}`)
      .join('\n');
    const finalRaw = await complete(
      ctx, model, buildReducePrompt(combined.slice(0, SINGLE_PASS_CHAR_LIMIT), language),
    );
    return parseNotes(finalRaw, language);
  } catch {
    return extractiveFallback(transcript, language);
  }
}

async function complete(
  ctx: LlamaContext,
  model: LlmModelInfo,
  prompt: string,
  schema: object = NOTES_SCHEMA,
): Promise<string> {
  const result = await ctx.completion({
    messages: [
      {
        role: 'system',
        content:
          'You are an expert meeting note taker. Respond ONLY with valid JSON — no markdown fences, no explanation.',
      },
      { role: 'user', content: prompt + (model.promptSuffix ?? '') },
    ],
    n_predict: 1024,
    temperature: 0.1,
    stop: model.stopTokens,
    // Grammar-constrained decoding: llama.cpp converts the schema to GBNF and
    // the sampler cannot emit invalid JSON — works on every model. The
    // extractJson/parseNotes fallbacks stay as a safety net.
    response_format: {
      type: 'json_schema',
      json_schema: { schema },
    },
  });
  return result.text;
}

// ---------------------------------------------------------------------------
// Output schemas (constrained generation)
// ---------------------------------------------------------------------------

const STRING_ARRAY = { type: 'array', items: { type: 'string' } };

/** Matches NOTES_FIELDS / NotesData. */
const NOTES_SCHEMA = {
  type: 'object',
  properties: {
    title: { type: 'string' },
    summary: { type: 'string' },
    key_topics: STRING_ARRAY,
    decisions: STRING_ARRAY,
    action_items: STRING_ARRAY,
    medical_terms: STRING_ARRAY,
  },
  required: ['title', 'summary', 'key_topics', 'decisions', 'action_items', 'medical_terms'],
} as const;

/** Per-chunk notes in map-reduce mode (no title). */
const CHUNK_SCHEMA = {
  type: 'object',
  properties: {
    summary: { type: 'string' },
    key_topics: STRING_ARRAY,
    decisions: STRING_ARRAY,
    action_items: STRING_ARRAY,
    medical_terms: STRING_ARRAY,
  },
  required: ['summary', 'key_topics', 'decisions', 'action_items', 'medical_terms'],
} as const;

// ---------------------------------------------------------------------------
// Prompt builders
// ---------------------------------------------------------------------------

const NOTES_FIELDS = `- "title": short descriptive title (string)
- "summary": 2-4 sentence summary (string)
- "key_topics": main topics discussed (string[])
- "decisions": decisions made (string[])
- "action_items": follow-up tasks (string[])
- "medical_terms": medical or health-related terms mentioned — medications, dosages, diagnoses, procedures (string[], empty if none)`;

function buildPrompt(body: string, language: string, isConversation: boolean): string {
  const langHint = language !== 'en' ? ` (transcript language: ${language})` : '';
  const label = isConversation ? 'CONVERSATION' : 'TRANSCRIPT';

  return `Given the following meeting${langHint}, produce structured meeting notes as JSON with these fields:
${NOTES_FIELDS}
${isConversation ? 'When speakers are labeled, mention who said or agreed to what in the summary and action items.\n' : ''}
Respond ONLY with valid JSON.

${label}:
${body}

JSON:`;
}

function buildChunkPrompt(chunk: string, language: string, part: number, total: number): string {
  const langHint = language !== 'en' ? ` (transcript language: ${language})` : '';
  return `This is part ${part} of ${total} of a longer meeting${langHint}. Summarize JUST this part as JSON with these fields:
- "summary": 2-3 sentence summary of this part (string)
- "key_topics": topics discussed in this part (string[])
- "decisions": decisions made in this part (string[])
- "action_items": follow-up tasks from this part (string[])
- "medical_terms": medical terms mentioned in this part (string[], empty if none)

Respond ONLY with valid JSON.

TRANSCRIPT PART:
${chunk}

JSON:`;
}

function buildReducePrompt(combined: string, language: string): string {
  const langHint = language !== 'en' ? ` (transcript language: ${language})` : '';
  return `Below are part-by-part notes from one meeting${langHint}. Merge them into final meeting notes as JSON with these fields:
${NOTES_FIELDS}

Merge duplicates, keep every distinct decision and action item. Respond ONLY with valid JSON.

PART NOTES:
${combined}

JSON:`;
}

// ---------------------------------------------------------------------------
// Parsing / chunking helpers
// ---------------------------------------------------------------------------

/** Splits at line/sentence boundaries, keeping chunks under maxChars. */
function splitChunks(text: string, maxChars: number): string[] {
  const units = text.includes('\n')
    ? text.split('\n')
    : (text.match(/[^.!?]+[.!?]+/g) ?? [text]);

  const chunks: string[] = [];
  let cur = '';
  for (const unit of units) {
    const sep = cur ? (text.includes('\n') ? '\n' : ' ') : '';
    if (cur.length + unit.length + sep.length > maxChars && cur) {
      chunks.push(cur);
      cur = unit;
    } else {
      cur += sep + unit;
    }
  }
  if (cur.trim()) chunks.push(cur);
  return chunks;
}

/** Extracts the JSON object from a raw completion, stripped of fences. */
function extractJson(raw: string): string {
  // Reasoning models (Qwen3) may emit <think>…</think> before the answer
  const cleaned = raw.replace(/<think>[\s\S]*?(<\/think>|$)/g, '');
  const fenceMatch = cleaned.match(/```(?:json)?\s*([\s\S]+?)\s*```/);
  let json = fenceMatch ? fenceMatch[1] : cleaned.trim();
  const braceMatch = json.match(/\{[\s\S]*\}/);
  if (braceMatch) json = braceMatch[0];
  return json;
}

/** Re-serializes a part's notes compactly; falls back to raw text on bad JSON. */
function compactPartJson(raw: string): string {
  try {
    return JSON.stringify(JSON.parse(extractJson(raw)));
  } catch {
    return raw.trim().slice(0, 600);
  }
}

function parseNotes(raw: string, language: string): NotesData {
  const data = JSON.parse(extractJson(raw));
  const arr = (v: unknown): string[] =>
    Array.isArray(v) ? v.map(String).filter(Boolean) : [];
  return {
    title: typeof data.title === 'string' && data.title ? data.title : 'Meeting Notes',
    summary: typeof data.summary === 'string' ? data.summary : '',
    key_topics: arr(data.key_topics),
    decisions: arr(data.decisions),
    action_items: arr(data.action_items),
    medical_terms: arr(data.medical_terms),
    language,
  };
}

// ---------------------------------------------------------------------------
// Extractive fallback (pure JS, no model needed)
// ---------------------------------------------------------------------------

export function extractiveFallback(transcript: string, language: string): NotesData {
  const sentences = transcript.match(/[^.!?]+[.!?]+/g) ?? [transcript];
  const words = transcript.toLowerCase().match(/\b\w+\b/g) ?? [];

  const stopwords = new Set([
    'the','a','an','is','are','was','were','be','been','and','or','but',
    'in','on','at','to','for','of','with','it','this','that','we','i',
    'he','she','they','you','ang','ng','sa','na','at','ay','ko','mo',
  ]);

  const freq: Record<string, number> = {};
  for (const w of words) if (!stopwords.has(w)) freq[w] = (freq[w] ?? 0) + 1;

  const scored = sentences.map((s) => {
    const sw = s.toLowerCase().match(/\b\w+\b/g) ?? [];
    const score = sw.reduce((acc, w) => acc + (freq[w] ?? 0), 0) / Math.max(sw.length, 1);
    return { s, score };
  });

  const topN = Math.min(5, Math.max(1, Math.floor(sentences.length / 4)));
  const summary = [...scored]
    .sort((a, b) => b.score - a.score)
    .slice(0, topN)
    .sort((a, b) => sentences.indexOf(a.s) - sentences.indexOf(b.s))
    .map((x) => x.s.trim())
    .join(' ');

  const actionKeywords = language === 'tl'
    ? ['kailangang','gagawin','dapat','follow up','tapusin']
    : ['will','should','must','need to','action','follow up','by ','todo','deadline'];

  const actionItems = sentences
    .filter((s) => actionKeywords.some((k) => s.toLowerCase().includes(k)))
    .slice(0, 6)
    .map((s) => s.trim());

  const keyTopics = [...scored]
    .sort((a, b) => b.score - a.score)
    .slice(0, 5)
    .map((x) => x.s.trim().split(/\s+/).slice(0, 8).join(' '));

  const title = transcript.trim().split(/\s+/).slice(0, 6).join(' ');

  return {
    title, summary, key_topics: keyTopics, decisions: [],
    action_items: actionItems, medical_terms: [], language,
  };
}
