/**
 * Medical term detection for transcript highlighting.
 *
 * Ported from `mobile/lib/medicalTerms.ts`. The lexicon is identical — keep
 * the two in sync — but the range-merging code is written against this
 * project's stricter `noUncheckedIndexedAccess` setting rather than using
 * non-null assertions.
 *
 * A curated lexicon (English + Tagalog + common PH brand names) is matched
 * against the transcript by `findMedicalTerms`. Matching is case-insensitive
 * on word boundaries, longest term first so "blood pressure" wins over
 * "blood".
 *
 * `findMedicalTerms` is the *only* place the lexicon runs, and it is called
 * only for the consultation template (see `App.tsx`). `highlightSpans`
 * highlights the terms it is handed and nothing else, so a corporate meeting
 * — which carries no `medical_terms` — renders with no highlighting at all.
 * Everything here is dormant until the clinical phase resumes.
 */

const CONDITIONS = [
  'hypertension', 'diabetes', 'diabetes mellitus', 'asthma', 'tuberculosis',
  'pneumonia', 'bronchitis', 'dengue', 'typhoid', 'malaria', 'measles',
  'chickenpox', 'influenza', 'covid', 'covid-19', 'uti', 'urinary tract infection',
  'gastritis', 'ulcer', 'hyperacidity', 'anemia', 'arthritis', 'gout',
  'migraine', 'vertigo', 'allergy', 'allergic rhinitis', 'sinusitis',
  'tonsillitis', 'pharyngitis', 'conjunctivitis', 'sore eyes', 'otitis media',
  'dermatitis', 'eczema', 'psoriasis', 'scabies', 'hepatitis', 'cirrhosis',
  'kidney stones', 'nephritis', 'goiter', 'hyperthyroidism', 'hypothyroidism',
  'stroke', 'heart attack', 'myocardial infarction', 'angina', 'arrhythmia',
  'heart failure', 'atrial fibrillation', 'epilepsy', 'seizure', 'depression',
  'anxiety', 'insomnia', 'obesity', 'malnutrition', 'dehydration', 'sepsis',
  'cancer', 'tumor', 'leukemia', 'lymphoma', 'fracture', 'sprain', 'scoliosis',
  'cataract', 'glaucoma', 'appendicitis', 'hernia', 'hemorrhoids', 'gerd',
  'acid reflux', 'fatty liver', 'osteoporosis', 'rheumatic', 'lupus',
  'altapresyon', 'diperensya sa puso', 'bukol', 'pasma', 'beke', 'bulutong',
  'tigdas', 'tulo', 'almuranas',
];

const SYMPTOMS = [
  'fever', 'cough', 'colds', 'headache', 'dizziness', 'nausea', 'vomiting',
  'diarrhea', 'constipation', 'fatigue', 'chest pain', 'shortness of breath',
  'palpitations', 'rashes', 'itching', 'swelling', 'numbness', 'cramps',
  'back pain', 'sore throat', 'runny nose', 'loss of appetite', 'weight loss',
  'blurred vision', 'bleeding', 'wheezing', 'phlegm', 'body malaise',
  'lagnat', 'ubo', 'sipon', 'hilo', 'pagkahilo', 'suka', 'pagsusuka',
  'pagtatae', 'tibi', 'sakit ng ulo', 'sakit ng tiyan', 'masakit ang dibdib',
  'hirap huminga', 'pamamaga', 'pangangati', 'pantal', 'manas', 'panginginig',
  'pagduduwal', 'plema', 'hapo', 'panghihina',
];

const MEDICATIONS = [
  'paracetamol', 'acetaminophen', 'ibuprofen', 'aspirin', 'mefenamic acid',
  'naproxen', 'diclofenac', 'celecoxib', 'tramadol', 'amoxicillin',
  'co-amoxiclav', 'ampicillin', 'azithromycin', 'erythromycin', 'cephalexin',
  'cefuroxime', 'ceftriaxone', 'ciprofloxacin', 'ofloxacin', 'doxycycline',
  'clindamycin', 'metronidazole', 'cotrimoxazole', 'losartan', 'amlodipine',
  'metoprolol', 'atenolol', 'carvedilol', 'enalapril', 'captopril',
  'telmisartan', 'valsartan', 'clonidine', 'furosemide', 'hydrochlorothiazide',
  'spironolactone', 'metformin', 'gliclazide', 'glimepiride', 'insulin',
  'sitagliptin', 'atorvastatin', 'rosuvastatin', 'simvastatin', 'fenofibrate',
  'omeprazole', 'esomeprazole', 'pantoprazole', 'ranitidine', 'famotidine',
  'antacid', 'salbutamol', 'budesonide', 'fluticasone', 'montelukast',
  'cetirizine', 'loratadine', 'fexofenadine', 'diphenhydramine',
  'chlorphenamine', 'dexamethasone', 'prednisone', 'prednisolone',
  'hydrocortisone', 'methylprednisolone', 'lagundi', 'sambong', 'carbocisteine',
  'ambroxol', 'guaifenesin', 'dextromethorphan', 'loperamide',
  'oral rehydration salts', 'oresol', 'multivitamins', 'ferrous sulfate',
  'folic acid', 'vitamin c', 'vitamin d', 'vitamin b complex', 'zinc',
  'calcium', 'potassium', 'warfarin', 'clopidogrel', 'apixaban', 'rivaroxaban',
  'levothyroxine', 'methimazole', 'allopurinol', 'colchicine', 'gabapentin',
  'pregabalin', 'amitriptyline', 'sertraline', 'fluoxetine', 'escitalopram',
  'alprazolam', 'diazepam', 'biogesic', 'neozep', 'bioflu', 'alaxan',
  'medicol', 'dolfenal', 'solmux', 'tuseran', 'decolgen', 'tempra',
  'kremil-s', 'diatabs', 'imodium', 'buscopan', 'ventolin', 'norvasc',
  'antibiotic', 'antibiotics', 'painkiller', 'pain reliever', 'antihistamine',
  'gamot', 'antibiyotiko', 'bakuna', 'reseta',
];

const PROCEDURES = [
  'x-ray', 'xray', 'ultrasound', 'ct scan', 'mri', 'ecg', 'ekg', '2d echo',
  'echocardiogram', 'cbc', 'complete blood count', 'urinalysis', 'fecalysis',
  'blood test', 'blood chemistry', 'fasting blood sugar', 'fbs', 'hba1c',
  'lipid profile', 'cholesterol test', 'creatinine', 'bun', 'sgpt', 'sgot',
  'liver function test', 'kidney function test', 'thyroid function test',
  'tsh', 'ft4', 'biopsy', 'endoscopy', 'colonoscopy', 'dialysis',
  'chemotherapy', 'radiation therapy', 'surgery', 'operation', 'vaccination',
  'immunization', 'swab test', 'rapid antigen', 'rt-pcr', 'pap smear',
  'mammogram', 'blood transfusion', 'iv fluids', 'intubation', 'nebulization',
  'nebulizer', 'physical therapy', 'opera', 'pagpapaopera', 'pagsusuri ng dugo',
  'ihi', 'dumi',
];

const VITALS_ANATOMY = [
  'blood pressure', 'bp', 'heart rate', 'pulse rate', 'respiratory rate',
  'temperature', 'oxygen saturation', 'blood sugar', 'glucose', 'cholesterol',
  'triglycerides', 'uric acid', 'hemoglobin', 'platelet', 'white blood cell',
  'kidney', 'liver', 'lungs', 'heart', 'pancreas', 'gallbladder', 'thyroid',
  'spine', 'artery', 'vein', 'presyon', 'puso', 'baga', 'atay', 'bato sa katawan',
  'apdo', 'lapay', 'bituka', 'sikmura',
];

const DOSAGE_PATTERNS = [
  /\b\d+(?:\.\d+)?\s?(?:mg|mcg|ml|cc|iu|units?)\b/gi,
  /\b(?:once|twice|thrice|three times|3x|2x|1x)\s+(?:a|per)\s+day\b/gi,
  /\bevery\s+\d+\s+hours?\b/gi,
  /\b(?:tuwing|bawat)\s+\d+\s+oras\b/gi,
];

const LEXICON = [
  ...CONDITIONS, ...SYMPTOMS, ...MEDICATIONS, ...PROCEDURES, ...VITALS_ANATOMY,
]
  // Longest first so multi-word terms win over their substrings
  .sort((a, b) => b.length - a.length);

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

const LEXICON_RE = new RegExp(
  `\\b(?:${LEXICON.map(escapeRegExp).join('|')})\\b`,
  'gi',
);

/** Unique medical terms found in the text, in order of first appearance. */
export function findMedicalTerms(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];

  for (const match of text.matchAll(LEXICON_RE)) {
    const key = match[0].toLowerCase();
    if (!seen.has(key)) {
      seen.add(key);
      out.push(match[0]);
    }
  }
  for (const re of DOSAGE_PATTERNS) {
    for (const match of text.matchAll(re)) {
      const key = match[0].toLowerCase();
      if (!seen.has(key)) {
        seen.add(key);
        out.push(match[0]);
      }
    }
  }
  return out;
}

export interface TextSpan {
  text: string;
  highlighted: boolean;
}

type Range = { start: number; end: number };

/**
 * Splits text into spans, marking occurrences of `terms`.
 *
 * Only the supplied terms are highlighted — the built-in lexicon is not
 * consulted here. Callers pass `notes.medical_terms`, which already merges
 * the lexicon sweep (`findMedicalTerms`) with whatever the LLM flagged, so
 * an empty list means "highlight nothing".
 */
export function highlightSpans(text: string, terms: string[] = []): TextSpan[] {
  const wanted = terms
    .map((t) => t.trim())
    // 2 rather than 1: keeps real short terms like "bp" while still guarding
    // against a stray "a" or "ng" highlighting the whole transcript.
    .filter((t) => t.length >= 2)
    // Longest first so an alternation prefers "blood pressure" over "blood".
    .sort((a, b) => b.length - a.length);
  if (!wanted.length) return [{ text, highlighted: false }];

  const re = new RegExp(
    `\\b(?:${wanted.map(escapeRegExp).join('|')})\\b`,
    'gi',
  );

  const ranges: Range[] = [];
  for (const match of text.matchAll(re)) {
    if (match.index === undefined) continue;
    ranges.push({ start: match.index, end: match.index + match[0].length });
  }
  if (!ranges.length) return [{ text, highlighted: false }];

  ranges.sort((a, b) => a.start - b.start);

  // Merge overlaps so nested matches ("blood" inside "blood pressure") do not
  // produce split highlights.
  const merged: Range[] = [];
  for (const range of ranges) {
    const last = merged[merged.length - 1];
    if (last && range.start <= last.end) {
      last.end = Math.max(last.end, range.end);
    } else {
      merged.push({ ...range });
    }
  }

  const spans: TextSpan[] = [];
  let pos = 0;
  for (const { start, end } of merged) {
    if (start > pos) spans.push({ text: text.slice(pos, start), highlighted: false });
    spans.push({ text: text.slice(start, end), highlighted: true });
    pos = end;
  }
  if (pos < text.length) spans.push({ text: text.slice(pos), highlighted: false });
  return spans;
}
