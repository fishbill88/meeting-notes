/**
 * Medical term detection for transcript highlighting.
 *
 * A curated lexicon (English + Tagalog + common PH brand names) is matched
 * against the transcript; terms the LLM flags via `medical_terms` are merged
 * in by the caller. Matching is case-insensitive on word boundaries, longest
 * term first so "blood pressure" wins over "blood".
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

const LEXICON = [...CONDITIONS, ...SYMPTOMS, ...MEDICATIONS, ...PROCEDURES, ...VITALS_ANATOMY]
  // Longest first so multi-word terms win over their substrings
  .sort((a, b) => b.length - a.length);

function escapeRegExp(s: string): string {
  return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

// All lexicon terms start/end with word characters, so \b boundaries are
// safe — and unlike lookbehinds, guaranteed to work on Hermes.
const LEXICON_RE = new RegExp(
  `\\b(?:${LEXICON.map(escapeRegExp).join('|')})\\b`,
  'gi',
);

/** Unique medical terms found in the text, in order of first appearance. */
export function findMedicalTerms(text: string): string[] {
  const seen = new Set<string>();
  const out: string[] = [];
  for (const m of text.matchAll(LEXICON_RE)) {
    const key = m[0].toLowerCase();
    if (!seen.has(key)) { seen.add(key); out.push(m[0]); }
  }
  for (const re of DOSAGE_PATTERNS) {
    for (const m of text.matchAll(re)) {
      const key = m[0].toLowerCase();
      if (!seen.has(key)) { seen.add(key); out.push(m[0]); }
    }
  }
  return out;
}

export interface TextSpan {
  text: string;
  highlighted: boolean;
}

/**
 * Splits text into spans, marking matches of the medical lexicon plus any
 * extra terms (e.g., LLM-flagged ones). Use for <Text> nesting or HTML.
 */
export function highlightSpans(text: string, extraTerms: string[] = []): TextSpan[] {
  const extras = extraTerms
    .map(t => t.trim())
    .filter(t => t.length >= 3)
    .sort((a, b) => b.length - a.length);

  const regexes: RegExp[] = [LEXICON_RE, ...DOSAGE_PATTERNS];
  if (extras.length) {
    regexes.push(new RegExp(
      `\\b(?:${extras.map(escapeRegExp).join('|')})\\b`,
      'gi',
    ));
  }

  // Collect and merge overlapping match ranges
  const ranges: Array<[number, number]> = [];
  for (const re of regexes) {
    re.lastIndex = 0;
    for (const m of text.matchAll(re)) {
      ranges.push([m.index!, m.index! + m[0].length]);
    }
  }
  if (!ranges.length) return [{ text, highlighted: false }];

  ranges.sort((a, b) => a[0] - b[0]);
  const merged: Array<[number, number]> = [ranges[0]];
  for (const [s, e] of ranges.slice(1)) {
    const last = merged[merged.length - 1];
    if (s <= last[1]) last[1] = Math.max(last[1], e);
    else merged.push([s, e]);
  }

  const spans: TextSpan[] = [];
  let pos = 0;
  for (const [s, e] of merged) {
    if (s > pos) spans.push({ text: text.slice(pos, s), highlighted: false });
    spans.push({ text: text.slice(s, e), highlighted: true });
    pos = e;
  }
  if (pos < text.length) spans.push({ text: text.slice(pos), highlighted: false });
  return spans;
}
