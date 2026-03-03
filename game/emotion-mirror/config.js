const HF_BASE = 'https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main'

export const EMOTIONS = {
  A: { label: 'happy',     bg: '#fef3c7', accent: '#92400e' },
  B: { label: 'focused',   bg: '#dbeafe', accent: '#1e3a8a' },
  C: { label: 'confused',  bg: '#ede9fe', accent: '#5b21b6' },
  D: { label: 'surprised', bg: '#ffedd5', accent: '#9a3412' },
  E: { label: 'tired',     bg: '#f3f4f6', accent: '#374151' },
  F: { label: 'dramatic',  bg: '#fee2e2', accent: '#7f1d1d' },
}

export const PROMPT = [
  'This is a photo of a person\'s face.',
  'Classify their expression with exactly one letter:',
  'A = happy (smiling, raised cheeks),',
  'B = focused (neutral, attentive gaze),',
  'C = confused (furrowed brows, uncertain look),',
  'D = surprised (wide eyes, raised eyebrows),',
  'E = tired (drooping eyelids, relaxed mouth),',
  'F = dramatic (exaggerated, intense expression).',
  'Reply with only the single letter.',
].join(' ')

export const MODEL = {
  id: 'LFM2.5-VL-1.6B-merge-linear-Q4-Q4',
  path: HF_BASE,
  size: '~1.8 GB',
  quantization: { decoder: 'q4', embedImages: 'q4' },
}

export const EXTERNAL_DATA_FILE_COUNTS = {
  'decoder_fp16': 2,
}

export const CAPTURE_INTERVAL_MS = 3000
