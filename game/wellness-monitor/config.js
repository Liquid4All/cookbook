const HF_BASE = 'https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-ONNX/resolve/main'

// Required by vl-model.js
export const EXTERNAL_DATA_FILE_COUNTS = {
  'decoder_fp16': 2,
}

export const MODEL = {
  id: 'LFM2.5-VL-1.6B-merge-linear-Q4-Q4',
  path: HF_BASE,
  size: '~1.8 GB',
  quantization: { decoder: 'q4', embedImages: 'q4' },
}

export const STATES = {
  A: { label: 'DRINKING WATER',  bg: '#dbeafe' },
  B: { label: 'DRINKING COFFEE', bg: '#fef3c7' },
  C: { label: 'WORKING',         bg: '#f0fdf4' },
  D: { label: 'TIRED',           bg: '#fff7ed' },
  E: { label: 'STRESSED',        bg: '#fef2f2' },
  F: { label: 'BREAK',           bg: '#f9fafb' },
}

export const PROMPT = [
  'Classify the scene with one letter. If no person is clearly visible in the frame, you MUST reply F.',
  'A = person drinking water (bottle or glass at or near mouth),',
  'B = person drinking coffee (cup or mug at or near mouth),',
  'C = person working or focused (face visible, looking at screen, typing, or reading),',
  'D = person tired or drowsy (face visible, drooping eyes, yawning, or head drooping),',
  'E = person stressed or tense (face visible, leaning forward intensely, furrowed brows, or jaw tight),',
  'F = no person visible, or person not at desk, or unclear.',
  'Reply with only the letter.',
].join(' ')

export const CAPTURE_INTERVAL_MS = 2000

export const NUDGE_RULES = [
  {
    id: 'no-water',
    message: 'Time to drink some water! You haven\'t had any in over 45 minutes.',
    cooldownMs: 10 * 60 * 1000,
  },
  {
    id: 'tired',
    message: 'You look tired. Consider taking a short break or stretching.',
    cooldownMs: 10 * 60 * 1000,
  },
  {
    id: 'stressed',
    message: 'You look stressed. Take a breath, relax your shoulders.',
    cooldownMs: 10 * 60 * 1000,
  },
  {
    id: 'focus-no-drink',
    message: 'You\'ve been focused for 45+ minutes without a drink. Hydrate!',
    cooldownMs: 15 * 60 * 1000,
  },
]

// Drink detection debounce — min gap between logged drinks of the same type
export const DRINK_DEBOUNCE_MS = 30 * 1000

// How long without water before the no-water nudge fires
export const NO_WATER_THRESHOLD_MS = 45 * 60 * 1000

// Focus streak threshold before the focus-no-drink nudge fires
export const FOCUS_STREAK_THRESHOLD_MS = 45 * 60 * 1000

// Number of recent states to keep for consecutive checks
export const RECENT_STATES_MAX = 8
