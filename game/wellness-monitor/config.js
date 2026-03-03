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
  A: { label: 'WORKING', bg: '#f0fdf4' },
  B: { label: 'BREAK',   bg: '#f9fafb' },
}

export const PROMPT = [
  'Look at this image. Reply with one letter only.',
  'B = no person visible.',
  'A = person visible.',
].join(' ')

export const CAPTURE_INTERVAL_MS = 2000

export const NUDGE_RULES = []

// Drink detection debounce — min gap between logged drinks of the same type
export const DRINK_DEBOUNCE_MS = 30 * 1000

// How long without water before the no-water nudge fires
export const NO_WATER_THRESHOLD_MS = 45 * 60 * 1000

// Focus streak threshold before the focus-no-drink nudge fires
export const FOCUS_STREAK_THRESHOLD_MS = 45 * 60 * 1000

// Number of recent states to keep for consecutive checks
export const RECENT_STATES_MAX = 8
