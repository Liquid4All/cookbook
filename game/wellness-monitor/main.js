import { VLModel } from './vl-model.js'
import {
  STATES,
  PROMPT,
  MODEL,
  CAPTURE_INTERVAL_MS,
  NUDGE_RULES,
  DRINK_DEBOUNCE_MS,
  NO_WATER_THRESHOLD_MS,
  FOCUS_STREAK_THRESHOLD_MS,
  RECENT_STATES_MAX,
} from './config.js'

// ── DOM refs ──────────────────────────────────────────────────────────────────
const overlay       = document.getElementById('overlay')
const btnLoad       = document.getElementById('btnLoad')
const progressEl    = document.getElementById('progress')
const progressFill  = document.getElementById('progressFill')
const videoEl       = document.getElementById('video')
const stateLabel    = document.getElementById('stateLabel')
const waterCountEl  = document.getElementById('waterCount')
const coffeeCountEl = document.getElementById('coffeeCount')
const lastDrinkEl   = document.getElementById('lastDrink')
const focusStreakEl = document.getElementById('focusStreak')
const recentEl      = document.getElementById('recentStates')
const nudgeBanner      = document.getElementById('nudgeBanner')
const inferenceDot     = document.getElementById('inferenceDot')
const transcriptEl     = document.getElementById('transcript')

// ── Constants ─────────────────────────────────────────────────────────────────
const TODAY   = new Date().toISOString().slice(0, 10)  // 'YYYY-MM-DD'
const LS_KEY  = `wellness-${TODAY}`

// ── Persistence ───────────────────────────────────────────────────────────────

function persistData() {
  try {
    localStorage.setItem(LS_KEY, JSON.stringify({ waterLog, coffeeLog, stateHistory }))
  } catch (e) { /* ignore quota errors */ }
}

function loadPersistedData() {
  try {
    const raw = localStorage.getItem(LS_KEY)
    if (!raw) return
    const data = JSON.parse(raw)
    waterLog.push(...(data.waterLog ?? []))
    coffeeLog.push(...(data.coffeeLog ?? []))
    stateHistory.push(...(data.stateHistory ?? []))
    // Seed recentStates from tail of restored history
    recentStates.push(...stateHistory.slice(-RECENT_STATES_MAX).map(e => e.key))
    // Restore focus streak if last recorded state was C
    if (stateHistory.at(-1)?.key === 'C') {
      focusStart = stateHistory.at(-1).ts
    }
  } catch (e) {
    console.warn('[wellness] could not restore persisted data', e)
  }
}

// ── State ─────────────────────────────────────────────────────────────────────
let model         = null
let running       = false

const waterLog     = []   // timestamps of A detections
const coffeeLog    = []   // timestamps of B detections
const recentStates = []   // last RECENT_STATES_MAX state keys
const stateHistory = []   // { key, ts } — every state transition, persisted

let lastStateKey = null   // last key pushed to stateHistory
let focusStart   = null   // timestamp when current C streak began
const lastNudgeTimes = new Map()  // nudgeId -> timestamp

let nudgeDismissTimer = null

loadPersistedData()
requestAnimationFrame(() => renderStats())

// ── Utilities ─────────────────────────────────────────────────────────────────
function log(msg) {
  progressEl.textContent = msg
}

function resizeToDataURL(videoEl, width = 320, height = 240) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  canvas.getContext('2d').drawImage(videoEl, 0, 0, width, height)
  return canvas.toDataURL('image/jpeg', 0.85)
}

function formatMinutes(ms) {
  const mins = Math.floor(ms / 60000)
  if (mins < 1) return 'just now'
  if (mins === 1) return '1 min ago'
  return `${mins} min ago`
}

function formatStreak(ms) {
  const mins = Math.floor(ms / 60000)
  if (mins < 1) return '<1 min'
  return `${mins} min`
}

// ── Drink logging ─────────────────────────────────────────────────────────────
function tryLogDrink(type) {
  const drinkLog = type === 'A' ? waterLog : coffeeLog
  const now  = Date.now()
  const last = drinkLog.at(-1) ?? 0
  if (now - last >= DRINK_DEBOUNCE_MS) {
    drinkLog.push(now)
    persistData()
  }
}

function lastDrinkTimestamp() {
  const lastWater  = waterLog.at(-1)  ?? 0
  const lastCoffee = coffeeLog.at(-1) ?? 0
  return Math.max(lastWater, lastCoffee)
}

// ── Nudge logic ───────────────────────────────────────────────────────────────
function checkNudges() {
  const now = Date.now()

  for (const rule of NUDGE_RULES) {
    const lastFired = lastNudgeTimes.get(rule.id) ?? 0
    if (now - lastFired < rule.cooldownMs) continue

    let triggered = false

    if (rule.id === 'no-water') {
      const lastWater = waterLog.at(-1) ?? 0
      triggered = (now - lastWater) > NO_WATER_THRESHOLD_MS
    }

    if (rule.id === 'focus-no-drink') {
      if (focusStart !== null) {
        const streakMs = now - focusStart
        const lastDrink = lastDrinkTimestamp()
        const noDrinkSinceFocus = lastDrink < focusStart
        triggered = streakMs > FOCUS_STREAK_THRESHOLD_MS && noDrinkSinceFocus
      }
    }

    if (triggered) {
      fireNudge(rule.id, rule.message)
      lastNudgeTimes.set(rule.id, now)
      break  // one nudge at a time
    }
  }
}

function fireNudge(id, message) {
  nudgeBanner.textContent = message
  nudgeBanner.classList.add('visible')

  if (nudgeDismissTimer) clearTimeout(nudgeDismissTimer)
  nudgeDismissTimer = setTimeout(() => {
    nudgeBanner.classList.remove('visible')
    nudgeDismissTimer = null
  }, 8000)
}

// ── UI updates ────────────────────────────────────────────────────────────────
function applyState(key) {
  const state = STATES[key]
  if (!state) return

  document.body.style.backgroundColor = state.bg
  stateLabel.textContent = state.label

  // Focus streak tracking
  if (key === 'A') {
    if (focusStart === null) focusStart = Date.now()
  } else {
    focusStart = null
  }

  // Recent states ring buffer
  recentStates.push(key)
  if (recentStates.length > RECENT_STATES_MAX) recentStates.shift()

  // Record state transitions to history
  if (key !== lastStateKey) {
    lastStateKey = key
    stateHistory.push({ key, ts: Date.now() })
    persistData()
  }

  renderStats()
  checkNudges()
}

function renderStats() {
  const now = Date.now()

  waterCountEl.textContent  = waterLog.length
  coffeeCountEl.textContent = coffeeLog.length

  const lastDrink = lastDrinkTimestamp()
  lastDrinkEl.textContent = lastDrink > 0 ? formatMinutes(now - lastDrink) : '—'

  if (focusStart !== null) {
    focusStreakEl.textContent = formatStreak(now - focusStart)
  } else {
    focusStreakEl.textContent = '—'
  }

  recentEl.innerHTML = recentStates
    .map(k => `<span class="state-badge ${k}">${STATES[k]?.label ?? k}</span>`)
    .join(' ')

  renderTranscript()
}

function renderTranscript() {
  if (!transcriptEl) return
  if (stateHistory.length === 0) {
    transcriptEl.innerHTML = ''
    return
  }

  const fmt = ts => {
    const d = new Date(ts)
    return `${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}:${String(d.getSeconds()).padStart(2, '0')}`
  }

  // Most recent first
  transcriptEl.innerHTML = stateHistory
    .slice()
    .reverse()
    .map(({ key, ts }) => `
      <div class="transcript-row">
        <span class="transcript-time">${fmt(ts)}</span>
        <span class="state-badge ${key}">${STATES[key]?.label ?? key}</span>
      </div>`)
    .join('')
}

// Refresh relative times every 30s without re-running inference
setInterval(renderStats, 30_000)

// ── Inference dot flash ───────────────────────────────────────────────────────
function flashDot() {
  inferenceDot.classList.add('flash')
  setTimeout(() => inferenceDot.classList.remove('flash'), 300)
}

// ── Capture & classify ────────────────────────────────────────────────────────
async function captureLoop() {
  while (running) {
    const start = Date.now()

    try {
      const src = resizeToDataURL(videoEl)
      const images = [src]
      const messageImageMap = new Map([[0, images]])

      const output = await model.generate(
        [{ role: 'user', content: PROMPT }],
        { maxNewTokens: 1, images, messageImageMap },
      )

      const key = output.trim().toUpperCase().match(/[A-B]/)?.[0]
      if (key) applyState(key)
      flashDot()
    } catch (err) {
      console.error('[wellness-monitor] inference error', err)
    }

    // Throttle to CAPTURE_INTERVAL_MS; subtract inference time
    const elapsed = Date.now() - start
    const delay = Math.max(0, CAPTURE_INTERVAL_MS - elapsed)
    await new Promise(resolve => setTimeout(resolve, delay))
  }
}

// ── GPU warmup ────────────────────────────────────────────────────────────────
async function warmup() {
  log('Compiling GPU shaders (first run only)...')
  const canvas = document.createElement('canvas')
  canvas.width = 64; canvas.height = 64
  const dummyImage = canvas.toDataURL('image/jpeg')
  const images = [dummyImage]
  const messageImageMap = new Map([[0, images]])
  await model.generate(
    [{ role: 'user', content: 'Describe.' }],
    { maxNewTokens: 1, images, messageImageMap },
  )
}

// ── Model load ────────────────────────────────────────────────────────────────
btnLoad.addEventListener('click', async () => {
  if (!navigator.gpu) {
    log('ERROR: WebGPU not available. Use Chrome 113+ or Edge 113+.')
    return
  }

  btnLoad.disabled = true
  model = new VLModel()

  try {
    await model.load(MODEL.path, {
      device: 'webgpu',
      quantization: MODEL.quantization,
      progressCallback: ({ status, progress, file }) => {
        if (status === 'loading' && file) {
          log(`Downloading: ${file}`)
          progressFill.style.width = `${Math.round(progress)}%`
        }
      },
    })

    await warmup()

    log('Starting webcam...')
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
    })
    videoEl.srcObject = stream
    await new Promise(resolve => { videoEl.onloadedmetadata = resolve })
    videoEl.play()

    overlay.style.display = 'none'
    stateLabel.textContent = 'ready'
    running = true
    captureLoop()

  } catch (err) {
    log(`ERROR: ${err.message}`)
    console.error(err)
    btnLoad.disabled = false
  }
})

// ── Transcript download ────────────────────────────────────────────────────────
document.getElementById('btnDownload').addEventListener('click', () => {
  if (stateHistory.length === 0) return

  const fmt = ts => {
    const d = new Date(ts)
    const date = d.toLocaleDateString('en-CA')  // YYYY-MM-DD
    const time = d.toLocaleTimeString('en-GB')  // HH:MM:SS
    return `${date} ${time}`
  }

  const rows = ['timestamp,state']
  for (const { key, ts } of stateHistory) {
    rows.push(`${fmt(ts)},${STATES[key]?.label ?? key}`)
  }

  const blob = new Blob([rows.join('\n')], { type: 'text/csv' })
  const url  = URL.createObjectURL(blob)
  const a    = document.createElement('a')
  a.href     = url
  a.download = `wellness-${TODAY}.csv`
  a.click()
  URL.revokeObjectURL(url)
})
