import { VLModel } from './vl-model.js'
import { EMOTIONS, PROMPT, MODEL } from './config.js'

// ── DOM refs ─────────────────────────────────────────────────────────────────
const overlay       = document.getElementById('overlay')
const btnLoad       = document.getElementById('btnLoad')
const progressEl    = document.getElementById('progress')
const progressFill  = document.getElementById('progressFill')
const videoEl       = document.getElementById('video')
const emotionLabel  = document.getElementById('emotionLabel')
const historyEl     = document.getElementById('history')
const countdownBar  = document.getElementById('countdownBar')

// ── State ────────────────────────────────────────────────────────────────────
let model       = null
let captureTimer = null
const history   = []   // last 6 emotion keys

// ── Utilities ────────────────────────────────────────────────────────────────
function log(msg) {
  progressEl.textContent = msg
}

function resizeToDataURL(videoEl, width = 320, height = 240) {
  const canvas = document.createElement('canvas')
  canvas.width = width
  canvas.height = height
  // draw un-mirrored (CSS mirror is just display, model gets the real frame)
  canvas.getContext('2d').drawImage(videoEl, 0, 0, width, height)
  return canvas.toDataURL('image/jpeg', 0.85)
}

// ── UI updates ───────────────────────────────────────────────────────────────
function applyEmotion(key) {
  const emotion = EMOTIONS[key]
  if (!emotion) return

  // Background
  document.body.style.backgroundColor = emotion.bg

  // Label
  emotionLabel.textContent = emotion.label
  emotionLabel.style.color = emotion.accent
  emotionLabel.classList.remove('pop')
  void emotionLabel.offsetWidth  // force reflow to restart animation
  emotionLabel.classList.add('pop')
  setTimeout(() => emotionLabel.classList.remove('pop'), 150)

  // History trail (keep last 6)
  history.push(key)
  if (history.length > 6) history.shift()

  historyEl.innerHTML = history
    .map((k, i) => {
      const e = EMOTIONS[k]
      return `<span class="history-item" style="color:${e.accent}">${e.label}</span>`
    })
    .join('<span style="color:#aaa;font-size:11px">→</span>')
}

// Flash the bar briefly to show a new result arrived
function pulseBar() {
  countdownBar.style.transition = 'none'
  countdownBar.style.transform  = 'scaleX(1)'
  void countdownBar.offsetWidth
  countdownBar.style.transition = 'transform 0.4s ease-out'
  countdownBar.style.transform  = 'scaleX(0)'
}

// ── Capture & classify ───────────────────────────────────────────────────────
async function capture() {
  const src = resizeToDataURL(videoEl)
  const images = [src]
  const messageImageMap = new Map([[0, images]])

  try {
    const output = await model.generate(
      [{ role: 'user', content: PROMPT }],
      { maxNewTokens: 1, images, messageImageMap },
    )
    const key = output.trim().toUpperCase().match(/[A-F]/)?.[0]
    if (key) applyEmotion(key)
    pulseBar()
  } catch (err) {
    console.error('[emotion-mirror] inference error', err)
  }

  // Run continuously — delay is just inference latency (~0.5s after warmup)
  if (captureTimer !== null) capture()
}

// ── GPU warmup ───────────────────────────────────────────────────────────────
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

// ── Model load ───────────────────────────────────────────────────────────────
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

    // Start webcam
    log('Starting webcam...')
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { width: 640, height: 480, facingMode: 'user' },
    })
    videoEl.srcObject = stream
    await new Promise(resolve => { videoEl.onloadedmetadata = resolve })
    videoEl.play()

    // Hide overlay, start continuous inference loop
    overlay.style.display = 'none'
    emotionLabel.textContent = 'ready'
    captureTimer = true   // flag: loop is running
    capture()

  } catch (err) {
    log(`ERROR: ${err.message}`)
    console.error(err)
    btnLoad.disabled = false
  }
})
