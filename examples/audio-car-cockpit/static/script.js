(function() {
  function clamp(value, min, max) {
    return Math.min(max, Math.max(min, value));
  }

  /* ========================================================================
     CAPTION - overlay of the transcribed audio
     ======================================================================== */

  class CaptionOverlay {
    constructor() {
      this.dialog = null;
      this.driverEl = null;
      this.modelEl = null;
      this.toolEl = null;
      this.fadeTimer = null;
      this.createDom();
    }

    createDom() {
      // Create "caption" with dialog element
      this.dialog = document.createElement('dialog');
      this.dialog.id = 'caption-dialog';
      this.dialog.innerHTML = `
        <div class="caption-container">
          <div id="sub-driver" class="caption-line driver-line"></div>
          <div id="sub-model" class="caption-line tts-text"></div>
          <div id="sub-tool" class="caption-line tool-text"></div>
        </div>
      `;
      document.body.appendChild(this.dialog);
      
      this.driverEl = document.getElementById('sub-driver');
      this.modelEl = document.getElementById('sub-model');
      this.toolEl = document.getElementById('sub-tool');
      
      // Force dialog open to allow manual visibility control via CSS opacity
      this.dialog.show(); 
    }

    updateDriver(text) {
      if (!text) return;
      this.driverEl.textContent = `STT: ${text}`;
      this.driverEl.style.display = 'block';
      this.show();
    }

    updateModel(text, toolName, toolValid = true) {
      this.modelEl.textContent = text ? `TTS: ${text}` : '';

      if (toolName) {
        this.toolEl.textContent = `Tool: ${toolName}`;
        this.toolEl.style.display = 'inline';
        this.toolEl.style.color = toolValid ? 'var(--success)' : 'var(--danger)';
      } else {
        this.toolEl.textContent = 'Tool: [None]';
        this.toolEl.style.display = 'inline';
        this.toolEl.style.color = 'var(--warning)';
      }

      this.show();
    }

    show() {
      if (this.fadeTimer) {
        clearTimeout(this.fadeTimer);
        this.fadeTimer = null;
      }
      
      // Reset animations
      this.dialog.classList.remove('fading');
      this.dialog.classList.add('visible');

      // Set timer to fade out after 10 seconds
      this.fadeTimer = setTimeout(() => {
        this.dialog.classList.add('fading');
        this.dialog.classList.remove('visible');
      }, 10000);
    }
  }

  /* ========================================================================
     COCKPIT CONTROLLER - Centralized state management and control
     ======================================================================== */

  class CockpitController {
    constructor() {
      // Windows state
      this.windows = {
        fl: false, // front-left
        fr: false, // front-right
        rl: false, // rear-left
        rr: false, // rear-right
      };

      // Media state
      this.media = {
        tracks: [
          { title: 'Neon Skyline', artist: 'Ocean Drive', album: 'Night Run', lengthSec: 192 },
          { title: 'Midnight Circuit', artist: 'Analog Youth', album: 'City Lights', lengthSec: 214 },
          { title: 'Aurora Lane', artist: 'Glass Horizon', album: 'Northern Routes', lengthSec: 201 },
          { title: 'Static Bloom', artist: 'Soft Repeat', album: 'Daybreak', lengthSec: 189 },
          { title: 'Chrome Echoes', artist: 'Silver Relay', album: 'Late Shift', lengthSec: 205 },
          { title: 'Night Parade', artist: 'Kinetic Haze', album: 'Boulevard', lengthSec: 223 },
          { title: 'Signal Drift', artist: 'Blue State', album: 'Afterglow', lengthSec: 208 },
        ],
        currentIndex: 0,
        isPlaying: false,
        currentTime: 0,
        timerId: null,
      };

      // Climate state
      this.climate = {
        currentTemp: 23,
        targetTemp: 23,
        fanLevel: 2,
        timerId: null,
      };

      // Navigation state
      this.navigation = {
        route: [],
        destination: null,
        currentIndex: -1,
        isRunning: false,
        timerId: null,
        stepStartTime: null,
        stepDurationMs: 0,
      };

      // Audio state
      this.audio = {
        isRecording: false,
        isProcessing: false,
        isPlayingAudio: false,
        mediaRecorder: null,
        audioChunks: [],
        recordedBlob: null,
        audioWs: null,
        selectedVoice: 'US female',
        audioContext: null,
        transcribedText: '',
        currentAudioSource: null,
        audioQueue: [],
        isPlayingQueue: false,
        bufferThresholdMs: 100, // Audio buffer
        totalBufferedMs: 0,
        isBuffering: false,
        nextStartTime: 0, // Schedule time for next audio chunk
        scheduledSources: [], // Track all scheduled sources
      };

      // UI element references (set during initialization)
      this.ui = {};
      
      // captions
      this.captions = new CaptionOverlay();
    }

    /* ====== WINDOWS CONTROL ====== */

    setWindow(id, open) {
      if (!(id in this.windows)) return false;
      this.windows[id] = open;
      this.renderWindows();
      return true;
    }

    toggleWindow(id) {
      if (!(id in this.windows)) return false;
      return this.setWindow(id, !this.windows[id]);
    }

    openAllWindows() {
      Object.keys(this.windows).forEach(id => this.setWindow(id, true));
    }

    closeAllWindows() {
      Object.keys(this.windows).forEach(id => this.setWindow(id, false));
    }

    getWindowState(id) {
      return id ? this.windows[id] : {...this.windows};
    }

    renderWindows() {
      Object.keys(this.windows).forEach(id => {
        const open = this.windows[id];
        const vizFrame = document.getElementById('viz-' + id);
        if (vizFrame) {
          if (open) vizFrame.classList.add('is-open');
          else vizFrame.classList.remove('is-open');
        }

        const button = document.querySelector('button[data-window-toggle="' + id + '"]');
        if (button) {
          button.textContent = open ? 'Close' : 'Open';
          if (!open) button.classList.add('active');
          else button.classList.remove('active');
        }
      });
    }

    /* ====== MEDIA CONTROL ====== */

    mediaPlay() {
      if (this.media.isPlaying) return false;
      this.media.isPlaying = true;
      this.startMediaTimer();
      this.renderMedia();
      return true;
    }

    mediaPause() {
      if (!this.media.isPlaying) return false;
      this.media.isPlaying = false;
      this.stopMediaTimer();
      this.renderMedia();
      return true;
    }

    mediaTogglePlayPause() {
      if (this.media.isPlaying) {
        return this.mediaPause();
      } else {
        return this.mediaPlay();
      }
    }

    mediaNext() {
      this.media.currentIndex = (this.media.currentIndex + 1) % this.media.tracks.length;
      this.media.currentTime = 0;
      this.renderMedia();
      return true;
    }

    mediaPrevious() {
      this.media.currentIndex = (this.media.currentIndex - 1 + this.media.tracks.length) % this.media.tracks.length;
      this.media.currentTime = 0;
      this.renderMedia();
      return true;
    }

    mediaSetTrack(index) {
      if (index < 0 || index >= this.media.tracks.length) return false;
      this.media.currentIndex = index;
      this.media.currentTime = 0;
      this.renderMedia();
      return true;
    }

    getMediaState() {
      const track = this.media.tracks[this.media.currentIndex];
      return {
        isPlaying: this.media.isPlaying,
        currentIndex: this.media.currentIndex,
        currentTime: this.media.currentTime,
        track: {
          title: track.title,
          artist: track.artist,
          album: track.album,
          lengthSec: track.lengthSec,
        },
      };
    }

    startMediaTimer() {
      if (this.media.timerId) return;
      this.media.timerId = setInterval(() => {
        const track = this.media.tracks[this.media.currentIndex];
        this.media.currentTime += 0.25;
        if (this.media.currentTime >= track.lengthSec) {
          this.mediaNext();
        } else {
          this.renderMedia();
        }
      }, 250);
    }

    stopMediaTimer() {
      if (this.media.timerId) {
        clearInterval(this.media.timerId);
        this.media.timerId = null;
      }
    }

    renderMedia() {
      const track = this.media.tracks[this.media.currentIndex];
      const formatTime = (sec) => {
        sec = Math.floor(sec);
        const m = Math.floor(sec / 60);
        const s = sec % 60;
        return m + ':' + (s < 10 ? '0' + s : s);
      };

      this.ui.mediaTitleEl.textContent = track.title;
      this.ui.mediaArtistEl.textContent = track.artist;
      this.ui.mediaAlbumEl.textContent = track.album;
      this.ui.mediaTotalTimeEl.textContent = formatTime(track.lengthSec);
      this.ui.mediaCurrentTimeEl.textContent = formatTime(this.media.currentTime);

      const progress = track.lengthSec ? (this.media.currentTime / track.lengthSec) * 100 : 0;
      this.ui.mediaProgressEl.style.width = clamp(progress, 0, 100) + '%';

      this.ui.mediaCard.classList.toggle('media-playing', this.media.isPlaying);
      this.ui.mediaProgressEl.classList.toggle('paused', !this.media.isPlaying);

      if (this.media.isPlaying) {
        this.ui.mediaPlayIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5h3v14H8zM13 5h3v14h-3z"/></svg>';
        this.ui.mediaStatusChip.classList.add('active');
        this.ui.mediaStatusText.textContent = 'Playing';
      } else {
        this.ui.mediaPlayIcon.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor"><path d="M8 5v14l11-7z"/></svg>';
        this.ui.mediaStatusChip.classList.remove('active');
        this.ui.mediaStatusText.textContent = 'Paused';
      }
    }

    /* ====== CLIMATE CONTROL ====== */

    climateSetTarget(temp) {
      temp = clamp(temp, 16, 30);
      this.climate.targetTemp = temp;
      this.renderClimate();
      return true;
    }

    climateAdjustTarget(delta) {
      return this.climateSetTarget(this.climate.targetTemp + delta);
    }

    climateSetFan(level) {
      level = clamp(level, 1, 4);
      this.climate.fanLevel = level;
      this.renderClimate();
      return true;
    }

    climateAdjustFan(delta) {
      return this.climateSetFan(this.climate.fanLevel + delta);
    }

    climateIncreaseFan() {
      return this.climateAdjustFan(1);
    }

    climateDecreaseFan() {
      return this.climateAdjustFan(-1);
    }

    getClimateState() {
      return {
        currentTemp: this.climate.currentTemp,
        targetTemp: this.climate.targetTemp,
        fanLevel: this.climate.fanLevel,
      };
    }

    renderClimate() {
      this.ui.climateCurrentEl.textContent = this.climate.currentTemp.toFixed(1).replace(/\.0$/, '');
      this.ui.climateTargetEl.textContent = this.climate.targetTemp.toFixed(1).replace(/\.0$/, '');
      this.ui.fanLevelEl.textContent = this.climate.fanLevel;

      this.ui.fanDots.forEach(dot => {
        const level = Number(dot.getAttribute('data-fan-dot'));
        dot.classList.toggle('active', level <= this.climate.fanLevel);
      });

      const diff = this.climate.targetTemp - this.climate.currentTemp;
      const absDiff = Math.abs(diff);
      this.ui.climateStatusPill.classList.remove('heating', 'cooling', 'idle');

      let mode = 'idle';
      if (absDiff < 0.2) {
        mode = 'idle';
        this.ui.climateStatusLabel.textContent = 'Constant';
        this.ui.climateStatusText.textContent = 'Maintaining temperature.';
      } else if (diff > 0) {
        mode = 'heating';
        this.ui.climateStatusLabel.textContent = 'Heating';
        this.ui.climateStatusText.textContent = 'Raising to ' + this.climate.targetTemp.toFixed(1).replace(/\.0$/, '') + '°C.';
      } else {
        mode = 'cooling';
        this.ui.climateStatusLabel.textContent = 'Cooling';
        this.ui.climateStatusText.textContent = 'Lowering to ' + this.climate.targetTemp.toFixed(1).replace(/\.0$/, '') + '°C.';
      }
      this.ui.climateStatusPill.classList.add(mode);

      const maxDelta = 8;
      const progress = clamp((absDiff / maxDelta) * 100, 0, 100);
      this.ui.climateProgress.style.width = progress + '%';

      this.ui.fanRotor.classList.add('active');
      const baseDuration = 2.2;
      const duration = baseDuration / this.climate.fanLevel;
      this.ui.fanRotor.style.animationDuration = duration.toFixed(2) + 's';
    }

    climateTick() {
      const diff = this.climate.targetTemp - this.climate.currentTemp;
      if (Math.abs(diff) < 0.05) {
        this.climate.currentTemp = this.climate.targetTemp;
        this.renderClimate();
        return;
      }
      const responsiveness = 0.06 + (this.climate.fanLevel - 1) * 0.06;
      const maxStep = 0.5 + (this.climate.fanLevel - 1) * 0.3;
      let delta = diff * responsiveness;
      delta = clamp(delta, -maxStep, maxStep);
      if (Math.abs(delta) > Math.abs(diff)) {
        delta = diff;
      }
      this.climate.currentTemp += delta;
      this.renderClimate();
    }

    startClimateLoop() {
      if (this.climate.timerId) return;
      this.climate.timerId = setInterval(() => this.climateTick(), 500);
    }

    /* ====== NAVIGATION CONTROL ====== */

    navSetDestination(destination) {
      if (!destination) {
        destination = this.generateDestination();
      }
      this.navigation.destination = destination;
      this.navigation.route = this.generateRoute();
      this.navigation.currentIndex = -1;
      this.navigation.isRunning = false;

      if (this.navigation.timerId) {
        cancelAnimationFrame(this.navigation.timerId);
        this.navigation.timerId = null;
      }

      this.renderNavigation();
      return true;
    }

    navStart() {
      // Set destination if not already set
      if (!this.navigation.destination) {
        this.navSetDestination();
      }

      if (!this.navigation.route.length) return false;
      if (this.navigation.currentIndex === -1) {
        this.setNavStep(0);
      }
      if (this.navigation.isRunning) return false;

      this.navigation.isRunning = true;
      this.navigation.stepStartTime = performance.now();
      this.renderNavigation();
      this.navigation.timerId = requestAnimationFrame((t) => this.navTick(t));
      return true;
    }

    navPause() {
      if (!this.navigation.isRunning) return false;
      this.navigation.isRunning = false;
      if (this.navigation.timerId) {
        cancelAnimationFrame(this.navigation.timerId);
        this.navigation.timerId = null;
      }
      this.renderNavigation();
      return true;
    }

    navToggle() {
      if (this.navigation.isRunning) {
        return this.navPause();
      } else {
        return this.navStart();
      }
    }

    navClear() {
      this.navigation.route = [];
      this.navigation.destination = null;
      this.navigation.currentIndex = -1;
      this.navigation.isRunning = false;
      if (this.navigation.timerId) {
        cancelAnimationFrame(this.navigation.timerId);
        this.navigation.timerId = null;
      }
      this.renderNavigation();
      return true;
    }

    getNavigationState() {
      return {
        destination: this.navigation.destination,
        isRunning: this.navigation.isRunning,
        currentIndex: this.navigation.currentIndex,
        totalSteps: this.navigation.route.length,
        currentStep: this.navigation.currentIndex >= 0 && this.navigation.currentIndex < this.navigation.route.length
          ? this.navigation.route[this.navigation.currentIndex]
          : null,
      };
    }

    generateDestination() {
      const cityNames = ['Downtown', 'Old Harbor', 'Skyline Point', 'Lakeside Plaza', 'Riverfront', 'Tech Campus', 'North Station', 'Central Park'];
      const poiNames = ['Supercharger', 'Office', 'Home', 'Airport', 'Concert Hall', 'Arena', 'Conference Center'];
      const randomFrom = (arr) => arr[Math.floor(Math.random() * arr.length)];
      const city = randomFrom(cityNames);
      const poi = randomFrom(poiNames);
      const suffix = Math.random() < 0.5 ? city : city + ' ' + (Math.floor(Math.random() * 50) + 1);
      return poi + ' · ' + suffix;
    }

    generateRoute() {
      const streetNames = ['Aurora Ave', 'Maple Street', 'Denver Road', 'Harbor Blvd', 'Circuit Way', 'Summit Lane', 'Crescent Drive', 'Riverside Way'];
      const cityNames = ['Downtown', 'Old Harbor', 'Skyline Point', 'Lakeside Plaza', 'Riverfront', 'Tech Campus', 'North Station', 'Central Park'];
      const randomFrom = (arr) => arr[Math.floor(Math.random() * arr.length)];

      const steps = [];
      const stepCount = 60 + Math.floor(Math.random() * 140);

      steps.push({
        text: 'Head to the nearest main road.',
        sub: 'Follow local guidance to leave your current street.',
        durationSec: 3 + Math.floor(Math.random() * 3)
      });

      for (let i = 1; i < stepCount - 1; i++) {
        const type = Math.random();
        if (type < 0.33) {
          const dir = Math.random() < 0.5 ? 'left' : 'right';
          steps.push({
            text: 'Turn ' + dir + ' onto ' + randomFrom(streetNames) + '.',
            sub: 'Use the ' + (Math.random() < 0.5 ? 'inner' : 'outer') + ' lane.',
            durationSec: 4 + Math.floor(Math.random() * 5)
          });
        } else if (type < 0.66) {
          const km = (0.8 + Math.random() * 4.2).toFixed(1);
          steps.push({
            text: 'Continue for ' + km + ' km.',
            sub: 'Stay on the current road.',
            durationSec: 4 + Math.floor(Math.random() * 16)
          });
        } else {
          const exitNum = 1 + Math.floor(Math.random() * 3);
          steps.push({
            text: 'At the roundabout, take exit ' + exitNum + '.',
            sub: 'Follow signs for ' + randomFrom(cityNames) + '.',
            durationSec: 4 + Math.floor(Math.random() * 4)
          });
        }
      }

      steps.push({
        text: 'You will arrive at your destination.',
        sub: 'Look for parking when prompted.',
        durationSec: 5
      });

      return steps;
    }

    setNavStep(index) {
      if (!this.navigation.route.length) return;
      this.navigation.currentIndex = index;
      this.navigation.stepStartTime = performance.now();

      const step = this.navigation.route[this.navigation.currentIndex];
      this.navigation.stepDurationMs = step.durationSec * 1000;

      this.renderNavigation();
    }

    navTick(timestamp) {
      if (!this.navigation.isRunning || this.navigation.currentIndex < 0 || !this.navigation.route.length) return;

      if (!this.navigation.stepStartTime) {
        this.navigation.stepStartTime = timestamp;
      }

      const elapsed = timestamp - this.navigation.stepStartTime;
      const ratio = clamp(elapsed / this.navigation.stepDurationMs, 0, 1);
      this.ui.navProgressEl.style.width = (ratio * 100) + '%';

      if (elapsed >= this.navigation.stepDurationMs) {
        if (this.navigation.currentIndex < this.navigation.route.length - 1) {
          this.setNavStep(this.navigation.currentIndex + 1);
        } else {
          this.navigation.isRunning = false;
          this.ui.navProgressEl.style.width = '100%';
          this.renderNavigation();
          return;
        }
      }

      this.navigation.timerId = requestAnimationFrame((t) => this.navTick(t));
    }

    renderNavigation() {
      // Destination
      if (this.navigation.destination) {
        this.ui.navDestinationNameEl.textContent = this.navigation.destination;
      } else {
        this.ui.navDestinationNameEl.textContent = 'No destination set';
      }

      // Current step
      if (this.navigation.currentIndex >= 0 && this.navigation.currentIndex < this.navigation.route.length) {
        const step = this.navigation.route[this.navigation.currentIndex];
        this.ui.navCurrentTextEl.textContent = step.text;
        this.ui.navCurrentSubEl.textContent = step.sub;
      } else if (this.navigation.route.length > 0) {
        this.ui.navCurrentTextEl.textContent = 'Route loaded. Press Start to begin.';
        this.ui.navCurrentSubEl.textContent = '\u00a0';
      } else {
        this.ui.navCurrentTextEl.textContent = 'Waiting for route…';
        this.ui.navCurrentSubEl.textContent = '\u00a0';
      }

      // Button text
      if (this.navigation.isRunning) {
        this.ui.navToggleBtn.textContent = 'Pause';
      } else {
        this.ui.navToggleBtn.textContent = 'Start';
      }

      // Status badge and text
      if (!this.navigation.route.length) {
        this.ui.navStatusBadge.textContent = 'Idle';
        this.ui.navStatusText.textContent = 'Route not active.';
      } else if (!this.navigation.isRunning) {
        this.ui.navStatusBadge.textContent = 'Ready';
        this.ui.navStatusText.textContent = 'Route loaded. Press Start to begin guidance.';
      } else if (this.navigation.currentIndex >= 0 && this.navigation.currentIndex < this.navigation.route.length - 1) {
        this.ui.navStatusBadge.textContent = 'Guiding';
        this.ui.navStatusText.textContent = 'Following step ' + (this.navigation.currentIndex + 1) + ' of ' + this.navigation.route.length + '.';
      } else if (this.navigation.currentIndex === this.navigation.route.length - 1) {
        this.ui.navStatusBadge.textContent = 'Arriving';
        this.ui.navStatusText.textContent = 'Approaching destination.';
      }

      // Next steps list
      this.ui.navListEl.innerHTML = '';
      const startIndex = this.navigation.currentIndex < 0 ? 0 : this.navigation.currentIndex + 1;
      const visibleSteps = this.navigation.route.slice(startIndex, startIndex + 4);
      visibleSteps.forEach((step, offset) => {
        const index = startIndex + offset;
        const li = document.createElement('li');
        li.className = 'nav-step-item upcoming';
        li.innerHTML = '<span class="index">' + (index + 1) + '.</span><span>' + step.text + '</span>';
        this.ui.navListEl.appendChild(li);
      });
    }

    /* ====== AUDIO ASSISTANT ====== */

    async audioStartRecording() {
      // If already recording, ignore
      if (this.audio.isRecording) return false;

      // If processing or playing, terminate early
      if (this.audio.isProcessing || this.audio.isPlayingAudio) {
        console.log('Terminating previous audio session...');
        this.audioTerminate();
      }

      try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        this.audio.mediaRecorder = new MediaRecorder(stream);
        this.audio.audioChunks = [];

        this.audio.mediaRecorder.ondataavailable = (e) => this.audio.audioChunks.push(e.data);
        this.audio.mediaRecorder.onstop = () => {
          this.audio.recordedBlob = new Blob(this.audio.audioChunks, { type: 'audio/webm' });
          stream.getTracks().forEach(track => track.stop());
          this.audioSendRequest();
        };

        this.audio.mediaRecorder.start();
        this.audio.isRecording = true;
        this.renderAudio();
        return true;
      } catch (err) {
        console.error('Microphone access denied:', err);
        return false;
      }
    }

    audioStopRecording() {
      if (!this.audio.isRecording || !this.audio.mediaRecorder) return false;

      if (this.audio.mediaRecorder.state === 'recording') {
        this.audio.mediaRecorder.stop();
        this.audio.isRecording = false;
        this.renderAudio();
      }
      return true;
    }

    async audioSendRequest() {
      if (!this.audio.recordedBlob) return;

      this.audio.isProcessing = true;
      this.audio.transcribedText = '';
      this.audio.audioQueue = [];
      this.audio.totalBufferedMs = 0;
      this.audio.isBuffering = false;
      this.audio.isPlayingQueue = false;
      this.audio.nextStartTime = 0;
      this.audio.scheduledSources = [];
      this.renderAudio();

      try {
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        // Connect to main server's audio endpoint
        const url = `${protocol}//${window.location.host}/ws-audio`;

        this.audio.audioWs = new WebSocket(url);

        this.audio.audioWs.onopen = async () => {
          // Convert to WAV and send
          const wavBlob = await this.convertToWav(this.audio.recordedBlob);
          const reader = new FileReader();
          reader.onload = () => {
            const audioB64 = reader.result.split(',')[1];
            const payload = {
              mode: 'asr',
              audio: audioB64,
              voice: this.audio.selectedVoice
            };
            this.audio.audioWs.send(JSON.stringify(payload));
          };
          reader.readAsDataURL(wavBlob);
        };

        this.audio.audioWs.onmessage = (e) => {
          const msg = JSON.parse(e.data);

          if (msg.type === 'text') {
            // Accumulate text chunks from STT (for logging)
            this.audio.transcribedText += msg.data;
          } else if (msg.type === 'caption') {
            // Handle captions from server
            if (msg.role === 'driver') {
              this.captions.updateDriver(msg.text);
            } else if (msg.role === 'model') {
              this.captions.updateModel(msg.text, msg.tool, msg.tool_valid);
            }
          } else if (msg.type === 'audio') {
            // Queue audio chunk for streaming playback
            this.queueAudioChunk(msg.data, msg.sample_rate);
          } else if (msg.type === 'done') {
            // Server has completed the full pipeline (ASR → Tool Calling → TTS)
            if (this.audio.transcribedText) {
              console.log('STT:', this.audio.transcribedText);
            }
            this.audio.isProcessing = false;
            this.renderAudio();
            if (this.audio.audioWs) {
              this.audio.audioWs.close();
              this.audio.audioWs = null;
            }
          } else if (msg.type === 'error') {
            console.error('Audio error:', msg.data);
            this.audio.isProcessing = false;
            this.renderAudio();
          }
        };

        this.audio.audioWs.onerror = (err) => {
          console.error('WebSocket error:', err);
          this.audio.isProcessing = false;
          this.renderAudio();
        };

        this.audio.audioWs.onclose = () => {
          if (this.audio.isProcessing) {
            this.audio.isProcessing = false;
            this.renderAudio();
          }
        };
      } catch (err) {
        console.error('Error sending audio:', err);
        this.audio.isProcessing = false;
        this.renderAudio();
      }
    }

    audioSendToTTS(text) {
      if (!this.audio.audioWs || this.audio.audioWs.readyState !== WebSocket.OPEN) {
        console.error('WebSocket not open for TTS');
        return;
      }

      const payload = {
        mode: 'tts',
        text: text,
        voice: this.audio.selectedVoice
      };

      this.audio.audioWs.send(JSON.stringify(payload));
    }

    async convertToWav(blob) {
      const audioContext = new AudioContext({ sampleRate: 16000 });
      const arrayBuffer = await blob.arrayBuffer();
      const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);

      const samples = audioBuffer.getChannelData(0);
      const wavBuffer = this.encodeWav(samples, 16000);
      return new Blob([wavBuffer], { type: 'audio/wav' });
    }

    encodeWav(samples, sampleRate) {
      const buffer = new ArrayBuffer(44 + samples.length * 2);
      const view = new DataView(buffer);

      const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
          view.setUint8(offset + i, string.charCodeAt(i));
        }
      };

      // WAV header
      writeString(0, 'RIFF');
      view.setUint32(4, 36 + samples.length * 2, true);
      writeString(8, 'WAVE');
      writeString(12, 'fmt ');
      view.setUint32(16, 16, true);
      view.setUint16(20, 1, true);
      view.setUint16(22, 1, true);
      view.setUint32(24, sampleRate, true);
      view.setUint32(28, sampleRate * 2, true);
      view.setUint16(32, 2, true);
      view.setUint16(34, 16, true);
      writeString(36, 'data');
      view.setUint32(40, samples.length * 2, true);

      // Convert float to 16-bit PCM
      let offset = 44;
      for (let i = 0; i < samples.length; i++) {
        const s = Math.max(-1, Math.min(1, samples[i]));
        view.setInt16(offset, s < 0 ? s * 0x8000 : s * 0x7FFF, true);
        offset += 2;
      }

      return buffer;
    }

    queueAudioChunk(base64Data, sampleRate) {
      // Decode PCM data
      const pcmBytes = Uint8Array.from(atob(base64Data), c => c.charCodeAt(0));
      const floatArray = new Float32Array(pcmBytes.buffer);

      // Calculate chunk duration in seconds
      const durationSec = floatArray.length / sampleRate;

      // Add to queue
      this.audio.audioQueue.push({ samples: floatArray, sampleRate, durationSec });
      this.audio.totalBufferedMs += durationSec * 1000;

      // If not playing and not buffering yet, start buffering
      if (!this.audio.isPlayingQueue && !this.audio.isBuffering) {
        this.audio.isBuffering = true;
      }

      // Start playing once we have enough buffered audio
      if (this.audio.isBuffering && this.audio.totalBufferedMs >= this.audio.bufferThresholdMs) {
        this.audio.isBuffering = false;
        this.audio.isPlayingQueue = true;
        this.scheduleAudioChunks();
      } else if (this.audio.isPlayingQueue) {
        // Already playing, schedule new chunks immediately
        this.scheduleAudioChunks();
      }
    }

    scheduleAudioChunks() {
      // Schedule all queued chunks
      while (this.audio.audioQueue.length > 0) {
        const chunk = this.audio.audioQueue.shift();
        this.audio.totalBufferedMs -= chunk.durationSec * 1000;

        // Create or reuse audio context
        if (!this.audio.audioContext || this.audio.audioContext.sampleRate !== chunk.sampleRate) {
          this.audio.audioContext = new AudioContext({ sampleRate: chunk.sampleRate });
          this.audio.nextStartTime = 0;
        }

        // Initialize or update next start time
        const now = this.audio.audioContext.currentTime;
        if (this.audio.nextStartTime === 0 || this.audio.nextStartTime < now) {
          // First chunk or fell behind - start immediately with small buffer
          this.audio.nextStartTime = now + 0.01; // 10ms buffer
        }

        // Create audio buffer
        const audioBuffer = this.audio.audioContext.createBuffer(1, chunk.samples.length, chunk.sampleRate);
        audioBuffer.getChannelData(0).set(chunk.samples);

        // Create and schedule source
        const source = this.audio.audioContext.createBufferSource();
        source.buffer = audioBuffer;
        source.connect(this.audio.audioContext.destination);

        // Schedule to start at exact time
        source.start(this.audio.nextStartTime);

        // Track scheduled source
        this.audio.scheduledSources.push(source);

        // Update next start time (seamless continuation)
        this.audio.nextStartTime += chunk.durationSec;

        // Set up cleanup for last chunk
        const isLastChunk = this.audio.audioQueue.length === 0;
        if (isLastChunk) {
          source.onended = () => {
            // Clean up finished sources
            this.audio.scheduledSources = this.audio.scheduledSources.filter(s => s !== source);

            // Check if we're done or should wait for more
            if (this.audio.audioQueue.length === 0 && !this.audio.isProcessing) {
              this.audio.isPlayingQueue = false;
              this.audio.isPlayingAudio = false;
              this.audio.nextStartTime = 0;
              this.audio.totalBufferedMs = 0;
              this.renderAudio();
            }
          };
        }
      }

      // Update UI
      if (!this.audio.isPlayingAudio && this.audio.scheduledSources.length > 0) {
        this.audio.isPlayingAudio = true;
        this.renderAudio();
      }
    }

    playAudio(base64Data, sampleRate) {
      // Legacy method for compatibility - now uses queue
      this.queueAudioChunk(base64Data, sampleRate);
    }

    audioTerminate() {
      // Stop all scheduled audio sources
      for (const source of this.audio.scheduledSources) {
        try {
          source.stop();
        } catch (e) {
          // Already stopped
        }
      }
      this.audio.scheduledSources = [];
      this.audio.currentAudioSource = null;

      // Close WebSocket connection
      if (this.audio.audioWs) {
        try {
          this.audio.audioWs.close();
        } catch (e) {
          // Already closed
        }
        this.audio.audioWs = null;
      }

      // Clear audio queue and buffering state
      this.audio.audioQueue = [];
      this.audio.isPlayingQueue = false;
      this.audio.totalBufferedMs = 0;
      this.audio.isBuffering = false;
      this.audio.nextStartTime = 0;

      // Reset state
      this.audio.isProcessing = false;
      this.audio.isPlayingAudio = false;
      this.audio.transcribedText = '';
      this.renderAudio();
    }

    audioSetVoice(voice) {
      this.audio.selectedVoice = voice;
      if (this.ui.voiceSelect) {
        this.ui.voiceSelect.value = voice;
      }
      return true;
    }

    getAudioState() {
      return {
        selectedVoice: this.audio.selectedVoice,
        isRecording: this.audio.isRecording,
        isProcessing: this.audio.isProcessing,
        isPlayingAudio: this.audio.isPlayingAudio,
      };
    }

    renderAudio() {
      const btn = this.ui.pushToTalkBtn;
      const btnText = this.ui.voiceBtnText;
      const indicator = this.ui.audioIndicator;
      const statusText = this.ui.audioStatusText;

      // Update button state
      btn.classList.remove('recording', 'processing');
      if (this.audio.isRecording) {
        btn.classList.add('recording');
        btnText.textContent = 'Recording...';
      } else if (this.audio.isProcessing) {
        btn.classList.add('processing');
        btnText.textContent = 'Processing...';
      } else {
        btnText.textContent = 'Hold to Talk';
      }

      // Update indicator state - only show when audio is playing
      indicator.classList.remove('active');
      if (this.audio.isPlayingAudio) {
        indicator.classList.add('active');
        statusText.textContent = 'Speaking';
      } else if (this.audio.isProcessing) {
        statusText.textContent = 'Processing';
      } else {
        statusText.textContent = 'Ready';
      }
    }

    /* ====== FULL STATE QUERY ====== */

    getFullState() {
      return {
        windows: this.getWindowState(),
        media: this.getMediaState(),
        climate: this.getClimateState(),
        navigation: this.getNavigationState(),
        audio: this.getAudioState(),
      };
    }
  }

  /* ========================================================================
     UI INITIALIZATION AND EVENT BINDING
     ======================================================================== */

  const controller = new CockpitController();

  function bindUIControls() {
    // Cache UI elements
    controller.ui = {
      // Media
      mediaCard: document.getElementById('media-card'),
      mediaTitleEl: document.getElementById('media-title'),
      mediaArtistEl: document.getElementById('media-artist'),
      mediaAlbumEl: document.getElementById('media-album'),
      mediaCurrentTimeEl: document.getElementById('media-current-time'),
      mediaTotalTimeEl: document.getElementById('media-total-time'),
      mediaProgressEl: document.getElementById('media-progress'),
      mediaPlayIcon: document.getElementById('media-play-icon'),
      mediaStatusChip: document.getElementById('media-status-chip'),
      mediaStatusText: document.getElementById('media-status-text'),

      // Audio
      pushToTalkBtn: document.getElementById('push-to-talk'),
      voiceBtnText: document.getElementById('voice-btn-text'),
      voiceSelect: document.getElementById('voice-select'),
      audioIndicator: document.getElementById('audio-indicator'),
      audioStatusText: document.getElementById('audio-status-text'),

      // Climate
      climateCurrentEl: document.getElementById('climate-current'),
      climateTargetEl: document.getElementById('climate-target'),
      climateStatusPill: document.getElementById('climate-status-pill'),
      climateStatusLabel: document.getElementById('climate-status-label'),
      climateStatusText: document.getElementById('climate-status-text'),
      climateProgress: document.getElementById('climate-progress'),
      fanRotor: document.getElementById('fan-rotor'),
      fanLevelEl: document.getElementById('fan-level'),
      fanDots: document.querySelectorAll('[data-fan-dot]'),

      // Navigation
      navDestinationNameEl: document.getElementById('nav-destination-name'),
      navToggleBtn: document.getElementById('nav-toggle'),
      navCurrentTextEl: document.getElementById('nav-current-text'),
      navCurrentSubEl: document.getElementById('nav-current-sub'),
      navProgressEl: document.getElementById('nav-progress'),
      navListEl: document.getElementById('nav-list'),
      navStatusBadge: document.getElementById('nav-status-badge'),
      navStatusText: document.getElementById('nav-status-text'),
    };

    // Windows
    controller.closeAllWindows();

    document.querySelectorAll('[data-window-toggle]').forEach(btn => {
      btn.addEventListener('click', () => {
        const id = btn.getAttribute('data-window-toggle');
        controller.toggleWindow(id);
      });
    });

    document.getElementById('windows-open-all').addEventListener('click', () => {
      controller.openAllWindows();
    });

    document.getElementById('windows-close-all').addEventListener('click', () => {
      controller.closeAllWindows();
    });

    // Media
    document.getElementById('media-prev').addEventListener('click', () => {
      controller.mediaPrevious();
    });

    document.getElementById('media-next').addEventListener('click', () => {
      controller.mediaNext();
    });

    document.getElementById('media-play-pause').addEventListener('click', () => {
      controller.mediaTogglePlayPause();
    });

    controller.renderMedia();

    // Audio Assistant
    const pushToTalkBtn = document.getElementById('push-to-talk');

    // Mouse events
    pushToTalkBtn.addEventListener('mousedown', () => {
      controller.audioStartRecording();
    });
    pushToTalkBtn.addEventListener('mouseup', () => {
      controller.audioStopRecording();
    });
    pushToTalkBtn.addEventListener('mouseleave', () => {
      if (controller.audio.isRecording) {
        controller.audioStopRecording();
      }
    });

    // Voice selection
    document.getElementById('voice-select').addEventListener('change', (e) => {
      controller.audioSetVoice(e.target.value);
    });

    controller.renderAudio();

    // Climate
    document.querySelectorAll('.temp-step-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const dir = btn.getAttribute('data-temp');
        controller.climateAdjustTarget(dir === 'up' ? 0.5 : -0.5);
      });
    });

    document.querySelectorAll('[data-fan]').forEach(btn => {
      btn.addEventListener('click', () => {
        const dir = btn.getAttribute('data-fan');
        controller.climateAdjustFan(dir === 'up' ? 1 : -1);
      });
    });

    controller.renderClimate();
    controller.startClimateLoop();

    // Navigation
    document.getElementById('nav-set-destination').addEventListener('click', () => {
      controller.navSetDestination();
    });

    document.getElementById('nav-toggle').addEventListener('click', () => {
      controller.navToggle();
    });

    document.getElementById('nav-clear').addEventListener('click', () => {
      controller.navClear();
    });

    controller.renderNavigation();
  }

  /* ========================================================================
     WEBSOCKET CLIENT - JSON-RPC over WebSocket
     ======================================================================== */

  class WebSocketClient {
    constructor() {
      this.ws = null;
      this.reconnectTimer = null;
      this.messageHandler = null;
    }

    connect(onMessage) {
      this.messageHandler = onMessage;
      const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const url = `${protocol}//${window.location.host}/ws`;

      this.ws = new WebSocket(url);
      this.ws.onopen = () => this.onOpen();
      this.ws.onmessage = (evt) => this.onMessage(evt);
      this.ws.onerror = (err) => this.onError(err);
      this.ws.onclose = () => this.onClose();
    }

    onOpen() {
      console.log('WebSocket connected');
      if (this.reconnectTimer) {
        clearTimeout(this.reconnectTimer);
        this.reconnectTimer = null;
      }
    }

    onMessage(event) {
      try {
        const message = JSON.parse(event.data);
        if (this.messageHandler) {
          this.messageHandler(message);
        }
      } catch (e) {
        console.error('Failed to parse WebSocket message:', e);
      }
    }

    onError(error) {
      console.error('WebSocket error:', error);
    }

    onClose() {
      console.log('WebSocket disconnected');
      this.ws = null;
      this.reconnectTimer = setTimeout(() => this.connect(this.messageHandler), 2000);
    }

    send(message) {
      if (this.ws && this.ws.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify(message));
      } else {
        console.warn('WebSocket not connected, cannot send message');
      }
    }

    sendRpcResponse(id, result) {
      this.send({ jsonrpc: '2.0', id, result });
    }
  }

  const wsClient = new WebSocketClient();

  /* ========================================================================
     FUNCTION DEFINITIONS - Programmatically accessible API specification
     ======================================================================== */

  // Schema builders for DRY parameter definitions
  const noParams = () => ({
      properties: {}
  });

  const windowIdParam = (desc, required = true) => ({
    properties: {
      id: {
        type: 'string',
        enum: ['fl', 'fr', 'rl', 'rr'],
        description: desc
      }
    },
    ...(required && { required: ['id'] })
  });

  const numParam = (name, desc, required = true) => ({
    properties: { [name]: { type: 'number', description: desc } },
    ...(required && { required: [name] })
  });

  const strParam = (name, desc, required = false) => ({
    properties: { [name]: { type: 'string', description: desc } },
    ...(required && { required: [name] })
  });

  // Compact function definition helper
  const fn = (name, description, parameters, handler) =>
    ({ name, description, parameters, handler });

  // API function registry
  const FUNCTION_DEFINITIONS = [
    // Windows
    fn('carWindows.set', 'Set the state of a specific window (open or closed).', {
      properties: {
        id: { type: 'string', enum: ['fl', 'fr', 'rl', 'rr'],
              description: 'Window identifier: "fl" (front-left), "fr" (front-right), "rl" (rear-left), "rr" (rear-right).' },
        open: { type: 'boolean', description: 'Set window to open (true) or closed (false).' }
      },
      required: ['id', 'open']
    }, p => controller.setWindow(p.id, p.open)),

    // fn('carWindows.toggle', 'Toggle a specific window between open and closed states.',
    //    windowIdParam('Window identifier.'), p => controller.toggleWindow(p.id)),

    fn('carWindows.openAll', 'Open all four windows simultaneously.',
       noParams(), () => controller.openAllWindows()),

    fn('carWindows.closeAll', 'Close all four windows simultaneously.',
       noParams(), () => controller.closeAllWindows()),

    // fn('carWindows.get', 'Get the current state of one or all windows.',
    //    windowIdParam('Optional: specific window ID. Omit to get all windows.', false),
    //    p => controller.getWindowState(p.id)),

    // Media
    fn('media.play', 'Start media playback.', noParams(), () => controller.mediaPlay()),
    fn('media.pause', 'Pause media playback.', noParams(), () => controller.mediaPause()),
    fn('media.togglePlayPause', 'Toggle between play and pause states.', noParams(), () => controller.mediaTogglePlayPause()),
    fn('media.next', 'Skip to the next track.', noParams(), () => controller.mediaNext()),
    fn('media.previous', 'Go back to the previous track.', noParams(), () => controller.mediaPrevious()),

    fn('media.setTrack', 'Jump to a specific track by index.',
       numParam('index', 'Track index (0-6 for the available tracks).'),
       p => controller.mediaSetTrack(p.index)),

    // fn('media.get', 'Get the current media player state including track info and playback status.',
    //    noParams(), () => controller.getMediaState()),

    // Climate
    fn('climate.setTarget', 'Set the target temperature for climate control.',
       numParam('temperature', 'Target temperature in Celsius (16-30).'),
       p => controller.climateSetTarget(p.temperature)),

    fn('climate.setFan', 'Set the fan speed level.',
       numParam('level', 'Fan level (1-4, where 1 is low and 4 is high).'),
       p => controller.climateSetFan(p.level)),

    fn('climate.increaseFan', 'Increase the fan speed by 1 level.',
       noParams(), () => controller.climateIncreaseFan()),

    fn('climate.decreaseFan', 'Decrease the fan speed by 1 level.',
       noParams(), () => controller.climateDecreaseFan()),

    // fn('climate.get', 'Get the current climate control state including temperatures and fan level.',
    //    noParams(), () => controller.getClimateState()),

    // Navigation
    fn('navigation.setDestination', 'Set a navigation destination and generate a route.',
       strParam('destination', 'Optional: destination name. If omitted, a random destination is generated.'),
       p => controller.navSetDestination(p.destination)),

    fn('navigation.start', 'Start turn-by-turn navigation guidance.', noParams(), () => controller.navStart()),
    fn('navigation.pause', 'Pause navigation guidance.', noParams(), () => controller.navPause()),
    fn('navigation.toggle', 'Toggle navigation guidance between running and paused states.', noParams(), () => controller.navToggle()),
    fn('navigation.clear', 'Clear the current route and destination.', noParams(), () => controller.navClear()),

    // fn('navigation.get', 'Get the current navigation state including destination, route, and progress.',
    //    noParams(), () => controller.getNavigationState()),

    // Audio
    // fn('audio.setVoice', 'Set the voice for text-to-speech output.', {
    //   properties: {
    //     voice: {
    //       type: 'string',
    //       enum: ['US male', 'US female', 'UK male', 'UK female'],
    //       description: 'Voice selection for TTS: "US male", "US female", "UK male", or "UK female".'
    //     }
    //   },
    //   required: ['voice']
    // }, p => controller.audioSetVoice(p.voice)),
    //
    // fn('audio.get', 'Get the current audio assistant state including selected voice.',
    //    noParams(), () => controller.getAudioState()),

    // System
    // fn('system.getState', 'Get the complete state of all cockpit subsystems (windows, media, climate, navigation, audio).',
    //    noParams(), () => controller.getFullState())
  ];

  // Build lookup map for fast access
  const FUNCTION_MAP = FUNCTION_DEFINITIONS.reduce((map, def) => {
    map[def.name] = def;
    return map;
  }, {});

  /* ========================================================================
     MESSAGE HANDLING - JSON-RPC command execution
     ======================================================================== */

  function handleRpcMessage(message) {
    // Handle incoming JSON-RPC messages
    if (message.method) {
      // This is a request or notification
      const result = executeCommand(message.method, message.params || {});

      // If there's an id, send a response
      if (message.id !== undefined) {
        wsClient.sendRpcResponse(message.id, result);
      }
    } else if (message.result !== undefined || message.error !== undefined) {
      // This is a response to our request
      console.log('Received RPC response:', message);
    }
  }

  function executeCommand(method, params) {
    // Special case: return function definitions
    if (method === 'system.getFunctions') {
      return FUNCTION_DEFINITIONS.map(def => ({
        name: def.name,
        description: def.description,
        parameters: def.parameters
      }));
    }

    // Look up and execute the handler
    const def = FUNCTION_MAP[method];
    if (def) {
      return def.handler(params);
    } else {
      throw new Error('Unknown method: ' + method);
    }
  }

  /* ========================================================================
     INITIALIZATION
     ======================================================================== */

  function init() {
    bindUIControls();
    wsClient.connect(handleRpcMessage);

    // Expose controller and API for debugging
    window.cockpitController = controller;
    window.cockpitWs = wsClient;
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
