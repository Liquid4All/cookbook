/**
 * FLUID SHADER LOGO with Canvas Masking
 */
(function() {
  const canvas = document.getElementById('liquidCanvas');
  if (!canvas) return;

  const gl = canvas.getContext('webgl', { alpha: true, premultipliedAlpha: false });
  if (!gl) {
    console.warn('WebGL not supported, logo shader disabled');
    return;
  }

  // Vertex Shader: pass-through geometry
  const vertexShaderSource = `
    attribute vec2 a_position;
    void main() {
      gl_Position = vec4(a_position, 0.0, 1.0);
    }
  `;

  // Fragment shader with text masking built-in
  // Inspired by: https://mini.gmshaders.com/p/turbulence
  const fragmentShaderSource = `
    precision highp float;
    uniform float u_time;
    uniform vec2 u_resolution;
  
    // Turbulent flow configuration
    #define TURB_NUM 7.0 
    #define TURB_AMP 1.2
    #define TURB_SPEED 0.5
    #define TURB_FREQ 1.2
    #define TURB_EXP 1.3
  
    // Rotational matrix helper
    mat2 rotate2d(float angle) {
      return mat2(cos(angle), -sin(angle), sin(angle), cos(angle));
    }
  
    // The turbulence calculator
    vec2 turbulence(vec2 p) {
      float freq = TURB_FREQ;
      mat2 rot = mat2(0.6, -0.8, 0.8, 0.6);
      
      for(float i = 0.0; i < TURB_NUM; i++) {
          float t = u_time * TURB_SPEED + sin(u_time * 0.2) * 2.0;
  
          float phase = freq * (p * rot).y + t + i * 1.5;
          
          // Displace
          p += TURB_AMP * rot[0] * sin(phase) / freq;
          
          // Rotate and scale for next octave
          rot *= mat2(0.6, -0.8, 0.8, 0.6);
          freq *= TURB_EXP;
      }
      return p;
    }

    // SDF for text rendering (simplified - we'll use actual text positioning)
    float textMask(vec2 uv) {
      // We'll render "Liquid AI" text
      // For now, return 1.0 in text area, 0.0 outside
      // This is a placeholder - actual text will be rendered via canvas compositing
      return 1.0;
    }

    void main() {
      // 1. Setup
      vec2 uv = (gl_FragCoord.xy * 0.8 - u_resolution) / min(u_resolution.x, u_resolution.y);
      
      // 2. To create patterns that never repeat, we drift the "camera" through the noise field.
      vec2 drift = vec2(u_time * 0.05, -u_time * 0.02);
      
      // We also slowly rotate the view to vary the angle of attack
      float tumble = sin(u_time * 0.1) * 0.15;
      
      // Apply changes to our coordinate system 'p'
      vec2 p = rotate2d(tumble) * (uv * 0.7) + drift;
  
      // 3. Apply Turbulence
      vec2 fluid = turbulence(p);
  
      // 4. Coloring
      vec3 deepIndigo = vec3(0.12, 0.02, 0.25); // Darker, richer base
      vec3 richPurple = vec3(0.55, 0.05, 0.85); // Vibrant purple body
      
      // Base mix based on fluid motion
      float bodyMix = sin(fluid.x * 1.5 + fluid.y * 0.5) * 0.5 + 0.5;
      vec3 col = mix(deepIndigo, richPurple, bodyMix);
  
      // 5. Electric Highlights
      // We calculate the "tension" in the surface
      float energy = dot(cos(fluid * 2.0), sin(fluid.yx * 2.2));
      
      vec3 electricOrange = vec3(1.0, 0.5, 0.0);
      vec3 electricCyan = vec3(0.0, 0.9, 1.0);
  
      // Using power of 4.0 makes the highlights thinner, 
      // leaving more screen real estate for the purple base.
      float orangeMask = pow(max(0.0, energy), 4.0) * 0.6;
  
      col += electricOrange * orangeMask;
  
      // 6. Contrast & Tone Mapping
      col = pow(col, vec3(0.95)); // Slight gamma tweak
      col *= 1.2; // Exposure boost
      col = smoothstep(-0.05, 1.2, col); // Soft clamp
  
      gl_FragColor = vec4(col, 1.0);
    }
  `;

  // WebGL setup
  function createShader(gl, type, source) {
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      console.error('Shader compile error:', gl.getShaderInfoLog(shader));
      gl.deleteShader(shader);
      return null;
    }
    return shader;
  }

  const vertexShader = createShader(gl, gl.VERTEX_SHADER, vertexShaderSource);
  const fragmentShader = createShader(gl, gl.FRAGMENT_SHADER, fragmentShaderSource);

  const program = gl.createProgram();
  gl.attachShader(program, vertexShader);
  gl.attachShader(program, fragmentShader);
  gl.linkProgram(program);

  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    console.error('Program link error:', gl.getProgramInfoLog(program));
  }
  gl.useProgram(program);

  // Full screen quad
  const positionBuffer = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, positionBuffer);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([
    -1, -1,  1, -1, -1,  1,
    -1,  1,  1, -1,  1,  1,
  ]), gl.STATIC_DRAW);

  const positionLocation = gl.getAttribLocation(program, "a_position");
  gl.enableVertexAttribArray(positionLocation);
  gl.vertexAttribPointer(positionLocation, 2, gl.FLOAT, false, 0, 0);

  const timeLocation = gl.getUniformLocation(program, "u_time");
  const resolutionLocation = gl.getUniformLocation(program, "u_resolution");

  // Create an offscreen canvas for text masking
  const maskCanvas = document.createElement('canvas');
  const maskCtx = maskCanvas.getContext('2d');

  // Render Loop
  function render(time) {
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;

    // Resize canvases if needed
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
      maskCanvas.width = width;
      maskCanvas.height = height;
      gl.viewport(0, 0, width, height);

      // Generate text mask
      maskCtx.clearRect(0, 0, width, height);
      maskCtx.fillStyle = 'white';
      maskCtx.font = '600 48px system-ui, -apple-system, BlinkMacSystemFont, sans-serif';
      maskCtx.textAlign = 'center';
      maskCtx.textBaseline = 'middle';
      maskCtx.letterSpacing = '-0.02em';
      maskCtx.fillText('Liquid AI', width / 2, height / 2);

      // Apply mask to canvas
      const maskData = maskCanvas.toDataURL();
      canvas.style.webkitMaskImage = `url(${maskData})`;
      canvas.style.maskImage = `url(${maskData})`;
      canvas.style.webkitMaskSize = 'contain';
      canvas.style.maskSize = 'contain';
      canvas.style.webkitMaskRepeat = 'no-repeat';
      canvas.style.maskRepeat = 'no-repeat';
      canvas.style.webkitMaskPosition = 'center';
      canvas.style.maskPosition = 'center';
    }

    // Render shader
    gl.uniform1f(timeLocation, time * 0.001);
    gl.uniform2f(resolutionLocation, width, height);
    gl.drawArrays(gl.TRIANGLES, 0, 6);

    requestAnimationFrame(render);
  }

  requestAnimationFrame(render);
})();
