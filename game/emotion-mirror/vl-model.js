/**
 * LFM2-VL Model Runner for ONNX Runtime Web
 *
 * Runs VL model inference using three ONNX models:
 * 1. embed_tokens.onnx - Text token embeddings
 * 2. embed_images.onnx - Image embeddings from patches
 * 3. decoder.onnx - Autoregressive decoder with conv state cache
 */

import * as ort from 'onnxruntime-web';
import { AutoTokenizer, env } from '@huggingface/transformers';
import { processImage, loadImage } from './vl-processor.js';
import { EXTERNAL_DATA_FILE_COUNTS } from './config.js';

let DEBUG = false;
export function setDebug(value) { DEBUG = value; }
const log = (...args) => { if (DEBUG) console.log(...args); };

const CACHE_NAME = 'onnx-models-v1';
const LARGE_FILE_THRESHOLD = 2 * 1024 * 1024 * 1024;

async function fetchWithProgress(url, options = {}, onProgress) {
  const response = await fetch(url, options);
  if (!response.ok) {
    throw new Error(`Fetch failed: ${response.status}`);
  }

  const contentLength = parseInt(response.headers.get('content-length') || '0', 10);
  if (!contentLength || !onProgress) {
    return response;
  }

  const reader = response.body.getReader();
  const chunks = [];
  let received = 0;

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    onProgress(received, contentLength);
  }

  const buffer = new Uint8Array(received);
  let offset = 0;
  for (const chunk of chunks) {
    buffer.set(chunk, offset);
    offset += chunk.length;
  }

  return new Response(new Blob([buffer]), {
    status: response.status,
    headers: new Headers(response.headers),
  });
}

async function fetchWithCache(url, options = {}, onProgress = null) {
  if (!url.startsWith('http://') && !url.startsWith('https://')) {
    return fetch(url, options);
  }

  const fileName = url.split('/').pop();

  try {
    const cache = await caches.open(CACHE_NAME);
    const cached = await cache.match(url);
    if (cached) {
      try {
        const buffer = await cached.clone().arrayBuffer();
        log(`[Cache HIT] ${fileName} (${(buffer.byteLength / 1024 / 1024).toFixed(1)} MB)`);
        return new Response(buffer, {
          status: cached.status,
          statusText: cached.statusText,
          headers: cached.headers,
        });
      } catch (bodyError) {
        log(`[Cache CORRUPT] ${fileName} - deleting and re-fetching`);
        await cache.delete(url);
      }
    }
  } catch (e) {
    log(`[Cache ERROR] ${e.message}`);
  }

  log(`[Network] Fetching ${fileName}...`);
  const response = await fetchWithProgress(url, options, onProgress);

  if (response.ok) {
    tryCacheResponse(url, response.clone());
  }

  return response;
}

async function tryCacheResponse(url, response) {
  try {
    if (navigator.storage?.estimate) {
      const { usage = 0, quota = 0 } = await navigator.storage.estimate();
      const available = quota - usage;
      const responseSize = parseInt(response.headers.get('content-length') || '0', 10);

      const BUFFER = 100 * 1024 * 1024;
      if (responseSize > 0 && available < responseSize + BUFFER) {
        log(`[Cache SKIP] Not enough space`);
        return;
      }
    }

    const cache = await caches.open(CACHE_NAME);
    await cache.put(url, response);
    log(`[Cached] ${url.split('/').pop()}`);
  } catch (e) {
    console.warn(`[Cache WRITE ERROR] ${url.split('/').pop()}:`, e.name, e.message);
  }
}

export async function clearModelCache() {
  return caches.delete(CACHE_NAME);
}

async function loadTokenizerFromPath(modelPath) {
  const isRemote = modelPath.startsWith('http://') || modelPath.startsWith('https://');
  const fetchOptions = isRemote ? { mode: 'cors', credentials: 'omit' } : {};

  const [tokenizerResponse, configResponse] = await Promise.all([
    fetchWithCache(`${modelPath}/tokenizer.json`, fetchOptions),
    fetchWithCache(`${modelPath}/tokenizer_config.json`, fetchOptions),
  ]);

  if (!tokenizerResponse.ok) throw new Error(`Failed to fetch tokenizer.json: ${tokenizerResponse.status}`);
  if (!configResponse.ok) throw new Error(`Failed to fetch tokenizer_config.json: ${configResponse.status}`);

  const tokenizerJSON = await tokenizerResponse.text();
  const configJSON = await configResponse.text();

  const tokenizerData = JSON.parse(tokenizerJSON);
  const specialTokens = {};

  if (tokenizerData.added_tokens) {
    for (const token of tokenizerData.added_tokens) {
      specialTokens[token.content] = token.id;
    }
  }

  const fakeModelId = `tokenizer-${Date.now()}`;
  const fileCache = {
    'tokenizer.json': tokenizerJSON,
    'tokenizer_config.json': configJSON,
  };

  const originalFetch = globalThis.fetch;
  globalThis.fetch = async (input, init) => {
    const url = typeof input === 'string' ? input : input.url;
    if (url.includes(fakeModelId)) {
      for (const [filename, content] of Object.entries(fileCache)) {
        if (url.includes(filename)) {
          return new Response(content, {
            status: 200,
            headers: { 'Content-Type': 'application/json' },
          });
        }
      }
      return new Response('Not found', { status: 404 });
    }
    return originalFetch(input, init);
  };

  const originalAllowLocal = env.allowLocalModels;
  env.allowLocalModels = false;

  try {
    const tokenizer = await AutoTokenizer.from_pretrained(fakeModelId);
    return { tokenizer, specialTokens };
  } finally {
    globalThis.fetch = originalFetch;
    env.allowLocalModels = originalAllowLocal;
  }
}

export class VLModel {
  constructor() {
    this.tokenizer = null;
    this.embedTokensSession = null;
    this.embedImagesSession = null;
    this.decoderSession = null;
    this.config = null;
    this.imageTokenId = null;
    this.eosTokenId = null;
    this.hiddenSize = 1024;
    this.imageCache = new Map();
  }

  clearImageCache() {
    this.imageCache.clear();
  }

  async load(modelPath, options = {}) {
    const { progressCallback, device = 'webgpu', quantization = null } = options;

    const report = (status, progress = 0, file = '') => {
      if (progressCallback) progressCallback({ status, progress, file });
    };

    const executionProviders = device === 'webgpu' ? ['webgpu', 'wasm'] : ['wasm'];

    report('loading', 0, 'tokenizer');
    const { tokenizer, specialTokens } = await loadTokenizerFromPath(modelPath);
    this.tokenizer = tokenizer;

    if (!this.tokenizer.chat_template) {
      try {
        const templateResponse = await fetch(`${modelPath}/chat_template.jinja`, {
          mode: 'cors', credentials: 'omit',
        });
        if (templateResponse.ok) {
          this.tokenizer.chat_template = await templateResponse.text();
        }
      } catch (e) {
        console.warn('Could not load chat template:', e);
      }
    }

    this.imageTokenId = specialTokens['<image>'] ?? null;
    this.imageStartTokenId = specialTokens['<|image_start|>'] ?? null;
    this.imageEndTokenId = specialTokens['<|image_end|>'] ?? null;
    this.eosTokenId = this.tokenizer.eos_token_id;

    report('loading', 10, 'config');
    const configResponse = await fetch(`${modelPath}/config.json`, {
      mode: 'cors', credentials: 'omit',
    });
    this.config = await configResponse.json();
    const textConfig = this.config.text_config || this.config;
    this.hiddenSize = textConfig.hidden_size || 1024;
    this.numKVHeads = textConfig.num_key_value_heads || 8;
    this.headDim = Math.floor(this.hiddenSize / (textConfig.num_attention_heads || 16));

    const fetchOptions = { mode: 'cors', credentials: 'omit' };

    const getExternalDataFiles = async (basePath, fileName) => {
      const fileCount = EXTERNAL_DATA_FILE_COUNTS[fileName] || 1;
      const files = [];

      const primaryUrl = `${basePath}/onnx/${fileName}.onnx_data`;
      try {
        const headResp = await fetch(primaryUrl, { method: 'HEAD', ...fetchOptions });
        if (!headResp.ok) return [];
        files.push({
          path: `${fileName}.onnx_data`,
          url: primaryUrl,
          size: parseInt(headResp.headers.get('content-length') || '0', 10),
        });
      } catch (e) {
        return [];
      }

      for (let i = 1; i < fileCount; i++) {
        const url = `${basePath}/onnx/${fileName}.onnx_data_${i}`;
        try {
          const resp = await fetch(url, { method: 'HEAD', ...fetchOptions });
          if (resp.ok) {
            files.push({
              path: `${fileName}.onnx_data_${i}`,
              url,
              size: parseInt(resp.headers.get('content-length') || '0', 10),
            });
          }
        } catch (e) { /* expected */ }
      }

      return files;
    };

    const loadOnnxWithExternalData = async (name, progress, quantSuffix = quantization, customProviders = null) => {
      const suffix = quantSuffix ? `_${quantSuffix}` : '';
      const fileName = `${name}${suffix}`;
      report('loading', progress, `${fileName}.onnx`);

      const onnxPath = `${modelPath}/onnx/${fileName}.onnx`;

      const makeProgressCallback = (file) => (received, total) => {
        const mb = (received / 1024 / 1024).toFixed(0);
        const totalMb = (total / 1024 / 1024).toFixed(0);
        report('loading', progress, `${file}: ${mb} / ${totalMb} MB`);
      };

      const dataFiles = await getExternalDataFiles(modelPath, fileName);
      const providers = customProviders || executionProviders;
      const sessionOptions = { executionProviders: providers };

      const onnxResponse = await fetchWithCache(onnxPath, fetchOptions, makeProgressCallback(`${fileName}.onnx`));
      if (!onnxResponse.ok) throw new Error(`Failed to fetch ${fileName}.onnx: ${onnxResponse.status}`);
      const onnxBuffer = await onnxResponse.arrayBuffer();

      if (dataFiles.length > 0) {
        sessionOptions.externalData = [];
        for (const f of dataFiles) {
          if (f.size > LARGE_FILE_THRESHOLD) {
            sessionOptions.externalData.push({ path: f.path, data: f.url });
          } else {
            const dataResponse = await fetchWithCache(f.url, fetchOptions, makeProgressCallback(f.path));
            if (!dataResponse.ok) throw new Error(`Failed to fetch ${f.path}: ${dataResponse.status}`);
            const dataBuffer = await dataResponse.arrayBuffer();
            sessionOptions.externalData.push({ path: f.path, data: new Uint8Array(dataBuffer) });
          }
        }
      }

      return ort.InferenceSession.create(new Uint8Array(onnxBuffer), sessionOptions);
    };

    const quantConfig = typeof quantization === 'object' ? quantization : {
      decoder: quantization,
      embedImages: quantization === 'q4' ? 'q8' : quantization,
    };

    this.embedTokensSession = await loadOnnxWithExternalData('embed_tokens', 20, quantConfig.decoder ? 'fp16' : null);
    this.embedImagesSession = await loadOnnxWithExternalData('embed_images', 40, quantConfig.embedImages || null);
    this.decoderSession    = await loadOnnxWithExternalData('decoder',       60, quantConfig.decoder    || null);

    report('done', 100, '');
  }

  async getImageEmbeddings(imageInputs) {
    const allEmbeddings = [];
    const tokensPerImage = [];

    for (const input of imageInputs) {
      if (this.imageCache.has(input)) {
        const cached = this.imageCache.get(input);
        allEmbeddings.push(cached.embeddings);
        tokensPerImage.push(cached.numTokens);
        continue;
      }

      const img = await loadImage(input);
      const processed = await processImage(img);

      const pixelValuesTensor = new ort.Tensor('float32', processed.pixelValues, processed.shape);
      const attentionMaskTensor = new ort.Tensor('int64', processed.attentionMask, [processed.numTiles, processed.shape[1]]);
      const spatialShapesTensor = new ort.Tensor('int64', processed.spatialShapes, [processed.numTiles, 2]);

      const outputs = await this.embedImagesSession.run({
        pixel_values: pixelValuesTensor,
        pixel_attention_mask: attentionMaskTensor,
        spatial_shapes: spatialShapesTensor,
      });

      const embeddings = outputs.image_features;
      const numTokens = embeddings.dims[0];
      const embeddingsCopy = new Float32Array(embeddings.data);
      this.imageCache.set(input, { embeddings: embeddingsCopy, numTokens });

      tokensPerImage.push(numTokens);
      allEmbeddings.push(embeddingsCopy);
    }

    const totalLength = allEmbeddings.reduce((sum, e) => sum + e.length, 0);
    const combined = new Float32Array(totalLength);
    let offset = 0;
    for (const emb of allEmbeddings) {
      combined.set(emb, offset);
      offset += emb.length;
    }

    return { embeddings: combined, numTokens: totalLength / this.hiddenSize, tokensPerImage };
  }

  async getTextEmbeddings(inputIds) {
    const inputTensor = new ort.Tensor(
      'int64',
      new BigInt64Array(inputIds.map(id => BigInt(id))),
      [1, inputIds.length]
    );
    const outputs = await this.embedTokensSession.run({ input_ids: inputTensor });
    return outputs.inputs_embeds;
  }

  buildCombinedEmbeddings1to1(inputIds, textEmbeddings, imageEmbeddings) {
    const [, , hiddenDim] = textEmbeddings.dims;
    const textEmb = textEmbeddings.data;
    const imgEmb = imageEmbeddings;

    const imagePositions = [];
    for (let i = 0; i < inputIds.length; i++) {
      if (inputIds[i] === this.imageTokenId) imagePositions.push(i);
    }

    const result = new Float32Array(textEmb);
    const numImageEmbeddings = imgEmb.length / hiddenDim;

    for (let i = 0; i < Math.min(imagePositions.length, numImageEmbeddings); i++) {
      const pos = imagePositions[i];
      const embStart = i * hiddenDim;
      result.set(imgEmb.slice(embStart, embStart + hiddenDim), pos * hiddenDim);
    }

    return new ort.Tensor('float32', result, textEmbeddings.dims);
  }

  initializeCache() {
    const cache = {};
    for (const name of this.decoderSession.inputNames) {
      if (name.startsWith('past_conv')) {
        cache[name] = new ort.Tensor('float32', new Float32Array(1 * this.hiddenSize * 3), [1, this.hiddenSize, 3]);
      } else if (name.startsWith('past_key_values')) {
        cache[name] = new ort.Tensor('float32', new Float32Array(0), [1, this.numKVHeads, 0, this.headDim]);
      }
    }
    return cache;
  }

  updateCache(cache, outputs) {
    for (const name of Object.keys(outputs)) {
      if (name.startsWith('present_conv')) {
        const cacheName = name.replace('present_conv', 'past_conv');
        if (cacheName in cache) cache[cacheName] = outputs[name];
      } else if (name.startsWith('present.')) {
        const cacheName = name.replace('present.', 'past_key_values.');
        if (cacheName in cache) cache[cacheName] = outputs[name];
      }
    }
  }

  async generate(messages, options = {}) {
    const { maxNewTokens = 256, onToken, images = [], messageImageMap = new Map() } = options;

    let imageEmbeddings = null;
    let tokensPerImage = [];

    if (images.length > 0) {
      const result = await this.getImageEmbeddings(images);
      imageEmbeddings = result.embeddings;
      tokensPerImage = result.tokensPerImage;
    }

    let promptMessages = messages;
    if (images.length > 0) {
      promptMessages = messages.map((msg, idx) => {
        if (msg.role === 'user' && messageImageMap.has(idx)) {
          const messageImages = messageImageMap.get(idx);
          const imageTokens = messageImages.map(() => '<image>').join('');
          return { ...msg, content: imageTokens + msg.content };
        }
        return msg;
      });
    }

    const prompt = this.tokenizer.apply_chat_template(promptMessages, {
      add_generation_prompt: true,
      tokenize: false,
    });

    const encoded = this.tokenizer.encode(prompt);
    let inputIds = [...encoded];

    if (images.length > 0) {
      const expandedIds = [];
      let imageIdx = 0;
      for (const id of inputIds) {
        if (id === this.imageTokenId && imageIdx < tokensPerImage.length) {
          if (this.imageStartTokenId) expandedIds.push(this.imageStartTokenId);
          const count = tokensPerImage[imageIdx];
          for (let i = 0; i < count; i++) expandedIds.push(this.imageTokenId);
          if (this.imageEndTokenId) expandedIds.push(this.imageEndTokenId);
          imageIdx++;
        } else {
          expandedIds.push(id);
        }
      }
      inputIds = expandedIds;
    }

    const textEmbeddings = await this.getTextEmbeddings(inputIds);
    let inputsEmbeds = images.length > 0
      ? this.buildCombinedEmbeddings1to1(inputIds, textEmbeddings, imageEmbeddings)
      : textEmbeddings;

    const cache = this.initializeCache();
    let curLen = inputsEmbeds.dims[1];
    let currentEmbeds = inputsEmbeds;
    const generatedTokens = [];

    for (let step = 0; step < maxNewTokens; step++) {
      const attentionMask = new ort.Tensor('int64', new BigInt64Array(curLen).fill(1n), [1, curLen]);
      const feeds = { inputs_embeds: currentEmbeds, attention_mask: attentionMask, ...cache };
      const outputs = await this.decoderSession.run(feeds);

      const logits = outputs.logits;
      const vocabSize = logits.dims[2];
      const logitsData = logits.data;
      const lastLogitStart = (logits.dims[1] - 1) * vocabSize;
      const lastLogits = logitsData.slice(lastLogitStart, lastLogitStart + vocabSize);

      let maxIdx = 0;
      let maxVal = lastLogits[0];
      for (let i = 1; i < vocabSize; i++) {
        if (lastLogits[i] > maxVal) { maxVal = lastLogits[i]; maxIdx = i; }
      }

      generatedTokens.push(maxIdx);

      if (onToken) {
        const tokenText = this.tokenizer.decode([maxIdx]);
        if (onToken(tokenText, maxIdx)) break;
      }

      if (maxIdx === this.eosTokenId) break;

      this.updateCache(cache, outputs);
      currentEmbeds = await this.getTextEmbeddings([maxIdx]);
      curLen++;
    }

    return this.tokenizer.decode(generatedTokens, { skip_special_tokens: true });
  }

  async dispose() {
    this.clearImageCache();
    this.tokenizer = null;
    for (const s of [this.embedTokensSession, this.embedImagesSession, this.decoderSession]) {
      if (s) { try { await s.release(); } catch (e) { /* ignore */ } }
    }
    this.embedTokensSession = null;
    this.embedImagesSession = null;
    this.decoderSession = null;
  }
}

export default VLModel;
