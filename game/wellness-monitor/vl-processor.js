/**
 * LFM2-VL Image Processor for WebGPU/ONNX Runtime Web
 *
 * Implements the image preprocessing logic from Lfm2VlImageProcessorFast:
 * 1. Split image into tiles (512x512)
 * 2. Extract 16x16 patches from each tile (32x32 = 1024 patches per tile)
 * 3. Flatten each patch to 768 values (16*16*3)
 * 4. Normalize: (pixel / 255 - 0.5) / 0.5 = pixel / 127.5 - 1
 *
 * Output shapes match Python processor:
 * - pixel_values: [num_tiles, 1024, 768]
 * - pixel_attention_mask: [num_tiles, 1024]
 */

const CONFIG = {
  tileSize: 512,
  maxTiles: 10,
  minTiles: 2,
  imageMean: [0.5, 0.5, 0.5],
  imageStd: [0.5, 0.5, 0.5],
  rescaleFactor: 1 / 255,
  useThumbnail: true,
  patchSize: 16,
  patchesPerTile: 32,
  downsampleFactor: 2,
  minImageTokens: 64,
  maxImageTokens: 256,
  maxPixelsTolerance: 2.0,
};

const NORM_SCALE = 1 / 127.5;
const NORM_OFFSET = -1.0;

const PRECOMPUTED_SIZES = {
  256: { width: 256, height: 256, patchesH: 16, patchesW: 16 },
  384: { width: 384, height: 384, patchesH: 24, patchesW: 24 },
  448: { width: 448, height: 448, patchesH: 28, patchesW: 28 },
  512: { width: 512, height: 512, patchesH: 32, patchesW: 32 },
};

function roundByFactor(number, factor) {
  return Math.round(number / factor) * factor;
}

function ceilByFactor(number, factor) {
  return Math.ceil(number / factor) * factor;
}

function floorByFactor(number, factor) {
  return Math.floor(number / factor) * factor;
}

function findClosestAspectRatio(aspectRatio, targetRatios, width, height, imageSize) {
  let bestRatioDiff = Infinity;
  let bestRatio = [1, 1];
  const area = width * height;

  for (const ratio of targetRatios) {
    const targetAspectRatio = ratio[0] / ratio[1];
    const ratioDiff = Math.abs(aspectRatio - targetAspectRatio);

    if (ratioDiff < bestRatioDiff) {
      bestRatioDiff = ratioDiff;
      bestRatio = ratio;
    } else if (ratioDiff === bestRatioDiff) {
      const targetArea = imageSize * imageSize * ratio[0] * ratio[1];
      if (area > 0.5 * targetArea) {
        bestRatio = ratio;
      }
    }
  }

  return bestRatio;
}

function isImageTooLarge(width, height) {
  const { patchSize, maxImageTokens, downsampleFactor, maxPixelsTolerance } = CONFIG;
  const hBar = Math.max(patchSize, roundByFactor(height, patchSize));
  const wBar = Math.max(patchSize, roundByFactor(width, patchSize));
  const maxPixels = maxImageTokens * (patchSize ** 2) * (downsampleFactor ** 2) * maxPixelsTolerance;
  return hBar * wBar > maxPixels;
}

function smartResize(width, height) {
  const { patchSize, downsampleFactor, minImageTokens, maxImageTokens } = CONFIG;
  const totalFactor = patchSize * downsampleFactor;
  const minPixels = minImageTokens * (patchSize ** 2) * (downsampleFactor ** 2);
  const maxPixels = maxImageTokens * (patchSize ** 2) * (downsampleFactor ** 2);

  let hBar = Math.max(totalFactor, roundByFactor(height, totalFactor));
  let wBar = Math.max(totalFactor, roundByFactor(width, totalFactor));

  if (hBar * wBar > maxPixels) {
    const beta = Math.sqrt((height * width) / maxPixels);
    hBar = Math.max(totalFactor, floorByFactor(height / beta, totalFactor));
    wBar = Math.max(totalFactor, floorByFactor(width / beta, totalFactor));
  } else if (hBar * wBar < minPixels) {
    const beta = Math.sqrt(minPixels / (height * width));
    hBar = ceilByFactor(height * beta, totalFactor);
    wBar = ceilByFactor(width * beta, totalFactor);
  }

  return { width: wBar, height: hBar };
}

function calculateTileGrid(width, height) {
  const { tileSize, minTiles, maxTiles } = CONFIG;
  const aspectRatio = width / height;

  const targetRatios = [];
  for (let n = minTiles; n <= maxTiles; n++) {
    for (let w = 1; w <= n; w++) {
      for (let h = 1; h <= n; h++) {
        if (w * h >= minTiles && w * h <= maxTiles) {
          if (!targetRatios.some(r => r[0] === w && r[1] === h)) {
            targetRatios.push([w, h]);
          }
        }
      }
    }
  }
  targetRatios.sort((a, b) => (a[0] * a[1]) - (b[0] * b[1]));

  if (targetRatios.length === 0) {
    return { rows: 1, cols: 1 };
  }

  const [gridWidth, gridHeight] = findClosestAspectRatio(
    aspectRatio, targetRatios, width, height, tileSize
  );

  return { rows: gridHeight, cols: gridWidth };
}

export async function processImage(image) {
  let width, height;
  let inputImageData = null;

  if (image instanceof ImageData) {
    width = image.width;
    height = image.height;
    inputImageData = image;
  } else if (image instanceof HTMLImageElement) {
    width = image.naturalWidth;
    height = image.naturalHeight;
  } else {
    width = image.width;
    height = image.height;
  }

  const { tileSize, patchSize, useThumbnail } = CONFIG;
  const patchesPerSide = CONFIG.patchesPerTile;
  const maxPatchesPerTile = patchesPerSide * patchesPerSide;
  const patchDim = patchSize * patchSize * 3;

  const needsSplitting = isImageTooLarge(width, height);

  if (needsSplitting) {
    const { rows, cols } = calculateTileGrid(width, height);
    const totalGridTiles = rows * cols;

    if (totalGridTiles > 1) {
      const numTiles = totalGridTiles + (useThumbnail ? 1 : 0);

      const pixelValues = new Float32Array(numTiles * maxPatchesPerTile * patchDim);
      const attentionMask = new BigInt64Array(numTiles * maxPatchesPerTile);
      const spatialShapes = new BigInt64Array(numTiles * 2);

      const targetWidth = tileSize * cols;
      const targetHeight = tileSize * rows;

      const resizedCanvas = document.createElement('canvas');
      resizedCanvas.width = targetWidth;
      resizedCanvas.height = targetHeight;
      const resizedCtx = resizedCanvas.getContext('2d');
      resizedCtx.drawImage(image, 0, 0, targetWidth, targetHeight);

      let tileIdx = 0;
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const tileCanvas = document.createElement('canvas');
          tileCanvas.width = tileSize;
          tileCanvas.height = tileSize;
          const tileCtx = tileCanvas.getContext('2d');

          tileCtx.drawImage(
            resizedCanvas,
            col * tileSize, row * tileSize, tileSize, tileSize,
            0, 0, tileSize, tileSize
          );

          const tileData = tileCtx.getImageData(0, 0, tileSize, tileSize);
          extractPatchesFromFullTile(tileData, pixelValues, attentionMask, tileIdx, patchesPerSide, maxPatchesPerTile);

          spatialShapes[tileIdx * 2] = BigInt(patchesPerSide);
          spatialShapes[tileIdx * 2 + 1] = BigInt(patchesPerSide);

          tileIdx++;
        }
      }

      if (useThumbnail) {
        const thumbResized = smartResize(width, height);
        const thumbWidth = thumbResized.width;
        const thumbHeight = thumbResized.height;

        const thumbCanvas = document.createElement('canvas');
        thumbCanvas.width = thumbWidth;
        thumbCanvas.height = thumbHeight;
        const thumbCtx = thumbCanvas.getContext('2d');
        thumbCtx.drawImage(image, 0, 0, thumbWidth, thumbHeight);

        const thumbData = thumbCtx.getImageData(0, 0, thumbWidth, thumbHeight);
        const thumbPatchesH = thumbHeight / patchSize;
        const thumbPatchesW = thumbWidth / patchSize;

        extractPatchesFromVariableSize(thumbData, pixelValues, attentionMask, tileIdx, thumbPatchesH, thumbPatchesW, maxPatchesPerTile);

        spatialShapes[tileIdx * 2] = BigInt(thumbPatchesH);
        spatialShapes[tileIdx * 2 + 1] = BigInt(thumbPatchesW);

        tileIdx++;
      }

      return {
        pixelValues,
        attentionMask,
        spatialShapes,
        numTiles,
        shape: [numTiles, maxPatchesPerTile, patchDim],
      };
    }
  }

  // SINGLE-TILE PATH
  let resizedWidth, resizedHeight, actualPatchesH, actualPatchesW;
  let imageData;

  const precomputed = PRECOMPUTED_SIZES[width];
  const isAlreadyAligned = precomputed && width === height;

  if (inputImageData && isAlreadyAligned) {
    resizedWidth = width;
    resizedHeight = height;
    actualPatchesH = precomputed.patchesH;
    actualPatchesW = precomputed.patchesW;
    imageData = inputImageData;
  } else if (isAlreadyAligned) {
    resizedWidth = width;
    resizedHeight = height;
    actualPatchesH = precomputed.patchesH;
    actualPatchesW = precomputed.patchesW;

    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = resizedWidth;
    resizedCanvas.height = resizedHeight;
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCtx.drawImage(image, 0, 0, resizedWidth, resizedHeight);
    imageData = resizedCtx.getImageData(0, 0, resizedWidth, resizedHeight);
  } else {
    const resized = smartResize(width, height);
    resizedWidth = resized.width;
    resizedHeight = resized.height;
    actualPatchesH = resizedHeight / patchSize;
    actualPatchesW = resizedWidth / patchSize;

    const resizedCanvas = document.createElement('canvas');
    resizedCanvas.width = resizedWidth;
    resizedCanvas.height = resizedHeight;
    const resizedCtx = resizedCanvas.getContext('2d');
    resizedCtx.drawImage(image, 0, 0, resizedWidth, resizedHeight);
    imageData = resizedCtx.getImageData(0, 0, resizedWidth, resizedHeight);
  }

  const numTiles = 1;
  const pixelValues = new Float32Array(numTiles * maxPatchesPerTile * patchDim);
  const attentionMask = new BigInt64Array(numTiles * maxPatchesPerTile);
  const spatialShapes = new BigInt64Array(numTiles * 2);

  extractPatchesFromVariableSize(imageData, pixelValues, attentionMask, 0, actualPatchesH, actualPatchesW, maxPatchesPerTile);

  spatialShapes[0] = BigInt(actualPatchesH);
  spatialShapes[1] = BigInt(actualPatchesW);

  return {
    pixelValues,
    attentionMask,
    spatialShapes,
    numTiles,
    shape: [numTiles, maxPatchesPerTile, patchDim],
  };
}

function extractPatchesFromFullTile(tileData, pixelValues, attentionMask, tileIdx, patchesPerSide, maxPatchesPerTile) {
  const patchSize = CONFIG.patchSize;
  const patchDim = patchSize * patchSize * 3;
  const tileWidth = tileData.width;

  const pixels = tileData.data;
  const tileOffset = tileIdx * maxPatchesPerTile * patchDim;
  const maskOffset = tileIdx * maxPatchesPerTile;

  let patchIdx = 0;

  for (let py = 0; py < patchesPerSide; py++) {
    for (let px = 0; px < patchesPerSide; px++) {
      const patchStartX = px * patchSize;
      const patchStartY = py * patchSize;

      attentionMask[maskOffset + patchIdx] = 1n;

      const patchOffset = tileOffset + patchIdx * patchDim;
      let outIdx = 0;

      for (let dy = 0; dy < patchSize; dy++) {
        const rowOffset = (patchStartY + dy) * tileWidth;
        for (let dx = 0; dx < patchSize; dx++) {
          const srcIdx = (rowOffset + patchStartX + dx) * 4;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx] * NORM_SCALE + NORM_OFFSET;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx + 1] * NORM_SCALE + NORM_OFFSET;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx + 2] * NORM_SCALE + NORM_OFFSET;
        }
      }

      patchIdx++;
    }
  }
}

function extractPatchesFromVariableSize(imageData, pixelValues, attentionMask, tileIdx, patchesH, patchesW, maxPatchesPerTile) {
  const patchSize = CONFIG.patchSize;
  const patchDim = patchSize * patchSize * 3;
  const imageWidth = imageData.width;

  const pixels = imageData.data;
  const tileOffset = tileIdx * maxPatchesPerTile * patchDim;
  const maskOffset = tileIdx * maxPatchesPerTile;

  const actualPatches = patchesH * patchesW;

  let patchIdx = 0;
  for (let py = 0; py < patchesH; py++) {
    for (let px = 0; px < patchesW; px++) {
      const patchStartX = px * patchSize;
      const patchStartY = py * patchSize;

      attentionMask[maskOffset + patchIdx] = 1n;

      const patchOffset = tileOffset + patchIdx * patchDim;
      let outIdx = 0;

      for (let dy = 0; dy < patchSize; dy++) {
        const rowOffset = (patchStartY + dy) * imageWidth;
        for (let dx = 0; dx < patchSize; dx++) {
          const srcIdx = (rowOffset + patchStartX + dx) * 4;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx] * NORM_SCALE + NORM_OFFSET;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx + 1] * NORM_SCALE + NORM_OFFSET;
          pixelValues[patchOffset + outIdx++] = pixels[srcIdx + 2] * NORM_SCALE + NORM_OFFSET;
        }
      }

      patchIdx++;
    }
  }

  for (let i = actualPatches; i < maxPatchesPerTile; i++) {
    attentionMask[maskOffset + i] = 0n;
  }
}

export function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => resolve(img);
    img.onerror = reject;
    img.src = src;
  });
}
