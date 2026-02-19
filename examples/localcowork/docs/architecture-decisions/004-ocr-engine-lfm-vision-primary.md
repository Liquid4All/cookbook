# ADR-004: OCR Engine — LFM Vision Primary, Tesseract Fallback

## Status
Accepted

## Context
LocalCowork's OCR server needs to extract text, structured data, and tables from images, scanned PDFs, and receipts. The current implementation provides three engine slots (PaddleOCR → Tesseract → error), but the project bundles a local LLM (LFM2.5-24B in production) that has vision/multimodal capabilities.

The question: should OCR go through a traditional engine (Tesseract/PaddleOCR) or through the local vision model?

## Decision
Use the local LFM Vision model as the **primary OCR engine**, with Tesseract as a **lightweight fallback**. PaddleOCR is dropped as a required dependency.

### Engine Priority Chain

```
1. LFM Vision (via model-gateway)  — primary, highest accuracy
2. Tesseract (via pytesseract)     — fallback, no GPU needed
3. pypdf text extraction           — for digital PDFs (no OCR needed)
```

## Rationale

| Factor | LFM Vision | Tesseract | PaddleOCR |
|--------|-----------|-----------|-----------|
| Structured extraction | Native — model understands layout | Requires post-processing | Good for tables |
| Accuracy on receipts | High — contextual understanding | Medium — character-level | High but heavy |
| Table extraction | Excellent — understands rows/cols | Poor — needs heuristics | Good |
| Binary size | 0 (already bundled) | ~30 MB | ~500 MB + PaddlePaddle |
| GPU requirement | Uses same GPU as LLM | CPU only | GPU recommended |
| Latency | ~2-5s per page | ~0.5-1s per page | ~1-3s per page |
| Offline support | Yes (local model) | Yes | Yes |

Three decisive factors:

1. **Already bundled.** The LFM model is already loaded in memory for chat. Using it for OCR adds zero binary size and reuses warm GPU memory. PaddleOCR adds 500MB+ of dependencies for a capability the LLM already has.

2. **Structured extraction is native.** The `ocr.extract_structured_data` tool currently uses regex heuristics to parse amounts, dates, and emails from OCR text. The LFM Vision model can directly output structured JSON from an image — eliminating the lossy OCR→text→regex pipeline.

3. **Tesseract covers the gap.** When the model server isn't running (e.g., during development, CI, or low-memory situations), Tesseract provides fast, lightweight OCR for basic text extraction. It's a system package (~30MB) with no Python ML dependencies.

## Implementation

### Tool Behavior

Each OCR tool follows the same pattern:

```python
async def execute(self, params):
    # 1. Try LFM Vision via model-gateway
    try:
        result = await model_gateway.vision_extract(
            image_path=params.path,
            prompt="Extract all text from this image",
            response_format={"type": "json_schema", ...}
        )
        return result
    except ModelUnavailableError:
        pass  # Model not running, fall through

    # 2. Fall back to Tesseract
    try:
        return _ocr_with_tesseract(params.path)
    except ImportError:
        pass

    # 3. Error — no engine available
    raise MCPError("No OCR engine available")
```

### Model Gateway Extension

The model-gateway service gains a `vision_extract()` method that:
- Sends the image as a base64-encoded data URL in the OpenAI vision format
- Includes a system prompt tailored to the extraction task (text, table, structured)
- Returns typed results matching the tool's return schema

### When Each Engine Is Used

| Scenario | Engine |
|----------|--------|
| Normal operation (model loaded) | LFM Vision |
| Model server not running | Tesseract |
| CI / automated testing | Tesseract (mocked) |
| Digital PDFs (not scanned) | pypdf (no OCR) |
| Quick text-only extraction | Tesseract (optional optimization) |

## Consequences

### Positive
- Eliminates PaddleOCR/PaddlePaddle dependency (~500MB saved)
- Structured data extraction becomes native (no regex heuristics)
- Table extraction accuracy improves significantly
- Single model serves both chat and vision — simpler operations

### Negative
- OCR latency increases when using vision model (~2-5s vs ~0.5s)
- Requires model server to be running for best accuracy
- Vision inference competes with chat inference for GPU time

### Risks
- If LFM2.5-24B's vision quality is poor, we fall back to Tesseract everywhere
- Mitigation: model-behavior tests include OCR accuracy benchmarks; if below threshold, auto-select Tesseract

## Migration Path
1. Current: Tesseract fallback chain (implemented in WS-1C)
2. WS-2B (Inference Client): Add vision endpoint support
3. WS-4A+ : Wire OCR tools to model-gateway `vision_extract()`
4. Integration tests: Verify accuracy against test fixtures (receipts, contracts, forms)
