// Central config — stub for Phase 1.
// Tool agent will populate ALL_TOOL_SCHEMAS; inference agent will use MODEL_CONFIG.

export type ExecutionProvider = 'webgpu' | 'wasm'

export interface ModelConfig {
  /** Human-readable model name shown in UI */
  name: string
  /** HuggingFace repo ID, e.g. "liquid-tech/LFM2-1.2B-Tool-ONNX" */
  hfRepo: string
  /** Filename within the repo */
  modelFile: string
  /** tokenizer.json filename */
  tokenizerFile: string
  /** Preferred execution provider (falls back to wasm if webgpu unavailable) */
  executionProvider: ExecutionProvider
  /** Generation parameters */
  generation: {
    maxNewTokens: number
    temperature: number
    topP: number
  }
}

export const MODEL_CONFIG: ModelConfig = {
  name: 'LFM2-1.2B-Tool',
  hfRepo: 'liquid-tech/LFM2-1.2B-Tool-ONNX',
  modelFile: 'model_q4.onnx',
  tokenizerFile: 'tokenizer.json',
  executionProvider:
    (import.meta.env.VITE_EXECUTION_PROVIDER as ExecutionProvider) ?? 'webgpu',
  generation: {
    maxNewTokens: 512,
    temperature: 0.1,
    topP: 0.9,
  },
}

// STUB: replaced by tool agent with the full schema array
// import { ALL_TOOL_SCHEMAS } from '@/tools/schemas'
// export const ACTIVE_TOOLS = ALL_TOOL_SCHEMAS
export const ACTIVE_TOOLS: unknown[] = []
