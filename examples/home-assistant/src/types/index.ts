// Shared TypeScript types — stub for Phase 1.
// Agents implementing tools, stores, and inference will expand this file.

export interface ChatMessage {
  id: string
  role: 'user' | 'assistant' | 'tool'
  content: string
  // toolCalls expanded by tool agent (Phase 3)
  toolCalls?: unknown[]
  toolCallId?: string
  isStreaming?: boolean
}
