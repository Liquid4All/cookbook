package ai.liquid.browsercontrol

import android.content.Context

/**
 * Browser control agent that uses LFM2-350M (GGUF) via LEAP SDK
 * to generate browser actions from DOM content and a goal.
 *
 * Action space:
 *   click(bid)        - Click element with BrowserGym ID
 *   fill(bid, text)   - Fill input field with text
 *   send_keys(text)   - Send keyboard input
 *   scroll(direction) - Scroll up/down
 *   noop()            - Do nothing
 */
class BrowserAgent private constructor(
    private val context: Context,
    private val modelPath: String?,
) {
    companion object {
        private const val SYSTEM_PROMPT = """You control a web browser through BrowserGym actions.
You must complete the given web task by interacting with the page.

Available actions:
- noop() - Do nothing
- click(bid) - Click element with BrowserGym ID (the number in brackets)
- fill(bid, text) - Fill input field with text
- send_keys(text) - Send keyboard input
- scroll(direction) - Scroll up/down

The page structure shows elements as: [bid] element_type 'element_text'
For example: [13] button 'Click Me!' means bid='13'

Reply with exactly ONE action on a single line, e.g.:
click('13')
fill('42', 'hello world')
noop()

Do not include explanations or multiple actions."""

        /**
         * Create a mock agent for demo purposes (no actual model loaded).
         */
        fun createMock(context: Context): BrowserAgent {
            return BrowserAgent(context, null)
        }

        /**
         * Create an agent with a real GGUF model via LEAP SDK.
         * Requires LEAP SDK dependency and a valid model file.
         *
         * @param context Android context
         * @param modelPath Path to the GGUF model file
         */
        fun createFromFile(context: Context, modelPath: String): BrowserAgent {
            // TODO: Initialize LEAP SDK LlmInference here
            // val llm = LlmInference.createFromFile(context, modelPath)
            return BrowserAgent(context, modelPath)
        }
    }

    /**
     * Execute one inference step given the current DOM and goal.
     *
     * @param domContent Current page DOM in BrowserGym format
     * @param goal The task goal
     * @return A BrowserGym action string
     */
    suspend fun executeStep(domContent: String, goal: String): String {
        if (modelPath == null) {
            // Mock mode: return a demo action
            return mockInference(domContent, goal)
        }

        val prompt = buildPrompt(domContent, goal)

        // TODO: Replace with actual LEAP SDK inference
        // return withContext(Dispatchers.IO) {
        //     llm.generateResponse(prompt)
        // }
        return mockInference(domContent, goal)
    }

    private fun buildPrompt(domContent: String, goal: String): String {
        return """$SYSTEM_PROMPT

Goal: $goal

Current page:
$domContent

Action:"""
    }

    private fun mockInference(domContent: String, goal: String): String {
        // Simple heuristic for demo purposes
        return when {
            "button" in domContent && "submit" in domContent.lowercase() -> "click('1')"
            "input" in domContent -> "fill('2', 'demo input')"
            else -> "noop()"
        }
    }
}
