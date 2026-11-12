package ai.liquid.koogleapsdk.agents.common

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject

/**
 * Minimal Kotlin wrapper around llama.cpp for tool calling
 * (see app/src/main/cpp/llama_agent_jni.cpp).
 *
 * @param nativeLibDir the app's native library folder, where llama.cpp's CPU backends are installed
 */
class LlamaChat(nativeLibDir: String) {
    init {
        System.loadLibrary("llama-agent-jni")
        nativeInit(nativeLibDir)
    }

    /** Loads the model. Call off the main thread. */
    fun load(modelPath: String) {
        check(nativeLoad(modelPath)) { "Failed to load the model" }
    }

    /**
     * Generates the next assistant message for [messages], given the [tools] the model may call.
     * Both use the OpenAI chat format. The reply's `tool_calls` are already parsed. Call off the main thread.
     */
    fun chat(messages: JsonArray, tools: JsonArray): JsonObject {
        val reply = Json.parseToJsonElement(nativeChat(messages.toString(), tools.toString())).jsonObject
        reply["error"]?.let { error("Generation failed: $it") }
        return reply
    }

    /** Frees the model and context. Call off the main thread. */
    fun unload() = nativeUnload()

    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeLoad(modelPath: String): Boolean
    private external fun nativeChat(messagesJson: String, toolsJson: String): String
    private external fun nativeUnload()
}
