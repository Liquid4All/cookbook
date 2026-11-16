package ai.liquid.koogleapsdk.agents.common

import ai.liquid.android.runtime.NativeAgentModel
import java.io.File
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonObject

class LlamaChat(nativeLibDir: String) {
    private val model = NativeAgentModel(nativeLibDir)
    suspend fun load(modelPath: String) = model.load(File(modelPath))
    suspend fun chat(messages: JsonArray, tools: JsonArray): JsonObject {
        val reply = Json.parseToJsonElement(model.chat(messages.toString(), tools.toString())).jsonObject
        reply["error"]?.let { error("Generation failed: $it") }
        return reply
    }
    fun close() = model.close()
}
