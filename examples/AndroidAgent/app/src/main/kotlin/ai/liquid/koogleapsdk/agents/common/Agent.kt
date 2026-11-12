package ai.liquid.koogleapsdk.agents.common

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.addJsonObject
import kotlinx.serialization.json.buildJsonArray
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.put
import kotlinx.serialization.json.putJsonObject

/**
 * A tool-calling agent: the model either calls tools or answers. Tool results go
 * back to the model until it answers in plain text or [maxIterations] is reached.
 */
class Agent(
    private val llm: LlamaChat,
    private val systemPrompt: String,
    private val tools: List<Tool<*, *>>,
    private val maxIterations: Int,
    private val onToolCallEvent: suspend (String) -> Unit,
    private val onErrorEvent: suspend (String) -> Unit,
    private val onAssistantMessage: suspend (String) -> String,
) {
    private val json = Json { ignoreUnknownKeys = true; isLenient = true; coerceInputValues = true }

    private val toolDefinitions = buildJsonArray {
        tools.forEach { tool ->
            addJsonObject {
                put("type", "function")
                putJsonObject("function") {
                    put("name", tool.name)
                    put("description", tool.description)
                    put("parameters", tool.parameters)
                }
            }
        }
    }

    suspend fun run(input: String): String = try {
        val messages = mutableListOf(message("system", systemPrompt), message("user", input))
        var answer: String? = null
        for (iteration in 1..maxIterations) {
            val reply = withContext(Dispatchers.Default) {
                llm.chat(buildJsonArray { messages.forEach { add(it) } }, toolDefinitions)
            }
            messages += reply

            val toolCalls = reply["tool_calls"]?.jsonArray.orEmpty()
            if (toolCalls.isEmpty()) {
                answer = onAssistantMessage(reply["content"]?.jsonPrimitive?.content.orEmpty())
                break
            }
            toolCalls.forEachIndexed { index, call ->
                val function = call.jsonObject.getValue("function").jsonObject
                val name = function.getValue("name").jsonPrimitive.content
                val arguments = function.getValue("arguments").jsonPrimitive.content
                onToolCallEvent("Tool $name, args $arguments")
                messages += buildJsonObject {
                    put("role", "tool")
                    put("tool_call_id", call.jsonObject["id"]?.jsonPrimitive?.content ?: "call_$index")
                    put("content", callTool(name, arguments))
                }
            }
        }
        answer ?: error("The agent did not answer within $maxIterations iterations")
    } catch (e: Exception) {
        onErrorEvent("${e.message}")
        throw e
    }

    // Runs one tool call. Errors go back to the model as the result, so it can retry.
    private suspend fun callTool(name: String, arguments: String): String {
        val tool = tools.find { it.name == name } ?: return "Error: unknown tool '$name'"
        return try {
            tool.call(arguments)
        } catch (e: Exception) {
            "Error: ${e.message}"
        }
    }

    private suspend fun <Args, Result> Tool<Args, Result>.call(arguments: String): String {
        val args = json.decodeFromString(argsSerializer, arguments.ifBlank { "{}" })
        return resultText(execute(args))
    }

    private fun message(role: String, content: String): JsonObject = buildJsonObject {
        put("role", role)
        put("content", content)
    }
}
