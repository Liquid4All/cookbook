package ai.liquid.androidkoogagent.agents.common

import ai.koog.agents.core.tools.ToolDescriptor
import ai.koog.prompt.dsl.ModerationResult
import ai.koog.prompt.dsl.Prompt
import ai.koog.prompt.executor.clients.LLMClient
import ai.koog.prompt.llm.LLMCapability
import ai.koog.prompt.llm.LLMProvider
import ai.koog.prompt.llm.LLModel
import ai.koog.prompt.message.Message
import ai.koog.prompt.message.ResponseMetaInfo
import ai.liquid.androidkoogagent.App
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.arm.aichat.isModelLoaded
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.flow.toList
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.put
import java.util.UUID

/** Koog's model handle for the GGUF loaded by the native Android llama.cpp module. */
val LocalLfmModel = LLModel(
    provider = LLMProvider.Meta,
    id = "LiquidAI/LFM3-1.7B-GGUF",
    capabilities = listOf(LLMCapability.Temperature, LLMCapability.Tools),
    contextLength = 8192,
    maxOutputTokens = 1024,
)

/**
 * Koog calls this client for each agent step. The model stays in the app process;
 * no network service or API key is needed for inference.
 */
object LocalLfmClient : LLMClient {
    private val mutex = Mutex()
    private val engine by lazy { AiChat.getInferenceEngine(App.context) }
    private val toolCall = Regex(
        "<tool_call>\\s*<function=([^>]+)>(.*?)</function>\\s*</tool_call>",
        RegexOption.DOT_MATCHES_ALL,
    )
    private val parameter = Regex(
        "<parameter=([^>]+)>(.*?)</parameter>",
        RegexOption.DOT_MATCHES_ALL,
    )
    private val thinking = Regex("<think>.*?</think>", RegexOption.DOT_MATCHES_ALL)

    override fun llmProvider(): LLMProvider = LLMProvider.Meta

    override suspend fun moderate(prompt: Prompt, model: LLModel): ModerationResult =
        throw UnsupportedOperationException("On-device moderation is not part of this example")

    override suspend fun execute(
        prompt: Prompt,
        model: LLModel,
        tools: List<ToolDescriptor>,
    ): List<Message.Response> = mutex.withLock {
        require(model.id == LocalLfmModel.id) { "Unexpected model: ${model.id}" }
        ensureModelLoaded()
        engine.resetConversation()
        engine.setSystemPrompt(systemPrompt(prompt, tools))
        val transcript = prompt.messages
            .filterNot { it.role == Message.Role.System }
            .joinToString("\n") { message ->
                val label = when (message) {
                    is Message.Tool.Call -> "Assistant tool call ${message.tool}"
                    is Message.Tool.Result -> "Tool result ${message.tool}"
                    else -> message.role.name
                }
                "$label: ${message.content}"
            }
        val raw = engine.sendUserPrompt(transcript, 1024).toList().joinToString("")
        val calls = toolCall.findAll(raw).map { match ->
            val name = match.groupValues[1].trim()
            require(tools.any { it.name == name }) { "Model requested unknown tool: $name" }
            val args = buildJsonObject {
                parameter.findAll(match.groupValues[2]).forEach { p ->
                    val value = p.groupValues[2].trim()
                    val argument = value.toLongOrNull()?.let { JsonPrimitive(it) }
                        ?: value.toDoubleOrNull()?.let { JsonPrimitive(it) }
                        ?: JsonPrimitive(value)
                    put(p.groupValues[1].trim(), argument)
                }
            }
            Message.Tool.Call(
                id = UUID.randomUUID().toString(),
                tool = name,
                content = args.toString(),
                metaInfo = ResponseMetaInfo.Empty,
            )
        }.toList()
        if (calls.isNotEmpty()) {
            calls
        } else {
            val answer = if (raw.contains("<think>") && !raw.contains("</think>")) {
                "The model stopped while reasoning. Try a shorter request."
            } else {
                raw.replace(thinking, "").trim()
            }
            listOf(
                Message.Assistant(
                    content = answer,
                    metaInfo = ResponseMetaInfo.Empty,
                )
            )
        }
    }

    private suspend fun ensureModelLoaded() {
        val state = engine.state.first {
            it is InferenceEngine.State.Initialized ||
                it.isModelLoaded ||
                it is InferenceEngine.State.Error
        }
        if (state is InferenceEngine.State.Error) throw state.exception
        if (state is InferenceEngine.State.Initialized) {
            require(modelPath.isFile) {
                "Model missing: import the LFM3-1.7B GGUF from the tool list."
            }
            engine.loadModel(modelPath.absolutePath)
        }
    }

    private fun systemPrompt(prompt: Prompt, tools: List<ToolDescriptor>): String {
        val instructions = prompt.messages
            .filterIsInstance<Message.System>()
            .joinToString("\n") { it.content }
        val declarations = tools.joinToString("\n") { tool ->
            val parameters = (tool.requiredParameters + tool.optionalParameters)
                .joinToString(", ") { "${it.name}: ${it.type.name.lowercase()}" }
            "${tool.name}($parameters): ${tool.description}"
        }
        return """
            $instructions

            Available tools:
            <tools>
            $declarations
            </tools>
            If a tool is needed, use LFM3 tool-call markup exactly:
            <tool_call><function=tool_name><parameter=argument_name>value</parameter></function></tool_call>
            Return plain text when no tool is needed. Never invent a tool result.
        """.trimIndent()
    }
}
