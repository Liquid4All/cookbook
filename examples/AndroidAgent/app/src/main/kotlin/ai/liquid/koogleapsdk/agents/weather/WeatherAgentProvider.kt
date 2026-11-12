package ai.liquid.koogleapsdk.agents.weather

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.extension.nodeExecuteMultipleTools
import ai.koog.agents.core.dsl.extension.nodeLLMRequestMultiple
import ai.koog.agents.core.dsl.extension.nodeLLMSendMultipleToolResults
import ai.koog.agents.core.dsl.extension.onAssistantMessage
import ai.koog.agents.core.dsl.extension.onMultipleToolCalls
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.features.eventHandler.feature.handleEvents
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import ai.liquid.koogleapsdk.agents.common.AgentProvider
import ai.liquid.koogleapsdk.agents.common.ExitTool
import ai.liquid.koogleapsdk.agents.common.modelsPath
import io.github.lemcoder.koog.edge.leap.LeapModels
import io.github.lemcoder.koog.edge.leap.getLeapLLMClient
import kotlinx.datetime.Clock

/**
 * Factory for creating weather forecast agents
 */
internal class WeatherAgentProvider : AgentProvider {
    override val title: String = "Weather Forecast"
    override val description: String =
        "Hi, I'm a weather agent. I can provide weather forecasts for any location."

    override suspend fun provideAgent(
        onToolCallEvent: suspend (String) -> Unit,
        onErrorEvent: suspend (String) -> Unit,
        onAssistantMessage: suspend (String) -> String,
    ): AIAgent<String, String> {
        val leapExecutor = SingleLLMPromptExecutor(getLeapLLMClient(modelsPath))

        // Create tool registry with weather tools
        val toolRegistry = ToolRegistry {
            tool(WeatherTools.CurrentDatetimeTool)
            tool(WeatherTools.AddDatetimeTool)
            tool(WeatherTools.WeatherForecastTool)

            tool(ExitTool)
        }

        @Suppress("DuplicatedCode")
        val strategy = strategy(title) {
            val nodeRequestLLM by nodeLLMRequestMultiple()
            val nodeAssistantMessage by node<String, String> { message -> onAssistantMessage(message) }
            val nodeExecuteToolMultiple by nodeExecuteMultipleTools(parallelTools = true)
            val nodeSendToolResultMultiple by nodeLLMSendMultipleToolResults()

            edge(nodeStart forwardTo nodeRequestLLM)

            edge(
                nodeRequestLLM forwardTo nodeExecuteToolMultiple
                        onMultipleToolCalls { true }
            )

            edge(
                nodeRequestLLM forwardTo nodeAssistantMessage
                        transformed { it.first() }
                        onAssistantMessage { true }
            )

            edge(
                nodeExecuteToolMultiple forwardTo nodeSendToolResultMultiple
            )

            edge(
                nodeSendToolResultMultiple forwardTo nodeAssistantMessage
                        transformed { it.first() }
                        onAssistantMessage { true }
            )

            edge(
                nodeAssistantMessage forwardTo nodeFinish
                        transformed { it }
            )
        }

        // Create agent config with proper prompt
        val agentConfig = AIAgentConfig(
            prompt = prompt("test") {
                system(
                    """
                    You are a helpful weather assistant.
                    You can provide weather forecasts for any location in the world and help the user plan their activities.
                    ALWAYS use the available tools to get weather data. NEVER say you do not have access to weather data.
                    ALWAYS use date and time tools to handle dates and times.
                    Today's date and time is ${Clock.System.now()}.
                    When you receive a tool result, always explain it to the user in natural language.
                    Use the tools at your disposal to:
                    1. Get the current date and time
                    2. Add days, hours, or minutes to a date
                    3. Get weather forecasts for specific locations and dates
                    Do not say you lack access to data; always use the tools.
                    """.trimIndent()
                )
            },
            model = LeapModels.Chat.LFM2_1_2B_Tool,
            maxAgentIterations = 50
        )

        // Return the agent
        return AIAgent(
            promptExecutor = leapExecutor,
            strategy = strategy,
            agentConfig = agentConfig,
            toolRegistry = toolRegistry,
        ) {
            handleEvents {
                onToolCallStarting { ctx ->
                    onToolCallEvent("Tool ${ctx.tool.name}, args ${ctx.toolArgs}")
                }

                onAgentExecutionFailed { ctx ->
                    onErrorEvent("${ctx.throwable.message}")
                }

                onAgentCompleted { ctx ->
                    // Skip finish event handling
                }
            }
        }
    }
}