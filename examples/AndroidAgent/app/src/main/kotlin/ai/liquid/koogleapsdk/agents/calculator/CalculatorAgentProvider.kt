package ai.liquid.koogleapsdk.agents.calculator

import ai.koog.agents.core.agent.AIAgent
import ai.koog.agents.core.agent.config.AIAgentConfig
import ai.koog.agents.core.dsl.builder.forwardTo
import ai.koog.agents.core.dsl.builder.strategy
import ai.koog.agents.core.dsl.extension.nodeExecuteTool
import ai.koog.agents.core.dsl.extension.nodeLLMRequest
import ai.koog.agents.core.dsl.extension.nodeLLMSendToolResult
import ai.koog.agents.core.dsl.extension.onAssistantMessage
import ai.koog.agents.core.dsl.extension.onToolCall
import ai.koog.agents.core.tools.ToolRegistry
import ai.koog.agents.features.eventHandler.feature.handleEvents
import ai.koog.prompt.dsl.prompt
import ai.koog.prompt.executor.llms.SingleLLMPromptExecutor
import ai.liquid.koogleapsdk.agents.common.AgentProvider
import ai.liquid.koogleapsdk.agents.common.ExitTool
import ai.liquid.koogleapsdk.agents.common.modelsPath
import io.github.lemcoder.koog.edge.leap.LeapLLMParams
import io.github.lemcoder.koog.edge.leap.LeapModels
import io.github.lemcoder.koog.edge.leap.getLeapLLMClient

/**
 * Factory for creating calculator agents
 */
internal class CalculatorAgentProvider : AgentProvider {
    override val title: String = "Calculator"
    override val description: String = "Hi, I'm a calculator agent, I can do math"

    override suspend fun provideAgent(
        onToolCallEvent: suspend (String) -> Unit,
        onErrorEvent: suspend (String) -> Unit,
        onAssistantMessage: suspend (String) -> String,
    ): AIAgent<String, String> {
        val leapExecutor = SingleLLMPromptExecutor(getLeapLLMClient(modelsPath))

        // Create tool registry with calculator tools
        val toolRegistry = ToolRegistry {
            tool(CalculatorTools.PlusTool)
            tool(CalculatorTools.MinusTool)
            tool(CalculatorTools.DivideTool)
            tool(CalculatorTools.MultiplyTool)

            tool(ExitTool)
        }

        @Suppress("DuplicatedCode")
        val strategy = strategy(title) {
            val nodeRequestLLM by nodeLLMRequest()
            val nodeToolExecute by nodeExecuteTool()
            val nodeSendToolResult by nodeLLMSendToolResult()

            edge(nodeStart forwardTo nodeRequestLLM)

            edge(
                nodeRequestLLM forwardTo nodeToolExecute
                        onToolCall { ctx ->
                    onToolCallEvent("Tool ${ctx.tool}")
                    true
                }
            )

            edge(
                nodeToolExecute forwardTo nodeSendToolResult
            )

            edge(
                nodeSendToolResult forwardTo nodeFinish
                        onAssistantMessage { ctx ->
                    onAssistantMessage(ctx.content)
                    true
                }
            )
        }

        // Create agent config with proper prompt
        val agentConfig = AIAgentConfig(
            prompt = prompt(
                "test",
                params = LeapLLMParams(
                    temperature = 0f
                )
            ) {
                system(
                    """
                    You are a calculator.
                    You will be provided a single math problem by the user.
                    Use tools at your disposal to solve it.
                    If you reference the result of a tool call in your answer, always explain it to the user in a clear sentence, e.g. 'The result is 4.'
                    Never assume the user can see the raw tool result.
                    """.trimIndent()
                )
            },
            model = LeapModels.Chat.LFM2_1_2B_Tool,
            maxAgentIterations = 10,
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