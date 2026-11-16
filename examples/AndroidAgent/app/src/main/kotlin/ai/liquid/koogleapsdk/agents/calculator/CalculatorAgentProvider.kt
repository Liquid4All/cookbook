package ai.liquid.koogleapsdk.agents.calculator

import ai.liquid.koogleapsdk.agents.common.Agent
import ai.liquid.koogleapsdk.agents.common.AgentProvider
import ai.liquid.koogleapsdk.agents.common.LocalModel

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
    ): Agent {
        // Return the agent with calculator tools
        return Agent(
            llm = LocalModel.get(),
            systemPrompt = """
                You are a calculator.
                You will be provided a single math problem by the user.
                Use tools at your disposal to solve it.
                If you reference the result of a tool call in your answer, always explain it to the user in a clear sentence, e.g. 'The result is 4.'
                Never assume the user can see the raw tool result.
                """.trimIndent(),
            tools = listOf(
                CalculatorTools.PlusTool,
                CalculatorTools.MinusTool,
                CalculatorTools.DivideTool,
                CalculatorTools.MultiplyTool,
            ),
            maxIterations = 10,
            onToolCallEvent = onToolCallEvent,
            onErrorEvent = onErrorEvent,
            onAssistantMessage = onAssistantMessage,
        )
    }
}
