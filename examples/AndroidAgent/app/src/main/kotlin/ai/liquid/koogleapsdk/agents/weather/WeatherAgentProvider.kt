package ai.liquid.koogleapsdk.agents.weather

import ai.liquid.koogleapsdk.agents.common.Agent
import ai.liquid.koogleapsdk.agents.common.AgentProvider
import ai.liquid.koogleapsdk.agents.common.LocalModel
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
    ): Agent {
        // Return the agent with weather tools
        return Agent(
            llm = LocalModel.get(),
            systemPrompt = """
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
                """.trimIndent(),
            tools = listOf(
                WeatherTools.CurrentDatetimeTool,
                WeatherTools.AddDatetimeTool,
                WeatherTools.WeatherForecastTool,
            ),
            maxIterations = 50,
            onToolCallEvent = onToolCallEvent,
            onErrorEvent = onErrorEvent,
            onAssistantMessage = onAssistantMessage,
        )
    }
}
