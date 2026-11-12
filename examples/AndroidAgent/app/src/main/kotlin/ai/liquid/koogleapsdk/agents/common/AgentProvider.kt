package ai.liquid.koogleapsdk.agents.common

/**
 * Interface for agent factory
 */
interface AgentProvider {
    /**
     * Title for the agent demo screen
     */
    val title: String

    /**
     * Description of the agent
     */
    val description: String

    suspend fun provideAgent(
        onToolCallEvent: suspend (String) -> Unit,
        onErrorEvent: suspend (String) -> Unit,
        onAssistantMessage: suspend (String) -> String
    ): Agent
}
