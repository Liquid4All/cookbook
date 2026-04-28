package ai.liquid.browsercontrol.domain

object PromptFormatter {
    fun formatUserPrompt(observation: BrowserObservation): String {
        return buildString {
            appendLine("Step ${observation.step + 1}")
            appendLine()
            appendLine("Goal: ${observation.goal}")

            if (observation.error != null) {
                appendLine()
                appendLine("Previous action error: ${observation.error}")
            }

            appendLine()
            appendLine("Page structure:")
            appendLine(observation.axtree)
            appendLine()
            append("What action do you take?")
        }
    }
}
