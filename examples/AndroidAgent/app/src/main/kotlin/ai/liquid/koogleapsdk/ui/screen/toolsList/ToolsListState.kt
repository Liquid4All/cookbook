package ai.liquid.koogleapsdk.ui.screen.toolsList

data class ToolsListState(
    val tools: List<ToolItem> = emptyList(),
) {
    data class ToolItem(
        val id: String,
        val name: String,
        val description: String,
    ) {
        companion object {
            const val TOOL_ID_CALCULATOR = "calculator"
            const val TOOL_ID_WEATHER = "weather"
        }
    }
}