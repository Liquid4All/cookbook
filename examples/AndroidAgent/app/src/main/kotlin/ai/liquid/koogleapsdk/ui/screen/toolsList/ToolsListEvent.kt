package ai.liquid.koogleapsdk.ui.screen.toolsList

sealed class ToolsListEvent {
    data class OnToolClick(val toolId: String) : ToolsListEvent()
}