package ai.liquid.koogleapsdk.ui.screen.toolsList

import ai.liquid.koogleapsdk.ui.common.MviViewModel
import ai.liquid.koogleapsdk.ui.navigation.Destination
import ai.liquid.koogleapsdk.ui.navigation.NavigationService
import ai.liquid.koogleapsdk.ui.screen.toolsList.util.allTools
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

class ToolsListViewModel(
    private val navigationService: NavigationService = NavigationService.Companion.Instance
) : MviViewModel<ToolsListState, ToolsListEvent>() {
    private val _state = MutableStateFlow(
        ToolsListState(
            tools = allTools
        )
    )
    override val state: StateFlow<ToolsListState> = _state.asStateFlow()

    override fun onEvent(event: ToolsListEvent) {
        when (event) {
            is ToolsListEvent.OnToolClick -> {
                navigationService.navigateTo(resolveDestination(event.toolId))
            }
        }
    }

    private fun resolveDestination(toolId: String): Destination {
        return when (toolId) {
            ToolsListState.ToolItem.Companion.TOOL_ID_CALCULATOR -> Destination.CalculatorTool
            ToolsListState.ToolItem.Companion.TOOL_ID_WEATHER -> Destination.WeatherTool
            else -> throw IllegalArgumentException("Unknown toolId: $toolId")
        }
    }
}