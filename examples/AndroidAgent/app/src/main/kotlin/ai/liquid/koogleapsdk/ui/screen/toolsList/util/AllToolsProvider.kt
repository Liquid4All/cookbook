package ai.liquid.koogleapsdk.ui.screen.toolsList.util

import ai.liquid.koogleapsdk.App
import ai.liquid.koogleapsdk.ui.screen.toolsList.ToolsListState
import ai.liquid.koogleapsdk.R
import ai.liquid.koogleapsdk.ui.screen.toolsList.ToolsListState.ToolItem.Companion.TOOL_ID_CALCULATOR
import ai.liquid.koogleapsdk.ui.screen.toolsList.ToolsListState.ToolItem.Companion.TOOL_ID_WEATHER

val allTools by lazy {
    listOf(
        ToolsListState.ToolItem(
            id = TOOL_ID_CALCULATOR,
            name = App.Companion.context.getString(R.string.calculator),
            description = App.Companion.context.getString(R.string.a_simple_calculator_tool),
        ),
        ToolsListState.ToolItem(
            id = TOOL_ID_WEATHER,
            name = App.Companion.context.getString(R.string.weather),
            description = App.Companion.context.getString(R.string.get_current_weather_information),
        ),
    )
}
