package ai.liquid.koogleapsdk.ui.navigation

sealed interface Destination {
    data object CalculatorTool : Destination

    data object WeatherTool : Destination

    data object ToolsList : Destination
}