package ai.liquid.koogleapsdk.ui.screen.weatherTool

sealed class WeatherToolEvent {
    data class OnSearchClick(val cityName: String) : WeatherToolEvent()
}