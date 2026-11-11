package ai.liquid.androidkoogagent.ui.screen.weatherTool

sealed class WeatherToolEvent {
    data class OnSearchClick(val cityName: String) : WeatherToolEvent()
}