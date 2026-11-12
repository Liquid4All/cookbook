package ai.liquid.koogleapsdk.ui.screen.weatherTool

import ai.liquid.koogleapsdk.agents.weather.WeatherAgentProvider
import ai.liquid.koogleapsdk.ui.common.MviViewModel
import ai.liquid.koogleapsdk.ui.util.SnackbarUtil
import android.util.Log
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

class WeatherToolViewModel : MviViewModel<WeatherToolState, WeatherToolEvent>() {
    private val _state = MutableStateFlow(WeatherToolState())
    override val state: StateFlow<WeatherToolState> = _state.asStateFlow()

    override fun onEvent(event: WeatherToolEvent) {
        when (event) {
            is WeatherToolEvent.OnSearchClick -> {
                fetchWeather(event.cityName)
            }
        }
    }

    internal fun fetchWeather(cityName: String) {
        // Simulate fetching weather data
        _state.update {
            it.copy(isLoading = true, weatherInfo = null)
        }

        viewModelScope.launch {
            val prompt = "What is the weather in: $cityName"
            try {
                val agent = WeatherAgentProvider().provideAgent(
                    onToolCallEvent = {
                        Log.w(this@WeatherToolViewModel::class.simpleName, "Tool call: $it")
                    },
                    onErrorEvent = {
                        SnackbarUtil.showSnackbar(
                            "Error occurred: $it",
                        )
                    },
                    onAssistantMessage = {
                        Log.w(this@WeatherToolViewModel::class.simpleName, "Assistant: $it")
                        val explained = it
                        _state.update { state ->
                            state.copy(
                                weatherInfo = explained,
                            )
                        }
                        explained
                    }
                )
                val result = agent.run(prompt)

                _state.update {
                    it.copy(
                        isLoading = false,
                        weatherInfo = result,
                    )
                }
            } catch (ex: Exception) {
                // TODO handle errors
            }
        }
    }
}