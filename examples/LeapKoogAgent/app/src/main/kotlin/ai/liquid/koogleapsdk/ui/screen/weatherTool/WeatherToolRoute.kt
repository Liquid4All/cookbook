package ai.liquid.koogleapsdk.ui.screen.weatherTool

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun WeatherToolRoute() {
    val viewModel = viewModel { WeatherToolViewModel() }
    val state by viewModel.state.collectAsStateWithLifecycle()

    WeatherToolScreen(
        state = state,
        onEvent = viewModel::onEvent
    )
}