package ai.liquid.koogleapsdk.ui.screen.calculatorTool

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun CalculatorToolRoute() {
    val viewModel = viewModel { CalculatorToolViewModel() }
    val state by viewModel.state.collectAsStateWithLifecycle()

    CalculatorToolScreen(
        state = state,
        onEvent = viewModel::onEvent
    )
}