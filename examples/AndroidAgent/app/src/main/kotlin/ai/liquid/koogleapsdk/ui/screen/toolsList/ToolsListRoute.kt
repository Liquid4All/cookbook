package ai.liquid.koogleapsdk.ui.screen.toolsList

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun ToolsListRoute() {
    val viewModel = viewModel { ToolsListViewModel() }
    val state by viewModel.state.collectAsStateWithLifecycle()

    ToolsListScreen(
        state = state,
        onEvent = viewModel::onEvent
    )
}