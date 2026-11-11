package ai.liquid.androidkoogagent.ui.screen.toolsList

import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.ui.platform.LocalContext
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import ai.liquid.androidkoogagent.agents.common.modelPath
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel

@Composable
fun ToolsListRoute() {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    var modelStatus by remember { mutableStateOf(if (modelPath.isFile) "Model ready" else "Import the model to begin") }
    val picker = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) scope.launch {
            modelStatus = "Copying model to app storage…"
            try {
                withContext(Dispatchers.IO) {
                    context.contentResolver.openInputStream(uri).use { input ->
                        requireNotNull(input) { "Could not read selected file" }
                        modelPath.parentFile?.mkdirs()
                        modelPath.outputStream().use { output -> input.copyTo(output) }
                    }
                }
                modelStatus = "Model ready"
            } catch (error: Exception) {
                modelStatus = "Import failed: ${error.message}"
            }
        }
    }
    val viewModel = viewModel { ToolsListViewModel() }
    val state by viewModel.state.collectAsStateWithLifecycle()

    ToolsListScreen(
        state = state,
        onEvent = viewModel::onEvent,
        modelStatus = modelStatus,
        onImportModel = { picker.launch(arrayOf("application/octet-stream", "*/*")) },
    )
}
