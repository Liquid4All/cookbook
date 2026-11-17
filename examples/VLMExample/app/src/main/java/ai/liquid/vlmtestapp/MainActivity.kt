package ai.liquid.vlmtestapp

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.activity.viewModels
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import coil3.compose.AsyncImage
import java.io.File

class MainActivity : ComponentActivity() {
    private val viewModel: VlmViewModel by viewModels()
    private var pendingPhoto: String? = null
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        pendingPhoto = savedInstanceState?.getString("pendingPhoto")
        val camera = registerForActivityResult(ActivityResultContracts.TakePicture()) { taken ->
            viewModel.capturing(false)
            val image = pendingPhoto?.let { File(it) }
            if (taken && image != null) viewModel.describe(image)
            pendingPhoto = null
        }
        enableEdgeToEdge()
        setContent {
            val state by viewModel.state.collectAsState()
            Scaffold(modifier = Modifier.fillMaxSize()) { padding ->
                Column(Modifier.padding(padding).padding(12.dp).verticalScroll(rememberScrollState())) {
                    Button(onClick = {
                        val image = File(externalCacheDir, "image-${System.currentTimeMillis()}.jpg")
                        pendingPhoto = image.absolutePath
                        viewModel.capturing(true)
                        camera.launch(FileProvider.getUriForFile(this@MainActivity, "$packageName.provider", image))
                    }, enabled = !state.busy && !state.capturing) { Text("Take a picture") }
                    state.image?.let { image ->
                        AsyncImage(model = image, contentDescription = "Captured image", modifier = Modifier.heightIn(100.dp, 200.dp))
                    }
                    Text(state.error ?: state.text)
                    if (state.error != null && state.image != null) {
                        Button(onClick = { viewModel.describe(checkNotNull(state.image)) }, enabled = !state.busy) { Text("Retry") }
                    }
                }
            }
        }
    }
    override fun onSaveInstanceState(outState: Bundle) {
        outState.putString("pendingPhoto", pendingPhoto)
        super.onSaveInstanceState(outState)
    }
}
