package ai.liquid.visiondemo

import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.Image
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.unit.dp
import android.graphics.BitmapFactory
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class VisionEngine {
    companion object { init { System.loadLibrary("vision-demo") } }
    external fun init(nativeDir: String)
    external fun infer(model: String, projector: String, image: String, question: String): String
}

class MainActivity : ComponentActivity() {
    private var modelPath by mutableStateOf("")
    private var projectorPath by mutableStateOf("")
    private var imagePath by mutableStateOf("")
    private var imageBitmap by mutableStateOf<Bitmap?>(null)
    private var status by mutableStateOf("Choose the model, projector, and an image to begin.")
    private var busy by mutableStateOf(false)
    private val engine = VisionEngine()

    private fun import(uri: Uri, filename: String): String {
        val file = File(filesDir, filename)
        contentResolver.openInputStream(uri)!!.use { input -> file.outputStream().use(input::copyTo) }
        return file.absolutePath
    }
    private val modelPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) { modelPath = import(uri, "model.gguf"); status = "Text model ready." }
    }
    private val projectorPicker = registerForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) { projectorPath = import(uri, "projector.gguf"); status = "Vision projector ready." }
    }
    private val imagePicker = registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
        if (uri != null) {
            imagePath = import(uri, "photo.jpg")
            imageBitmap = BitmapFactory.decodeFile(imagePath)
            status = "Image ready. Ask a question about it."
        }
    }
    private val camera = registerForActivityResult(ActivityResultContracts.TakePicturePreview()) { bitmap ->
        if (bitmap != null) {
            val file = File(filesDir, "photo.jpg")
            file.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, 90, it) }
            imagePath = file.absolutePath
            imageBitmap = bitmap
            status = "Photo ready. Ask a question about it."
        }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        engine.init(applicationInfo.nativeLibraryDir)
        setContent {
            val scope = rememberCoroutineScope()
            var question by remember { mutableStateOf("Describe this image.") }
            MaterialTheme {
                Column(Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp), verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    Text("LFM3 Vision", style = MaterialTheme.typography.headlineMedium)
                    Text("Run LFM3-VL-2B locally with llama.cpp")
                    Button(onClick = { modelPicker.launch(arrayOf("*/*")) }, enabled = !busy) { Text(if (modelPath.isEmpty()) "Choose LFM3-VL model" else "Model selected") }
                    Button(onClick = { projectorPicker.launch(arrayOf("*/*")) }, enabled = !busy) { Text(if (projectorPath.isEmpty()) "Choose vision projector" else "Projector selected") }
                    Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                        Button(onClick = { imagePicker.launch("image/*") }, enabled = !busy) { Text("Gallery") }
                        Button(onClick = { camera.launch(null) }, enabled = !busy) { Text("Camera") }
                    }
                    imageBitmap?.let { Image(it.asImageBitmap(), "Selected image", Modifier.fillMaxWidth().height(220.dp)) }
                    OutlinedTextField(value = question, onValueChange = { question = it }, label = { Text("Question") }, modifier = Modifier.fillMaxWidth())
                    Button(onClick = {
                        busy = true; status = "Analyzing image on device…"
                        scope.launch {
                            status = withContext(Dispatchers.Default) { engine.infer(modelPath, projectorPath, imagePath, question) }
                            busy = false
                        }
                    }, enabled = !busy && modelPath.isNotEmpty() && projectorPath.isNotEmpty() && imagePath.isNotEmpty()) { Text("Analyze image") }
                    if (busy) CircularProgressIndicator()
                    Text(status)
                }
            }
        }
    }
}
