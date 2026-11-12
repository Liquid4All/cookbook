package ai.liquid.vlmtestapp

import android.app.DownloadManager
import android.net.Uri
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.ui.Modifier
import android.util.Log
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.heightIn
import androidx.compose.material3.Button
import androidx.compose.runtime.getValue
import androidx.compose.runtime.livedata.observeAsState
import androidx.compose.ui.Alignment
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import androidx.lifecycle.MutableLiveData
import androidx.lifecycle.lifecycleScope
import coil3.compose.AsyncImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

class MainActivity : ComponentActivity() {
    private val generateText: MutableLiveData<String> = MutableLiveData<String>()
    private val isPhotoTaken: MutableLiveData<Boolean> = MutableLiveData<Boolean>(false)
    private var photoCount = 0
    private val imageFileLiveData: MutableLiveData<File> = MutableLiveData<File>()
    private val isTakePictureButtonEnabled: MutableLiveData<Boolean> =
        MutableLiveData<Boolean>(true)
    private lateinit var modelRunner: LlamaVlm

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        val launchCamera =
            registerForActivityResult(ActivityResultContracts.TakePicture()) { result ->
                if (result) {
                    isPhotoTaken.value = true
                    val file = File(
                        this@MainActivity.externalCacheDir,
                        "image_to_process_$photoCount.jpg"
                    )
                    imageFileLiveData.value = file
                    lifecycleScope.launch {
                        generateWithImage(file.inputStream().readBytes())
                        isTakePictureButtonEnabled.value = true
                    }
                } else {
                    isTakePictureButtonEnabled.value = true
                }
            }
        enableEdgeToEdge()
        setContent {
            Scaffold(modifier = Modifier.fillMaxSize()) { innerPadding ->
                val generateTextState by generateText.observeAsState()
                val isPhotoTakenState by isPhotoTaken.observeAsState(false)
                val imageFileCache by imageFileLiveData.observeAsState()
                val isTakePictureButtonEnabledState by isTakePictureButtonEnabled.observeAsState(
                    false
                )
                Box(modifier = Modifier.padding(innerPadding)) {
                    Column(
                        modifier = Modifier
                            .padding(4.dp)
                            .fillMaxSize(), horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Button(onClick = {
                            generateText.value = ""
                            photoCount += 1
                            val newImageFile = File(
                                this@MainActivity.externalCacheDir,
                                "image_to_process_$photoCount.jpg"
                            )
                            val imageUri = FileProvider.getUriForFile(
                                this@MainActivity,
                                applicationContext.packageName + ".provider",
                                newImageFile
                            );
                            isTakePictureButtonEnabled.value = false
                            launchCamera.launch(imageUri)
                        }, enabled = isTakePictureButtonEnabledState) {
                            Text("Take a picture")
                        }
                        if (isPhotoTakenState) {
                            AsyncImage(
                                model = imageFileCache,
                                contentDescription = "Translated description of what the image contains",
                                modifier = Modifier.heightIn(100.dp, 200.dp)
                            )
                        }

                        Text(generateTextState ?: "", modifier = Modifier)
                    }
                }

            }
        }
    }

    suspend fun generateWithImage(imageData: ByteArray) {
        if (!this::modelRunner.isInitialized) {
            val repoUrl = "https://huggingface.co/LiquidAI/LFM2.5-VL-1.6B-GGUF/resolve/main"
            val modelFile = File(getExternalFilesDir(null), "LFM2.5-VL-1.6B-Q8_0.gguf")
            val mmprojFile = File(getExternalFilesDir(null), "mmproj-LFM2.5-VL-1.6b-Q8_0.gguf")

            generateText.value = "Loading the model..."

            // Download the model and its vision projector if they are not on the device yet
            downloadIfMissing("$repoUrl/${modelFile.name}", modelFile)
            downloadIfMissing("$repoUrl/${mmprojFile.name}", mmprojFile)

            modelRunner = withContext(Dispatchers.Default) {
                LlamaVlm(applicationInfo.nativeLibraryDir).also {
                    it.load(modelFile.absolutePath, mmprojFile.absolutePath)
                }
            }
        }
        generateText.value = "Looking at the image and generating a description..."

        var isGenerationStarted = false
        modelRunner.generate(
            systemPrompt = "You are a helpful multimodal assistant by Liquid AI.",
            prompt = "Describe this image.",
            image = imageData,
        )
            .onEach {
                if (!isGenerationStarted) {
                    isGenerationStarted = true
                    generateText.value = ""
                }
                generateText.value = generateText.value + it
            }.onCompletion {
                Log.d("MainActivity", generateText.value.toString())
            }.collect()
    }

    private suspend fun downloadIfMissing(url: String, file: File) {
        if (file.exists()) return
        // Model needs to be downloaded
        generateText.value = "Starting download..."

        // Start the download
        val downloadManager = getSystemService(DownloadManager::class.java)
        val downloadId = downloadManager.enqueue(
            DownloadManager.Request(Uri.parse(url))
                .setTitle("Downloading VLM Model")
                .setDestinationUri(Uri.fromFile(file))
        )

        // Poll progress updates until download completes
        while (true) {
            downloadManager.query(DownloadManager.Query().setFilterById(downloadId)).use { cursor ->
                if (cursor.moveToFirst()) {
                    val downloadedSizeInBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                    val totalSizeInBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                    when (cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))) {
                        DownloadManager.STATUS_SUCCESSFUL -> {
                            generateText.value = "Download complete!"
                            return
                        }
                        DownloadManager.STATUS_FAILED -> throw IllegalStateException("Model download failed")
                        else -> {
                            val downloadedMB = downloadedSizeInBytes / (1024 * 1024)
                            val totalMB = totalSizeInBytes / (1024 * 1024)
                            val percentage = if (totalSizeInBytes > 0) {
                                (downloadedSizeInBytes * 100.0 / totalSizeInBytes).toInt()
                            } else {
                                0
                            }
                            generateText.value = "Downloading: $percentage% ($downloadedMB MB / $totalMB MB)"
                        }
                    }
                }
            }
            delay(500)
        }
    }
}
