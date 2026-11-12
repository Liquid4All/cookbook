package ai.liquid.recipegenerator

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File


@Serializable
data class Recipe(
    val name: String,

    val ingredients: List<String>,

    val cookingTime: Int,

    val isVegetarian: Boolean,

    val steps: List<String>,

)

// JSON schema for Recipe. llama.cpp turns it into a grammar, so the model can only generate matching JSON.
private const val RECIPE_SCHEMA = """
{
  "type": "object",
  "description": "A recipe for cooking great dishes",
  "properties": {
    "name": { "type": "string" },
    "ingredients": { "type": "array", "items": { "type": "string" }, "description": "Ingredients for making the dishes" },
    "cookingTime": { "type": "integer", "description": "Cooking time in minutes" },
    "isVegetarian": { "type": "boolean", "description": "Whether the meal is vegetarian" },
    "steps": { "type": "array", "items": { "type": "string" }, "description": "Steps of cooking" }
  },
  "required": ["name", "ingredients", "cookingTime", "isVegetarian", "steps"]
}
"""

class MainActivityViewModel: ViewModel() {
    private var modelRunner: LlamaJson? = null
    var recipeState: Recipe? by mutableStateOf(null)
    var status: String by mutableStateOf("Not ready to generate")

    private val modelUrl = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q5_K_M.gguf"
    private val modelFileName = "LFM2.5-1.2B-Instruct-Q5_K_M.gguf"

    suspend fun loadModel(context: Context) {
        val modelFile = File(context.getExternalFilesDir(null), modelFileName)

        status = "Checking model status..."

        if (!modelFile.exists()) {
            // Model needs to be downloaded
            status = "Starting download..."

            // Start the download
            val downloadManager = context.getSystemService(DownloadManager::class.java)
            val downloadId = downloadManager.enqueue(
                DownloadManager.Request(Uri.parse(modelUrl))
                    .setTitle("Downloading Recipe Generator Model")
                    .setDestinationUri(Uri.fromFile(modelFile))
            )

            // Poll progress updates until download completes
            var downloading = true
            while (downloading) {
                downloadManager.query(DownloadManager.Query().setFilterById(downloadId)).use { cursor ->
                    if (!cursor.moveToFirst()) return@use
                    val downloadedSizeInBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR))
                    val totalSizeInBytes = cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_TOTAL_SIZE_BYTES))
                    when (cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))) {
                        DownloadManager.STATUS_SUCCESSFUL -> {
                            status = "Download complete!"
                            downloading = false
                        }
                        DownloadManager.STATUS_FAILED -> {
                            status = "Error: model download failed"
                            return
                        }
                        else -> {
                            val downloadedMB = downloadedSizeInBytes / (1024 * 1024)
                            val totalMB = totalSizeInBytes / (1024 * 1024)
                            val percentage = if (totalSizeInBytes > 0) {
                                (downloadedSizeInBytes * 100.0 / totalSizeInBytes).toInt()
                            } else {
                                0
                            }
                            status = "Downloading: $percentage% ($downloadedMB MB / $totalMB MB)"
                        }
                    }
                }
                if (downloading) delay(500)
            }
        }

        // Load the model
        status = "Loading model..."
        try {
            modelRunner = withContext(Dispatchers.Default) {
                LlamaJson(context.applicationInfo.nativeLibraryDir).also { it.load(modelFile.absolutePath) }
            }
            status = "Model loaded"
        } catch (e: Exception) {
            Log.e("RecipeGenerator", "Error loading model", e)
            status = "Error: ${e.message}"
        }
    }

    fun generateRecipe(context: Context) {
        viewModelScope.launch {
            if (modelRunner == null) {
                loadModel(context)
            }
            val modelRunner = checkNotNull(modelRunner)
            status = "Generating the recipe..."
            // The schema goes into the system prompt and constrains generation as a grammar
            val systemPrompt =
                "You are a recipe generator bot that reads the user's message and generates JSON output " +
                    "that follows this JSON schema: $RECIPE_SCHEMA"
            Log.d(MainActivityViewModel::class.simpleName, "Generating recipe with JSON schema constraint: $RECIPE_SCHEMA")
            val buffer = StringBuilder()
            try {
                modelRunner.generate(systemPrompt, "A recipe for a dinner dish", RECIPE_SCHEMA)
                    .onEach { response ->
                        buffer.append(response)
                    }.collect()

                val generatedText = buffer.toString()
                Log.d("RecipeGenerator", "Generated text: $generatedText")

                val recipe = Json.decodeFromString<Recipe>(generatedText)
                Log.d("RecipeGenerator", "Successfully parsed recipe: ${recipe.name}")
                recipeState = recipe
                status = "Recipe generated successfully!"
            } catch (e: Exception) {
                Log.e("RecipeGenerator", "Error generating recipe", e)
                status = "Error: ${e.message}\n\nGenerated text:\n${buffer.toString()}"
            }
        }
    }

    override fun onCleared() {
        super.onCleared()

        // Unload model asynchronously to avoid ANR
        // Do NOT use runBlocking here - it blocks the main thread
        CoroutineScope(Dispatchers.IO).launch {
            try {
                modelRunner?.unload()
            } catch (e: Exception) {
                Log.e("RecipeGenerator", "Error unloading model", e)
            }
        }
    }
}