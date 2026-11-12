package ai.liquid.recipegenerator

import ai.liquid.leap.LeapClient
import ai.liquid.leap.LeapJson
import ai.liquid.leap.ModelRunner
import ai.liquid.leap.GenerationOptions
import ai.liquid.leap.downloader.LeapModelDownloader
import ai.liquid.leap.downloader.LeapModelDownloaderNotificationConfig
import ai.liquid.leap.message.MessageResponse
import android.content.Context
import android.util.Log
import ai.liquid.leap.structuredoutput.Generatable
import ai.liquid.leap.structuredoutput.GeneratableFactory
import ai.liquid.leap.structuredoutput.Guide
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.takeWhile
import kotlinx.coroutines.launch
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.jsonObject


@Serializable
@Generatable("A recipe for cooking great dishes")
data class Recipe(
    val name: String,

    @Guide("Ingredients for making the dishes")
    val ingredients: List<String>,

    @Guide("Cooking time in minutes")
    val cookingTime: Int,

    @Guide("Whether the meal is vegetarian")
    val isVegetarian: Boolean,

    @Guide("Steps of cooking")
    val steps: List<String>,

)

class MainActivityViewModel: ViewModel() {
    private var modelRunner: ModelRunner? = null
    var recipeState: Recipe? by mutableStateOf(null)
    var status: String by mutableStateOf("Not ready to generate")

    private val modelName = "LFM2-700M"
    private val quantType = "Q8_0"
    private var downloader: LeapModelDownloader? = null

    private fun getDownloader(context: Context): LeapModelDownloader {
        if (downloader == null) {
            downloader = LeapModelDownloader(
                context,
                notificationConfig = LeapModelDownloaderNotificationConfig.build {
                    notificationTitleDownloading = "Downloading Recipe Generator Model"
                    notificationTitleDownloaded = "Model Ready!"
                }
            )
        }
        return downloader!!
    }

    suspend fun loadModel(context: Context) {
        val downloader = getDownloader(context)

        status = "Checking model status..."
        val currentStatus = downloader.queryStatus(modelName, quantType)

        if (currentStatus is LeapModelDownloader.ModelDownloadStatus.NotOnLocal) {
            // Model needs to be downloaded
            status = "Starting download..."

            // Observe download progress
            val progressFlow = downloader.observeDownloadProgress(modelName, quantType)

            // Start the download
            downloader.requestDownloadModel(modelName, quantType)

            // Collect progress updates until download completes
            progressFlow
                .onEach { progress ->
                    if (progress != null) {
                        val downloadedMB = progress.downloadedSizeInBytes / (1024 * 1024)
                        val totalMB = progress.totalSizeInBytes / (1024 * 1024)
                        val percentage = if (progress.totalSizeInBytes > 0) {
                            (progress.downloadedSizeInBytes * 100.0 / progress.totalSizeInBytes).toInt()
                        } else {
                            0
                        }
                        status = "Downloading: $percentage% ($downloadedMB MB / $totalMB MB)"
                    } else {
                        val downloadStatus = downloader.queryStatus(modelName, quantType)
                        if (downloadStatus is LeapModelDownloader.ModelDownloadStatus.Downloaded) {
                            status = "Download complete!"
                        }
                    }
                }
                .takeWhile { progress ->
                    progress != null || downloader.queryStatus(modelName, quantType) is LeapModelDownloader.ModelDownloadStatus.DownloadInProgress
                }
                .collect()
        }

        // Load the model
        status = "Loading model..."
        try {
            modelRunner = downloader.loadModel(modelName, quantType)
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
            // Simple system prompt - the schema will be automatically injected by the SDK
            val conversation = modelRunner.createConversation(
                "You are a recipe generator bot that reads the user's message and generates JSON output."
            )
            val options = GenerationOptions()
            options.setResponseFormatType<Recipe>()
            Log.d(MainActivityViewModel::class.simpleName, "Generating recipe with automatic schema injection. Constraint: ${options.jsonSchemaConstraint}")
            val buffer = StringBuilder()
            try {
                conversation.generateResponse("A recipe for a dinner dish", options)
                    .onEach { response ->
                        when (response) {
                            is MessageResponse.Chunk -> {
                                buffer.append(response.text)
                            }

                            else -> {
                                // ignore
                            }
                        }
                    }.collect()

                val generatedText = buffer.toString()
                Log.d("RecipeGenerator", "Generated text: $generatedText")

                val recipe = GeneratableFactory.createFromJsonObject<Recipe>(
                    LeapJson.parseToJsonElement(generatedText).jsonObject
                )
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