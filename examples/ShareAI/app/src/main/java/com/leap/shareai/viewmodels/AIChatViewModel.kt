package com.leap.shareai.viewmodels

import android.app.DownloadManager
import android.content.Context
import android.net.Uri
import android.util.Log
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import com.leap.shareai.model.ChatMessageDisplayItem
import com.leap.shareai.model.Role
import com.leap.shareai.webscraping.WebPageState
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableSharedFlow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharedFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.withContext
import java.io.File


class AIChatViewModel : ViewModel() {

    companion object {
        private const val TAG = "LeapViewModel"
        private const val MODEL_URL = "https://huggingface.co/LiquidAI/LFM2.5-350M-GGUF/resolve/main/LFM2.5-350M-Q8_0.gguf"
        private const val MODEL_FILE = "LFM2.5-350M-Q8_0.gguf"
        private const val PROMPT = "Make a summary less than 500 words of the following text. Use Markdown If needed:\n\n"
    }

    private val _state = mutableStateOf<LeapState>(LeapState.Idle)
    val state: State<LeapState> = _state

    private var modelRunner: InferenceEngine? = null
    private var generationJob: Job? = null

    private val _responseChunks = MutableSharedFlow<String>()
    val responseChunks: SharedFlow<String> = _responseChunks

    private val _reasoningChunks = MutableSharedFlow<String>()
    val reasoningChunks: SharedFlow<String> = _reasoningChunks

    private val _isLoading = MutableSharedFlow<Boolean>()
    val isLoading: SharedFlow<Boolean> = _isLoading

    private val _messages = MutableStateFlow<List<ChatMessageDisplayItem>>(emptyList())
    val messages: StateFlow<List<ChatMessageDisplayItem>> = _messages

    // Temporary holders for current model response
    private var currentResponseText = StringBuilder()
    private var currentReasoningText = StringBuilder()

    fun loadModel(context: Context) {
        _state.value = LeapState.Loading
        viewModelScope.launch {
            _isLoading.emit(true)
            try {
                val modelFile = File(context.getExternalFilesDir(null), MODEL_FILE)

                // Check if model needs to be downloaded
                if (!modelFile.exists()) {
                    // Model needs to be downloaded
                    Log.d(TAG, "Model not found locally, starting download...")

                    // Start the download
                    val downloadManager = context.getSystemService(DownloadManager::class.java)
                    val downloadId = downloadManager.enqueue(
                        DownloadManager.Request(Uri.parse(MODEL_URL))
                            .setTitle("Downloading ShareAI Model")
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
                                    Log.d(TAG, "Download complete!")
                                    downloading = false
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
                                    Log.d(TAG, "Downloading: $percentage% ($downloadedMB MB / $totalMB MB)")
                                }
                            }
                        }
                        if (downloading) delay(500)
                    }
                }

                modelRunner = withContext(Dispatchers.Default) {
                    AiChat.getInferenceEngine(context.applicationContext).also {
                        it.loadModel(modelFile.absolutePath)
                    }
                }
                Log.d(TAG, "Model loaded")
            } catch (e: Exception) {
                _state.value = LeapState.Error("Failed to load model: ${e.message}")
                Log.e(TAG, "Failed to load model: ${e.message}")
            } finally {
                _isLoading.emit(false)
            }
        }
    }

    fun sendMessage(userInput: String) {
        // Add user's message to the list
        Log.d(TAG, "Sending user input: $userInput")
        val massage = PROMPT + userInput.trim()
        val userMessage = ChatMessageDisplayItem(Role.USER, massage)
        _messages.update { it + userMessage }

        generateResponse(massage)
    }


    private fun generateResponse(input: String) {

        val runner = modelRunner ?: run {
            Log.e(TAG, "Model not loaded")
            return
        }

        generationJob?.cancel() // cancel previous if any

        generationJob = viewModelScope.launch {
            _state.value = LeapState.Thinking
            runner.sendUserPrompt(input)
                .onCompletion { e ->
                    Log.d(TAG, "Generation complete")
                    if (e == null) {
                        _state.value = LeapState.Success(currentResponseText.toString())
                        val aiMessage = ChatMessageDisplayItem(
                            role = Role.ASSISTANT,
                            text = currentResponseText.toString(),
                            reasoning = currentReasoningText.takeIf { it.isNotBlank() }
                                ?.toString()
                        )
                        _messages.update { it + aiMessage }
                        Log.d(TAG, "AI message completed and added to list")
                    }
                }
                .catch { e ->
                    Log.e(TAG, "Error generating response: ${e.message}")
                    _state.value = LeapState.Error("Error: ${e.message}")
                }
                .collect { chunk ->
                    _state.value = LeapState.Generating
                    currentResponseText.append(chunk)
                    _responseChunks.emit(chunk)
                }
        }
    }

    fun cancelGeneration() {
        generationJob?.cancel()
        Log.d(TAG, "Generation canceled")
    }

    override fun onCleared() {
        super.onCleared()

        // Unload model asynchronously to avoid ANR
        // Do NOT use runBlocking here - it blocks the main thread
        CoroutineScope(Dispatchers.IO).launch {
            try {
                modelRunner?.cleanUp()
            } catch (e: Exception) {
                Log.e(TAG, "Error unloading model", e)
            }
        }
    }
}

sealed class LeapState {
    object Idle: LeapState()
    object Thinking : LeapState()
    object Generating : LeapState()
    object Loading : LeapState()
    data class Error(val message: String) : LeapState()
    data class Success(val response: String) : LeapState()
}
