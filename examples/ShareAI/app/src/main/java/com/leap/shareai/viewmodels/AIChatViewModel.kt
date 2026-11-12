package com.leap.shareai.viewmodels

import ai.liquid.leap.Conversation
import ai.liquid.leap.LeapClient
import ai.liquid.leap.LeapModelLoadingException
import ai.liquid.leap.ModelRunner
import ai.liquid.leap.downloader.LeapModelDownloader
import ai.liquid.leap.downloader.LeapModelDownloaderNotificationConfig
import ai.liquid.leap.message.ChatMessage
import ai.liquid.leap.message.MessageResponse
import android.content.Context
import android.util.Log
import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.leap.shareai.model.ChatMessageDisplayItem
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
import kotlinx.coroutines.flow.takeWhile
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers


class AIChatViewModel : ViewModel() {

    companion object {
        private const val TAG = "LeapViewModel"
        private const val MODEL_NAME = "LFM2-350M"
        private const val QUANTIZATION_SLUG = "Q8_0"
        private const val PROMPT = "Make a summary less than 500 words of the following text. Use Markdown If needed:\n\n"
    }

    private val _state = mutableStateOf<LeapState>(LeapState.Idle)
    val state: State<LeapState> = _state

    private var modelRunner: ModelRunner? = null
    private var conversation: Conversation? = null
    private var generationJob: Job? = null
    private var downloader: LeapModelDownloader? = null

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
                // Reuse downloader instance to avoid concurrent download issues
                if (downloader == null) {
                    downloader = LeapModelDownloader(
                        context,
                        notificationConfig = LeapModelDownloaderNotificationConfig.build {
                            notificationTitleDownloading = "Downloading ShareAI Model"
                            notificationTitleDownloaded = "Model Ready!"
                        }
                    )
                }
                val downloaderInstance = downloader!!

                // Check if model needs to be downloaded
                val currentStatus = downloaderInstance.queryStatus(MODEL_NAME, QUANTIZATION_SLUG)

                if (currentStatus is LeapModelDownloader.ModelDownloadStatus.NotOnLocal) {
                    // Model needs to be downloaded
                    Log.d(TAG, "Model not found locally, starting download...")

                    // Observe download progress
                    val progressFlow = downloaderInstance.observeDownloadProgress(MODEL_NAME, QUANTIZATION_SLUG)

                    // Start the download
                    downloaderInstance.requestDownloadModel(MODEL_NAME, QUANTIZATION_SLUG)

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
                                Log.d(TAG, "Downloading: $percentage% ($downloadedMB MB / $totalMB MB)")
                            } else {
                                val downloadStatus = downloaderInstance.queryStatus(MODEL_NAME, QUANTIZATION_SLUG)
                                if (downloadStatus is LeapModelDownloader.ModelDownloadStatus.Downloaded) {
                                    Log.d(TAG, "Download complete!")
                                }
                            }
                        }
                        .takeWhile { progress ->
                            progress != null || downloaderInstance.queryStatus(MODEL_NAME, QUANTIZATION_SLUG) is LeapModelDownloader.ModelDownloadStatus.DownloadInProgress
                        }
                        .collect()
                }

                modelRunner = downloaderInstance.loadModel(
                    modelName = MODEL_NAME,
                    quantizationType = QUANTIZATION_SLUG,
                )
                conversation = modelRunner?.createConversation()
                Log.d(TAG, "Model loaded and conversation created")
            } catch (e: LeapModelLoadingException) {
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
        val userMessage = ChatMessageDisplayItem(ChatMessage.Role.USER, massage)
        _messages.update { it + userMessage }

        generateResponse(massage)
    }


    private fun generateResponse(input: String) {

        val conv = conversation ?: run {
            Log.e(TAG, "Conversation not initialized")
            return
        }

        generationJob?.cancel() // cancel previous if any

        generationJob = viewModelScope.launch {
            _state.value = LeapState.Thinking
            conv.generateResponse(input)
                .onCompletion {
                    Log.d(TAG, "Generation complete")
                }
                .catch { e ->
                    Log.e(TAG, "Error generating response: ${e.message}")
                    _state.value = LeapState.Error("Error: ${e.message}")
                }
                .collect { response ->
                    when (response) {
                        is MessageResponse.Chunk -> {
                            _state.value = LeapState.Generating
                            currentResponseText.append(response.text)
                            _responseChunks.emit(response.text)
                        }

                        is MessageResponse.ReasoningChunk -> {
                            _state.value = LeapState.Thinking
                            currentReasoningText.append(response.reasoning)
                            _reasoningChunks.emit(response.reasoning)
                        }

                        is MessageResponse.Complete -> {
                            _state.value = LeapState.Success(currentResponseText.toString())
                            val aiMessage = ChatMessageDisplayItem(
                                role = ChatMessage.Role.ASSISTANT,
                                text = currentResponseText.toString(),
                                reasoning = currentReasoningText.takeIf { it.isNotBlank() }
                                    ?.toString()
                            )
                            _messages.update { it + aiMessage }
                            Log.d(TAG, "AI message completed and added to list")
                        }

                        else -> Unit
                    }
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
                modelRunner?.unload()
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
