package ai.liquid.sloganapp

import ai.liquid.leap.LeapClient
import ai.liquid.leap.ModelRunner
import ai.liquid.leap.downloader.LeapModelDownloader
import ai.liquid.leap.downloader.LeapModelDownloaderNotificationConfig
import ai.liquid.leap.message.MessageResponse
import android.graphics.Color
import android.os.Bundle
import android.view.ViewGroup
import android.widget.Button
import android.widget.EditText
import android.widget.LinearLayout
import android.widget.ScrollView
import android.widget.TextView
import androidx.activity.ComponentActivity
import androidx.core.view.ViewCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updateLayoutParams
import androidx.lifecycle.lifecycleScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.flow.takeWhile
import kotlinx.coroutines.launch

// System prompt constants
private const val SYSTEM_PROMPT = "You are a marketing expert. Suggest engaging and creative slogans based on a business description provided by the user."
// User prompt template.
private const val USER_PROMPT_TEMPLATE = "/no_think Suggest slogans for this business: %s"


class MainActivity : ComponentActivity() {
    private lateinit var modelRunner: ModelRunner
    private var modelLoaded = false
    private var generationJob: Job? = null
    private var downloader: LeapModelDownloader? = null

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.main_activity_layout)
        configMainLayout()
    }

    /**
     * The method to initialize the model.
     *
     * @param textView the main TextView to show generated contents
     * @param modelStatus model status indicator TextView
     * @return return whether the model is successfully loaded
     */
    private suspend fun loadModel(textView: TextView, modelStatus: TextView): Boolean {
        try {
            textView.text = getText(R.string.model_loading_in_progress_label)
            modelStatus.visibility = android.view.View.VISIBLE
            modelStatus.text = resources.getText(R.string.model_status_loading)

            modelStatus.setTextColor(resources.getColor(R.color.model_loading_status_color, null))

            val modelName = "Qwen3-0.6B"
            val quantType = "Q8_0"

            // Reuse downloader instance to avoid concurrent download issues
            if (downloader == null) {
                downloader = LeapModelDownloader(
                    this@MainActivity,
                    notificationConfig = LeapModelDownloaderNotificationConfig.build {
                        notificationTitleDownloading = "Downloading Slogan Generator Model"
                        notificationTitleDownloaded = "Model Ready!"
                    }
                )
            }
            val downloaderInstance = downloader!!

            // Check if model needs to be downloaded
            val currentStatus = downloaderInstance.queryStatus(modelName, quantType)

            if (currentStatus is LeapModelDownloader.ModelDownloadStatus.NotOnLocal) {
                // Model needs to be downloaded
                textView.text = "Starting download..."

                // Observe download progress
                val progressFlow = downloaderInstance.observeDownloadProgress(modelName, quantType)

                // Start the download
                downloaderInstance.requestDownloadModel(modelName, quantType)

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
                            textView.text = "Downloading: $percentage% ($downloadedMB MB / $totalMB MB)"
                        } else {
                            val downloadStatus = downloaderInstance.queryStatus(modelName, quantType)
                            if (downloadStatus is LeapModelDownloader.ModelDownloadStatus.Downloaded) {
                                textView.text = "Download complete!"
                            }
                        }
                    }
                    .takeWhile { progress ->
                        progress != null || downloaderInstance.queryStatus(modelName, quantType) is LeapModelDownloader.ModelDownloadStatus.DownloadInProgress
                    }
                    .collect()
            }

            modelRunner = downloaderInstance.loadModel(modelName, quantType)
            modelLoaded = true

            modelStatus.text = resources.getText(R.string.model_status_loaded)
            modelStatus.setTextColor(Color.GREEN)
            textView.text = getText(R.string.model_loaded_label)
            return true
        } catch (e: Exception) {
            textView.text =
                getString(R.string.error_loading_ai_model_oops_an_error_happened, e.message)
            modelStatus.text = getText(R.string.model_status_error)
            modelStatus.setTextColor(Color.RED)
        }
        return false
    }

    /**
     * Invoke the model to generate a translation from the user provided input.
     *
     * @param textView the main TextView to show generated contents
     * @param userInput the user input
     * @param generateButton the button for invoking generation. This button will be changed into "Stop" button when the generation starts.
     * @param onComplete the callback to invoke when the generation is completed
     */
    private fun generateContent(
        textView: TextView,
        userInput: String,
        generateButton: Button,
        onComplete: () -> Unit
    ) {
        generationJob = lifecycleScope.launch {
            textView.text = getText(R.string.generation_in_progress_label)
            val conversation = modelRunner.createConversation(SYSTEM_PROMPT)

            conversation.generateResponse(USER_PROMPT_TEMPLATE.format(userInput)).onEach { chunk ->
                when (chunk) {
                    is MessageResponse.Chunk -> {
                        if (textView.text.startsWith(getText(R.string.generation_in_progress_label))) {
                            textView.text = chunk.text
                        } else {
                            textView.append(chunk.text)
                        }
                    }
                    else -> {}
                }
                (textView.parent as ScrollView).post {
                    (textView.parent as ScrollView).fullScroll(ScrollView.FOCUS_DOWN)
                }
            }.onCompletion {
                generateButton.text = getText(R.string.generate_button_label)
                generationJob = null
                onComplete()
            }.catch { e ->
                textView.text =
                    "Error in translation: ${e.message} Try again? 🎨"
                generateButton.text = getText(R.string.generate_button_label)
                generationJob = null
            }
                .collect()
        }
    }


    /**
     * Create the main UI layout.
     */
    private fun configMainLayout() {
        val modelStatus = findViewById<TextView>(R.id.model_status)
        val userInput = findViewById<EditText>(R.id.user_input)
        val button = findViewById<Button>(R.id.generate_button)
        val textView = findViewById<TextView>(R.id.generated_text)
        val scrollView = findViewById<ScrollView>(R.id.generated_text_scrollview)

        button.setOnClickListener {
            if (generationJob?.isActive == true) {
                generationJob?.cancel()
                generationJob = null
                button.text = getText(R.string.generate_button_label)

                val currentText = textView.text.toString()
                if (currentText.startsWith("Generating...") &&
                    currentText.trim() == "Generating..."
                ) {
                    textView.text = getText(R.string.notice_if_generation_stopped)
                } else {
                    textView.append("\n\n[Generation stopped...]")
                }

                scrollView.post {
                    scrollView.fullScroll(ScrollView.FOCUS_DOWN)
                }
            } else {
                val input = userInput.text.toString().trim()
                if (input.isNotEmpty()) {
                    button.text = getText(R.string.stop_button_label)
                    if (!modelLoaded) {
                        lifecycleScope.launch {
                            if (loadModel(textView, modelStatus)) {
                                generateContent(textView, input, button) {}
                            }
                        }

                    } else {
                        generateContent(textView, input, button) {}
                    }
                } else {
                    textView.text = getText(R.string.notice_if_no_input_provided)
                }
            }
        }

        val mainLayout = findViewById<LinearLayout>(R.id.main_activity_layout)
        ViewCompat.setOnApplyWindowInsetsListener(mainLayout) { v, windowInsets ->
            val insets = windowInsets.getInsets(WindowInsetsCompat.Type.systemBars())
            v.updateLayoutParams<ViewGroup.MarginLayoutParams> {
                leftMargin = insets.left
                rightMargin = insets.right
                topMargin = insets.top
                bottomMargin = insets.bottom
            }
            WindowInsetsCompat.CONSUMED
        }
    }
}
