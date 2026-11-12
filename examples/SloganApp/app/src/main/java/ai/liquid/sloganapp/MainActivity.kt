package ai.liquid.sloganapp

import android.app.DownloadManager
import android.graphics.Color
import android.net.Uri
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
import com.arm.aichat.AiChat
import com.arm.aichat.InferenceEngine
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.Job
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.catch
import kotlinx.coroutines.flow.collect
import kotlinx.coroutines.flow.onCompletion
import kotlinx.coroutines.flow.onEach
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.File

// System prompt constants
private const val SYSTEM_PROMPT = "You are a marketing expert. Suggest engaging and creative slogans based on a business description provided by the user."
// User prompt template.
private const val USER_PROMPT_TEMPLATE = "Suggest slogans for this business: %s"


class MainActivity : ComponentActivity() {
    private lateinit var modelRunner: InferenceEngine
    private var modelLoaded = false
    private var generationJob: Job? = null

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

            val modelUrl = "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q5_K_M.gguf"
            val modelFile = File(getExternalFilesDir(null), "LFM2.5-1.2B-Instruct-Q5_K_M.gguf")

            // Check if model needs to be downloaded
            if (!modelFile.exists()) {
                // Model needs to be downloaded
                textView.text = "Starting download..."

                // Start the download
                val downloadManager = getSystemService(DownloadManager::class.java)
                val downloadId = downloadManager.enqueue(
                    DownloadManager.Request(Uri.parse(modelUrl))
                        .setTitle("Downloading Slogan Generator Model")
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
                                textView.text = "Download complete!"
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
                                textView.text = "Downloading: $percentage% ($downloadedMB MB / $totalMB MB)"
                            }
                        }
                    }
                    if (downloading) delay(500)
                }
            }

            modelRunner = withContext(Dispatchers.Default) {
                AiChat.getInferenceEngine(applicationContext).also {
                    it.loadModel(modelFile.absolutePath)
                    it.setSystemPrompt(SYSTEM_PROMPT)
                }
            }
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

            modelRunner.sendUserPrompt(USER_PROMPT_TEMPLATE.format(userInput)).onEach { chunk ->
                if (textView.text.startsWith(getText(R.string.generation_in_progress_label))) {
                    textView.text = chunk
                } else {
                    textView.append(chunk)
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
