package ai.liquid.vlmtestapp

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import ai.liquid.android.runtime.ModelStore
import ai.liquid.android.runtime.Models
import ai.liquid.android.runtime.NativeVisionModel
import java.io.File
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class VisionState(
    val text: String = "Take a picture to describe it",
    val image: File? = null,
    val busy: Boolean = false,
    val capturing: Boolean = false,
    val error: String? = null,
)

class VlmViewModel(application: Application) : AndroidViewModel(application) {
    private val _state = MutableStateFlow(VisionState())
    val state = _state.asStateFlow()
    private var model: NativeVisionModel? = null
    private var loaded = false
    private var job: Job? = null

    fun capturing(value: Boolean) { _state.update { it.copy(capturing = value) } }
    fun describe(image: File) {
        if (job?.isActive == true) return
        _state.value = VisionState(image = image, text = "Preparing model...", busy = true)
        job = viewModelScope.launch {
            try {
                val app = getApplication<Application>()
                if (!loaded) {
                    val store = ModelStore(checkNotNull(app.getExternalFilesDir(null)))
                    val progress: (ai.liquid.android.runtime.DownloadProgress) -> Unit = { p ->
                        _state.update { it.copy(text = "Downloading model: ${p.downloaded * 100 / p.total}%") }
                    }
                    val weights = store.ensure(Models.vision, progress)
                    val projector = store.ensure(Models.projector, progress)
                    _state.update { it.copy(text = "Loading model...") }
                    val runner = model ?: NativeVisionModel(app.applicationInfo.nativeLibraryDir).also { model = it }
                    runner.load(weights, projector)
                    loaded = true
                }
                val output = StringBuilder()
                checkNotNull(model).generate("You are a helpful multimodal assistant by Liquid AI.",
                    "Describe this image.", image.readBytes()).collect { piece ->
                    output.append(piece)
                    _state.update { it.copy(text = output.toString()) }
                }
            } catch (cancelled: CancellationException) {
                throw cancelled
            } catch (error: Exception) {
                _state.update { it.copy(error = error.message ?: "Unable to describe the image") }
            } catch (error: LinkageError) {
                _state.update { it.copy(error = "Cannot load the native runtime: ${error.message}") }
            } finally {
                _state.update { it.copy(busy = false) }
            }
        }
    }

    override fun onCleared() {
        model?.close()
        super.onCleared()
    }
}
