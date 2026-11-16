package ai.liquid.android.runtime

import android.app.Application
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.CancellationException
import kotlinx.coroutines.Job
import kotlinx.coroutines.cancelAndJoin
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import kotlinx.coroutines.launch

data class GenerationState(
    val output: String = "",
    val status: String = "Ready",
    val busy: Boolean = false,
    val error: String? = null,
)

/** Retains the model across rotation and closes it only when this owner is cleared. */
open class TextTaskViewModel(
    application: Application,
    private val spec: ModelSpec,
    private val system: String,
    private val schema: String = "",
) : AndroidViewModel(application) {
    private val _generation = MutableStateFlow(GenerationState())
    val generation = _generation.asStateFlow()
    private var model: NativeTextModel? = null
    private var ready = false
    private var job: Job? = null

    fun generate(prompt: String) {
        if (job?.isActive == true || prompt.isBlank()) return
        _generation.value = GenerationState(status = "Preparing model...", busy = true)
        job = viewModelScope.launch {
            try {
                val app = getApplication<Application>()
                if (!ready) {
                    val file = ModelStore(checkNotNull(app.getExternalFilesDir(null))).ensure(spec) { progress ->
                        _generation.update { it.copy(status = "Downloading model: ${progress.downloaded * 100 / progress.total}%") }
                    }
                    _generation.update { it.copy(status = "Loading model...") }
                    val runner = model ?: NativeTextModel(app.applicationInfo.nativeLibraryDir).also { model = it }
                    runner.load(file)
                    ready = true
                }
                _generation.update { it.copy(status = "Generating...") }
                val output = StringBuilder()
                checkNotNull(model).generate(system, prompt, schema).collect { piece ->
                    output.append(piece)
                    _generation.update { it.copy(output = output.toString()) }
                }
                completed(output.toString())
                _generation.update { it.copy(status = "Complete") }
            } catch (cancelled: CancellationException) {
                _generation.update { it.copy(status = "Stopped") }
                throw cancelled
            } catch (error: Exception) {
                _generation.update { it.copy(status = "Please retry", error = error.message ?: "Unable to generate a response") }
            } catch (error: LinkageError) {
                _generation.update { it.copy(status = "Cannot load the native runtime", error = error.message) }
            } finally {
                _generation.update { it.copy(busy = false) }
            }
        }
    }

    protected open fun completed(output: String) {}
    fun stop() { job?.cancel() }
    suspend fun stopAndJoin() { job?.cancelAndJoin() }
    override fun onCleared() {
        model?.close()
        super.onCleared()
    }
}
