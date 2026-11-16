package ai.liquid.android.runtime

import java.io.File
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow

class NativeVisionModel(private val nativeLibDir: String) {
    private var handle = 0L
    private var loaded = false
    private var loadedPath: String? = null
    private var loadedProjectorPath: String? = null
    private val runner = NativeTaskRunner { if (handle != 0L) nativeUnload(handle); handle = 0 }
    init { System.loadLibrary("liquid-vision-jni") }

    suspend fun load(model: File, projector: File) = runner.run {
        if (loaded) {
            check(loadedPath == model.absolutePath && loadedProjectorPath == projector.absolutePath) {
                "Create a new runner to change models or projectors"
            }
            return@run
        }
        if (handle == 0L) handle = nativeInit(nativeLibDir.toByteArray(Charsets.UTF_8))
        check(nativeLoad(handle, model.absolutePath.toByteArray(Charsets.UTF_8), projector.absolutePath.toByteArray(Charsets.UTF_8))) {
            "Failed to load the model or projector"
        }
        loadedPath = model.absolutePath
        loadedProjectorPath = projector.absolutePath
        loaded = true
    }

    fun generate(system: String, prompt: String, image: ByteArray, maxTokens: Int = 512): Flow<String> = callbackFlow {
        runner.run { active ->
            check(loaded) { "Model is not loaded" }
            val callback = object : NativeTextModel.TokenCallback {
                override fun shouldContinue() = active()
                override fun onToken(bytes: ByteArray) = active() && trySend(bytes.toString(Charsets.UTF_8)).isSuccess
            }
            checkGeneration(nativeGenerate(handle, system.toByteArray(Charsets.UTF_8), prompt.toByteArray(Charsets.UTF_8), image, callback, maxTokens))
        }
        close()
    }.buffer(Channel.UNLIMITED)

    fun close() = runner.close()
    private external fun nativeInit(directory: ByteArray): Long
    private external fun nativeLoad(handle: Long, model: ByteArray, projector: ByteArray): Boolean
    private external fun nativeGenerate(handle: Long, system: ByteArray, prompt: ByteArray, image: ByteArray, callback: NativeTextModel.TokenCallback, maxTokens: Int): Int
    private external fun nativeUnload(handle: Long)
}
