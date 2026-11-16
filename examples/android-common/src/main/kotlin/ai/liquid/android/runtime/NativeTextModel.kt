package ai.liquid.android.runtime

import java.io.File
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow

class NativeTextModel(private val nativeLibDir: String) {
    interface TokenCallback {
        fun onToken(bytes: ByteArray): Boolean
        fun shouldContinue(): Boolean
    }
    private var handle = 0L
    private var loaded = false
    private var loadedPath: String? = null
    private val runner = NativeTaskRunner { if (handle != 0L) nativeUnload(handle); handle = 0 }
    init { System.loadLibrary("liquid-text-jni") }

    suspend fun load(file: File) = runner.run {
        if (loaded) { check(loadedPath == file.absolutePath) { "Create a new runner to change models" }; return@run }
        if (handle == 0L) handle = nativeInit(nativeLibDir.toByteArray(Charsets.UTF_8))
        check(nativeLoad(handle, file.absolutePath.toByteArray(Charsets.UTF_8))) { "Failed to load the model" }
        loadedPath = file.absolutePath
        loaded = true
    }

    /** Each call starts a new conversation. An optional JSON schema constrains the output. */
    fun generate(system: String, prompt: String, schema: String = "", maxTokens: Int = 1024): Flow<String> = callbackFlow {
        runner.run { active ->
            check(loaded) { "Model is not loaded" }
            val callback = object : TokenCallback {
                override fun shouldContinue() = active()
                override fun onToken(bytes: ByteArray) = active() && trySend(bytes.toString(Charsets.UTF_8)).isSuccess
            }
            val code = nativeGenerate(handle, system.toByteArray(Charsets.UTF_8), prompt.toByteArray(Charsets.UTF_8),
                schema.toByteArray(Charsets.UTF_8), callback, maxTokens)
            checkGeneration(code)
        }
        close()
    }.buffer(Channel.UNLIMITED)

    fun close() = runner.close()
    private external fun nativeInit(directory: ByteArray): Long
    private external fun nativeLoad(handle: Long, path: ByteArray): Boolean
    private external fun nativeGenerate(handle: Long, system: ByteArray, prompt: ByteArray, schema: ByteArray, callback: TokenCallback, maxTokens: Int): Int
    private external fun nativeUnload(handle: Long)
}

internal fun checkGeneration(code: Int) {
    check(code == 0) {
        when (code) {
            1 -> "The prompt exceeds the model context; shorten or split your input"
            5 -> "Generation cancelled"
            6 -> "Generation reached its token or context limit before completing; try a shorter request"
            10 -> "Unable to decode the image; try another photo"
            else -> "Generation failed (code $code)"
        }
    }
}
