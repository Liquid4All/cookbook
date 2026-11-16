package ai.liquid.android.runtime

import java.io.File

class NativeAgentModel(private val nativeLibDir: String) {
    fun interface CancellationCallback { fun shouldContinue(): Boolean }
    private var handle = 0L
    private var loaded = false
    private var loadedPath: String? = null
    private val runner = NativeTaskRunner { if (handle != 0L) nativeUnload(handle); handle = 0 }
    init { System.loadLibrary("liquid-agent-jni") }

    suspend fun load(file: File) = runner.run {
        if (loaded) { check(loadedPath == file.absolutePath) { "Create a new runner to change models" }; return@run }
        if (handle == 0L) handle = nativeInit(nativeLibDir.toByteArray(Charsets.UTF_8))
        check(nativeLoad(handle, file.absolutePath.toByteArray(Charsets.UTF_8))) { "Failed to load the model" }
        loadedPath = file.absolutePath
        loaded = true
    }

    /** llama.cpp returns the parsed assistant message, including structured tool_calls. */
    suspend fun chat(messages: String, tools: String, maxTokens: Int = 1024): String = runner.run { active ->
        check(loaded) { "Model is not loaded" }
        nativeChat(handle, messages.toByteArray(Charsets.UTF_8), tools.toByteArray(Charsets.UTF_8),
            CancellationCallback { active() }, maxTokens).toString(Charsets.UTF_8)
    }

    fun close() = runner.close()
    private external fun nativeInit(directory: ByteArray): Long
    private external fun nativeLoad(handle: Long, path: ByteArray): Boolean
    private external fun nativeChat(handle: Long, messages: ByteArray, tools: ByteArray, callback: CancellationCallback, maxTokens: Int): ByteArray
    private external fun nativeUnload(handle: Long)
}
