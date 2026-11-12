package ai.liquid.vlmtestapp

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn

/**
 * Minimal Kotlin wrapper around llama.cpp's mtmd library (see app/src/main/cpp/vlm_jni.cpp).
 *
 * @param nativeLibDir the app's native library folder, where llama.cpp's CPU backends are installed
 */
class LlamaVlm(nativeLibDir: String) {
    fun interface TokenCallback {
        fun onToken(bytes: ByteArray)
    }

    init {
        System.loadLibrary("vlm-jni")
        nativeInit(nativeLibDir)
    }

    /** Loads the language model and its vision projector (mmproj). Call off the main thread. */
    fun load(modelPath: String, mmprojPath: String) {
        check(nativeLoad(modelPath, mmprojPath)) { "Failed to load the model" }
    }

    /** Describes an image (JPEG or PNG bytes) and streams the generated text. */
    fun generate(systemPrompt: String, prompt: String, image: ByteArray): Flow<String> =
        callbackFlow {
            val rc = nativeGenerate(systemPrompt, prompt, image) { bytes ->
                trySend(String(bytes, Charsets.UTF_8))
            }
            if (rc != 0) close(IllegalStateException("Generation failed (code $rc)")) else close()
        }.buffer(Channel.UNLIMITED).flowOn(Dispatchers.Default)

    /** Frees the model, projector and context. Call off the main thread. */
    fun unload() = nativeUnload()

    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeLoad(modelPath: String, mmprojPath: String): Boolean
    private external fun nativeGenerate(
        systemPrompt: String,
        prompt: String,
        image: ByteArray,
        callback: TokenCallback,
    ): Int
    private external fun nativeUnload()
}
