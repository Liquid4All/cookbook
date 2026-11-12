package ai.liquid.recipegenerator

import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.channels.Channel
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.buffer
import kotlinx.coroutines.flow.callbackFlow
import kotlinx.coroutines.flow.flowOn

/**
 * Minimal Kotlin wrapper around llama.cpp that generates JSON constrained to a schema
 * (see app/src/main/cpp/llama_json_jni.cpp).
 *
 * @param nativeLibDir the app's native library folder, where llama.cpp's CPU backends are installed
 */
class LlamaJson(nativeLibDir: String) {
    fun interface TokenCallback {
        fun onToken(bytes: ByteArray)
    }

    init {
        System.loadLibrary("llama-json-jni")
        nativeInit(nativeLibDir)
    }

    /** Loads the model. Call off the main thread. */
    fun load(modelPath: String) {
        check(nativeLoad(modelPath)) { "Failed to load the model" }
    }

    /** Generates a response that matches [jsonSchema] and streams the generated text. */
    fun generate(systemPrompt: String, prompt: String, jsonSchema: String): Flow<String> =
        callbackFlow {
            val rc = nativeGenerate(systemPrompt, prompt, jsonSchema) { bytes ->
                trySend(String(bytes, Charsets.UTF_8))
            }
            if (rc != 0) close(IllegalStateException("Generation failed (code $rc)")) else close()
        }.buffer(Channel.UNLIMITED).flowOn(Dispatchers.Default)

    /** Frees the model and context. Call off the main thread. */
    fun unload() = nativeUnload()

    private external fun nativeInit(nativeLibDir: String)
    private external fun nativeLoad(modelPath: String): Boolean
    private external fun nativeGenerate(
        systemPrompt: String,
        prompt: String,
        jsonSchema: String,
        callback: TokenCallback,
    ): Int
    private external fun nativeUnload()
}
