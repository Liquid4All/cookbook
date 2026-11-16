package ai.liquid.koogleapsdk.agents.common

import ai.liquid.android.runtime.ModelStore
import ai.liquid.android.runtime.Models
import ai.liquid.koogleapsdk.App
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock

/** The process owns one model; its native queue serializes requests from every agent. */
object LocalModel {
    private val mutex = Mutex()
    private var llm: LlamaChat? = null

    suspend fun get(): LlamaChat = mutex.withLock {
        llm ?: run {
            val context = App.context
            val file = ModelStore(checkNotNull(context.getExternalFilesDir(null))).ensure(Models.text)
            val model = LlamaChat(context.applicationInfo.nativeLibraryDir)
            try {
                model.load(file.absolutePath)
                model.also { llm = it }
            } catch (error: Exception) {
                model.close()
                throw error
            }
        }
    }
}
