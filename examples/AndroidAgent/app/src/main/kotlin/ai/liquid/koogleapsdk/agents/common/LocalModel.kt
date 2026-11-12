package ai.liquid.koogleapsdk.agents.common

import ai.liquid.koogleapsdk.App
import android.app.DownloadManager
import android.net.Uri
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.io.File

/**
 * The on-device model shared by all agents. Downloaded once from Hugging Face,
 * then loaded the first time an agent needs it.
 */
object LocalModel {
    private const val MODEL_URL =
        "https://huggingface.co/LiquidAI/LFM2.5-1.2B-Instruct-GGUF/resolve/main/LFM2.5-1.2B-Instruct-Q5_K_M.gguf"

    private val mutex = Mutex()
    private var llm: LlamaChat? = null

    suspend fun get(): LlamaChat = mutex.withLock {
        llm ?: run {
            val context = App.context
            val file = File(context.getExternalFilesDir(null), MODEL_URL.substringAfterLast('/'))
            if (!file.exists()) download(file)
            withContext(Dispatchers.Default) {
                LlamaChat(context.applicationInfo.nativeLibraryDir).also { it.load(file.absolutePath) }
            }.also { llm = it }
        }
    }

    private suspend fun download(file: File) {
        val downloadManager = App.context.getSystemService(DownloadManager::class.java)
        val downloadId = downloadManager.enqueue(
            DownloadManager.Request(Uri.parse(MODEL_URL))
                .setTitle("Downloading LFM2.5-1.2B-Instruct")
                .setDestinationUri(Uri.fromFile(file))
        )
        while (true) {
            downloadManager.query(DownloadManager.Query().setFilterById(downloadId)).use { cursor ->
                if (cursor.moveToFirst()) {
                    when (cursor.getInt(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_STATUS))) {
                        DownloadManager.STATUS_SUCCESSFUL -> return
                        DownloadManager.STATUS_FAILED -> {
                            file.delete()
                            error("Model download failed")
                        }
                        else -> Log.d("LocalModel", "Downloaded ${cursor.getLong(cursor.getColumnIndexOrThrow(DownloadManager.COLUMN_BYTES_DOWNLOADED_SO_FAR)) / (1024 * 1024)} MB")
                    }
                }
            }
            delay(500)
        }
    }
}
