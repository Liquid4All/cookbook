package ai.liquid.android.runtime

import java.io.File
import java.io.FileOutputStream
import java.net.HttpURLConnection
import java.net.URL
import java.security.MessageDigest
import java.util.concurrent.ConcurrentHashMap
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.currentCoroutineContext
import kotlinx.coroutines.ensureActive
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

data class ModelSpec(val name: String, val url: String, val size: Long, val sha256: String)
data class DownloadProgress(val name: String, val downloaded: Long, val total: Long)

/** A final filename denotes a complete, verified download. Interrupted transfers restart safely. */
class ModelStore(private val directory: File) {
    companion object { private val locks = ConcurrentHashMap<String, Mutex>() }

    suspend fun ensure(spec: ModelSpec, progress: (DownloadProgress) -> Unit = {}): File {
        require(File(spec.name).name == spec.name && spec.size > 0)
        val destination = File(directory, spec.name)
        return locks.getOrPut(destination.canonicalPath) { Mutex() }.withLock {
            withContext(Dispatchers.IO) {
                check(directory.isDirectory || directory.mkdirs()) { "Cannot create model directory" }
                if (destination.isFile && valid(destination, spec)) return@withContext destination
                check(!destination.exists() || destination.delete()) { "Cannot replace invalid model" }
                val partial = File(directory, "${spec.name}.part")
                check(!partial.exists() || partial.delete()) { "Cannot remove interrupted download" }
                val connection = URL(spec.url).openConnection() as HttpURLConnection
                connection.connectTimeout = 15_000
                connection.readTimeout = 15_000
                connection.instanceFollowRedirects = true
                try {
                    check(connection.responseCode == HttpURLConnection.HTTP_OK) {
                        "Download failed: HTTP ${connection.responseCode}"
                    }
                    val digest = MessageDigest.getInstance("SHA-256")
                    var downloaded = 0L
                    connection.inputStream.use { input ->
                        FileOutputStream(partial).use { output ->
                            val buffer = ByteArray(256 * 1024)
                            while (true) {
                                currentCoroutineContext().ensureActive()
                                val count = input.read(buffer)
                                if (count < 0) break
                                downloaded += count
                                check(downloaded <= spec.size) { "Downloaded model exceeds expected size" }
                                output.write(buffer, 0, count)
                                digest.update(buffer, 0, count)
                                progress(DownloadProgress(spec.name, downloaded, spec.size))
                            }
                            output.fd.sync()
                        }
                    }
                    currentCoroutineContext().ensureActive()
                    check(downloaded == spec.size && digest.digest().hex() == spec.sha256) {
                        "Downloaded model failed its size or SHA-256 check; retry the download"
                    }
                    check(partial.renameTo(destination)) { "Cannot finalize model download" }
                    destination
                } finally {
                    connection.disconnect()
                    partial.delete()
                }
            }
        }
    }

    private suspend fun valid(file: File, spec: ModelSpec): Boolean {
        if (file.length() != spec.size) return false
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { input ->
            val buffer = ByteArray(256 * 1024)
            while (true) {
                currentCoroutineContext().ensureActive()
                val count = input.read(buffer)
                if (count < 0) break
                digest.update(buffer, 0, count)
            }
        }
        return digest.digest().hex() == spec.sha256
    }
}

private fun ByteArray.hex() = joinToString("") { "%02x".format(it.toInt() and 255) }
