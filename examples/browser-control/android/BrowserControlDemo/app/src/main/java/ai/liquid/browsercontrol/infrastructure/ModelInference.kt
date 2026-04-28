package ai.liquid.browsercontrol.infrastructure

import ai.liquid.leap.Conversation
import ai.liquid.leap.ModelRunner
import ai.liquid.leap.downloader.LeapModelDownloader
import ai.liquid.leap.downloader.LeapModelDownloaderNotificationConfig
import ai.liquid.leap.message.MessageResponse
import android.content.Context
import android.os.Environment
import android.util.Log
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.flow
import kotlinx.coroutines.flow.flowOn
import kotlinx.coroutines.withContext
import java.io.File
import java.net.HttpURLConnection
import java.net.URL

class ModelInference(private val context: Context) {

    private var modelRunner: ModelRunner? = null
    private var conversation: Conversation? = null

    private val downloader = LeapModelDownloader(
        context,
        notificationConfig = LeapModelDownloaderNotificationConfig.build {
            notificationTitleDownloading = "正在加载浏览器控制模型"
            notificationTitleDownloaded = "模型已就绪"
        }
    )

    companion object {
        private const val TAG = "ModelInference"
        const val MODEL_NAME = "LFM2-350M"
        const val QUANTIZATION = "Q8_0"
        // 本地GGUF文件名（通过adb push放到此路径）
        const val LOCAL_GGUF_FILENAME = "LFM2-350M-Q8_0.gguf"
    }

    /**
     * 获取SDK缓存目录路径（用于adb push目标）
     */
    fun getModelCacheFolder(): File? {
        return downloader.getModelResourceFolder(MODEL_NAME, QUANTIZATION)
    }

    /**
     * 将SDK内部缓存路径映射到外部可访问路径（adb push目标）
     * 内部: /data/user/0/<pkg>/files/leap_models/LFM2-350M/Q8_0/
     * 外部: /sdcard/Android/data/<pkg>/files/leap_models/LFM2-350M/Q8_0/
     */
    private fun getExternalMirrorOfCacheFolder(): File? {
        val cacheFolder = getModelCacheFolder() ?: return null
        val internalFilesDir = context.filesDir.path       // /data/user/0/<pkg>/files
        val externalFilesDir = context.getExternalFilesDir(null)?.path ?: return null
        if (!cacheFolder.path.startsWith(internalFilesDir)) return null
        val relativePath = cacheFolder.path.removePrefix(internalFilesDir)
        return File("$externalFilesDir$relativePath")
    }

    /**
     * 查找本地GGUF文件，按优先级:
     * 1. SDK内部缓存目录（已缓存）
     * 2. SDK缓存的外部镜像路径（adb push推荐目标）
     *    /sdcard/Android/data/<pkg>/files/leap_models/LFM2-350M/Q8_0/
     * 3. /sdcard/Android/data/<pkg>/files/
     * 4. /sdcard/Download/
     */
    fun findLocalGgufFile(): File? {
        // 1. SDK内部缓存目录 — 用直接路径检查，避免 listFiles() 权限问题
        //    （Device Explorer 推送的文件归属 shell，app 无法 list 目录，但 exists() 可用）
        val cacheFolder = getModelCacheFolder()
        if (cacheFolder != null) {
            val directPath = File(cacheFolder, LOCAL_GGUF_FILENAME)
            if (directPath.exists() && directPath.length() > 0) {
                Log.i(TAG, "内部缓存命中(直接路径): ${directPath.path}")
                return directPath
            }
            // 降级：尝试 listFiles()（正常 app 拥有文件时可用）
            val cachedFile = cacheFolder.listFiles()?.firstOrNull { it.extension == "gguf" }
            if (cachedFile?.exists() == true) {
                Log.i(TAG, "内部缓存命中(listFiles): ${cachedFile.path}")
                return cachedFile
            }
        }

        // 2. SDK缓存的外部镜像（与adb push路径完全一致）
        val externalMirror = getExternalMirrorOfCacheFolder()
        if (externalMirror != null) {
            val candidate = File(externalMirror, LOCAL_GGUF_FILENAME)
            if (candidate.exists()) return candidate
        }

        // 3. /sdcard 根目录（最简单的 adb push 目标，app 可直接读取）
        //    adb push LFM2-350M-Q8_0.gguf /sdcard/LFM2-350M-Q8_0.gguf
        val sdcardRoot = Environment.getExternalStorageDirectory()
        val sdcardCandidate = File(sdcardRoot, LOCAL_GGUF_FILENAME)
        if (sdcardCandidate.exists() && sdcardCandidate.length() > 0) {
            Log.i(TAG, "sdcard根目录命中: ${sdcardCandidate.path}")
            return sdcardCandidate
        }

        // 4. 其他外部路径
        val searchPaths = listOf(
            context.getExternalFilesDir(null),
            context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        )
        for (dir in searchPaths) {
            if (dir == null) continue
            val candidate = File(dir, LOCAL_GGUF_FILENAME)
            if (candidate.exists() && candidate.length() > 0) return candidate
        }
        return null
    }

    /**
     * 获取推荐的adb push目标路径。
     * 将SDK内部缓存路径映射为外部可访问路径，确保路径一致。
     */
    fun getRecommendedAdbPushPath(): String {
        // 推荐直接推到 /sdcard 根目录，app 可直接读取，无需 run-as
        val sdcard = Environment.getExternalStorageDirectory().path
        return "$sdcard/$LOCAL_GGUF_FILENAME"
    }

    fun getRecommendedJsonPushPath(): String {
        val sdcard = Environment.getExternalStorageDirectory().path
        val jsonFilename = "$MODEL_NAME-$QUANTIZATION.json"
        return "$sdcard/$jsonFilename"
    }

    /**
     * 若文件不在SDK缓存目录，则复制过去
     */
    private suspend fun ensureModelInCache(sourceFile: File): File {
        val cacheFolder = getModelCacheFolder()
            ?: throw IllegalStateException("无法获取模型缓存目录")

        if (!cacheFolder.exists()) cacheFolder.mkdirs()

        val destFile = File(cacheFolder, sourceFile.name)
        if (destFile.canonicalPath == sourceFile.canonicalPath) return destFile

        Log.i(TAG, "复制模型文件到缓存: ${sourceFile.path} -> ${destFile.path}")
        withContext(Dispatchers.IO) {
            sourceFile.copyTo(destFile, overwrite = true)
        }
        return destFile
    }

    /** 递归列出目录下所有文件（含子目录），输出到 logcat。 */
    private fun listFilesRecursive(dir: File?, tag: String, prefix: String = "") {
        if (dir == null) {
            Log.i(tag, "${prefix}(目录为null)")
            return
        }
        if (!dir.exists()) {
            Log.i(tag, "${prefix}${dir.path} (不存在)")
            return
        }
        if (!dir.isDirectory) {
            Log.i(tag, "${prefix}${dir.name}  [${dir.length()} bytes]")
            return
        }
        Log.i(tag, "${prefix}${dir.path}/")
        val children = dir.listFiles()
        if (children.isNullOrEmpty()) {
            Log.i(tag, "${prefix}  (空目录)")
        } else {
            children.sortedBy { it.name }.forEach { child ->
                listFilesRecursive(child, tag, "$prefix  ")
            }
        }
    }

    /**
     * 确保 JSON metadata 文件存在于内部缓存。
     * 优先级：1) 已在缓存  2) 本地路径（与GGUF相同查找逻辑）  3) 网络下载（最后手段）
     */
    private suspend fun ensureJsonManifest(modelName: String, quantization: String): Boolean {
        val cacheFolder = getModelCacheFolder() ?: return false
        val jsonFilename = "$modelName-$quantization.json"
        val jsonFile = File(cacheFolder, jsonFilename)

        // 1. 已在内部缓存
        if (jsonFile.exists() && jsonFile.length() > 0) {
            Log.i(TAG, "JSON manifest已存在: ${jsonFile.path} (${jsonFile.length()} bytes)")
            return true
        }

        // 2. 在本地路径查找（/sdcard 根目录优先，app 可直接读取）
        val localSearchDirs = listOfNotNull(
            Environment.getExternalStorageDirectory(),          // /sdcard/
            getExternalMirrorOfCacheFolder(),
            context.getExternalFilesDir(null),
            context.getExternalFilesDir(Environment.DIRECTORY_DOWNLOADS),
            Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
        )
        for (dir in localSearchDirs) {
            val candidate = File(dir, jsonFilename)
            if (candidate.exists() && candidate.length() > 0) {
                Log.i(TAG, "本地JSON manifest命中: ${candidate.path}，复制到缓存...")
                return withContext(Dispatchers.IO) {
                    try {
                        cacheFolder.mkdirs()
                        candidate.copyTo(jsonFile, overwrite = true)
                        Log.i(TAG, "JSON manifest复制完成 (${jsonFile.length()} bytes)")
                        true
                    } catch (e: Exception) {
                        Log.w(TAG, "JSON manifest复制失败: ${e.message}")
                        false
                    }
                }
            }
        }

        // 3. 网络下载（最后手段，模拟器可能无法访问）
        val jsonUrl = "https://huggingface.co/LiquidAI/LeapBundles/resolve/main/$modelName-GGUF/$quantization.json"
        Log.i(TAG, "本地未找到JSON manifest，尝试网络下载: $jsonUrl")
        return withContext(Dispatchers.IO) {
            try {
                val conn = URL(jsonUrl).openConnection() as HttpURLConnection
                conn.connectTimeout = 15000
                conn.readTimeout = 15000
                conn.connect()
                if (conn.responseCode == 200) {
                    cacheFolder.mkdirs()
                    jsonFile.writeBytes(conn.inputStream.readBytes())
                    Log.i(TAG, "JSON manifest网络下载成功 (${jsonFile.length()} bytes)")
                    true
                } else {
                    Log.w(TAG, "JSON manifest下载失败: HTTP ${conn.responseCode}")
                    false
                }
            } catch (e: Exception) {
                Log.w(TAG, "JSON manifest网络下载失败: ${e.message}")
                Log.w(TAG, "请手动推送: adb push $jsonFilename /sdcard/$jsonFilename")
                Log.w(TAG, "然后: adb shell \"run-as ai.liquid.browsercontrol cp /sdcard/$jsonFilename files/leap_models/$modelName-$quantization/$jsonFilename\"")
                false
            }
        }
    }

    suspend fun loadModel(
        modelName: String = MODEL_NAME,
        quantization: String = QUANTIZATION,
        systemPrompt: String
    ) {
        // 打印所有候选路径，方便调试
        val internalCache = getModelCacheFolder()
        val externalMirror = getExternalMirrorOfCacheFolder()
        val appExtDir = context.getExternalFilesDir(null)
        Log.i(TAG, "=== 路径诊断 ===")
        Log.i(TAG, "内部缓存目录: ${internalCache?.path} | exists=${internalCache?.exists()}")
        Log.i(TAG, "外部镜像目录: ${externalMirror?.path} | exists=${externalMirror?.exists()}")
        // 直接按已知文件名探测，不依赖 listFiles()（Device Explorer 推送的文件权限可能导致 listFiles() 返回 null）
        val ggufInternal = internalCache?.let { File(it, LOCAL_GGUF_FILENAME) }
        val jsonInternal = internalCache?.let { File(it, "$modelName-$quantization.json") }
        val ggufExternal = externalMirror?.let { File(it, LOCAL_GGUF_FILENAME) }
        val jsonExternal = externalMirror?.let { File(it, "$modelName-$quantization.json") }
        Log.i(TAG, "内部 GGUF: exists=${ggufInternal?.exists()} size=${ggufInternal?.length()}")
        Log.i(TAG, "内部 JSON: exists=${jsonInternal?.exists()} size=${jsonInternal?.length()}")
        Log.i(TAG, "外部 GGUF: exists=${ggufExternal?.exists()} size=${ggufExternal?.length()}")
        Log.i(TAG, "外部 JSON: exists=${jsonExternal?.exists()} size=${jsonExternal?.length()}")
        // listFiles() 结果（仅供参考，权限不足时返回 null）
        Log.i(TAG, "内部缓存 listFiles: ${internalCache?.listFiles()?.joinToString { "${it.name}(${it.length()})" } ?: "null(无权限或为空)"}")
        Log.i(TAG, "外部镜像 listFiles: ${externalMirror?.listFiles()?.joinToString { "${it.name}(${it.length()})" } ?: "null(无权限或为空)"}")
        Log.i(TAG, "===============")

        // 1. 检查本地是否已有GGUF文件
        val localFile = findLocalGgufFile()
        if (localFile != null) {
            Log.i(TAG, "找到本地模型文件: ${localFile.path}")
            ensureModelInCache(localFile)
            // 下载 JSON manifest（小文件），让 downloader 识别模型已就绪
            val jsonOk = ensureJsonManifest(modelName, quantization)
            Log.i(TAG, "JSON manifest: ${if (jsonOk) "就绪" else "获取失败，将尝试继续"}")
        }

        // 2. 检查状态（GGUF + JSON 都在时返回 Downloaded）
        val status = downloader.queryStatus(modelName, quantization)
        Log.i(TAG, "downloader状态: $status")

        if (status is LeapModelDownloader.ModelDownloadStatus.NotOnLocal && localFile == null) {
            val pushTarget = getRecommendedAdbPushPath()
            val msg = "未找到本地模型文件。\n" +
                "请先创建目录并推送:\n" +
                "adb shell mkdir -p \"${externalMirror?.path}\"\n" +
                "adb push $LOCAL_GGUF_FILENAME \"$pushTarget\""
            Log.e(TAG, msg)
            throw IllegalStateException(msg)
        }

        modelRunner = downloader.loadModel(
            modelName = modelName,
            quantizationType = quantization
        )

        conversation = modelRunner?.createConversation(systemPrompt)
    }

    fun generateAction(userPrompt: String): Flow<String> = flow {
        val conv = conversation ?: throw IllegalStateException("Model not loaded")

        conv.generateResponse(userPrompt).collect { response ->
            when (response) {
                is MessageResponse.Chunk -> {
                    emit(response.text)
                }
                else -> {
                    // 其他消息类型忽略
                }
            }
        }
    }.flowOn(Dispatchers.Default) // 推理在后台线程，避免阻塞 UI

    fun isLoaded(): Boolean = conversation != null

    fun cleanup() {
        conversation = null
        modelRunner = null
    }
}
