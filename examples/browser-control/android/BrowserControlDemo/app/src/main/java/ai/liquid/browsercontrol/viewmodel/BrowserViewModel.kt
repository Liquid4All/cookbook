package ai.liquid.browsercontrol.viewmodel

import ai.liquid.browsercontrol.domain.BrowserAction
import ai.liquid.browsercontrol.domain.BrowserObservation
import ai.liquid.browsercontrol.domain.PromptFormatter
import ai.liquid.browsercontrol.domain.parseAction
import ai.liquid.browsercontrol.infrastructure.ActionExecutor
import ai.liquid.browsercontrol.infrastructure.ModelInference
import ai.liquid.browsercontrol.infrastructure.WebViewAccessibility
import android.app.Application
import android.webkit.WebView
import androidx.lifecycle.AndroidViewModel
import androidx.lifecycle.viewModelScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withContext
import java.io.InputStreamReader
import kotlin.coroutines.resume

sealed class ModelState {
    object Idle : ModelState()
    data class Loading(val progress: String) : ModelState()
    object Ready : ModelState()
    data class Error(val message: String) : ModelState()
}

sealed class TaskState {
    object Idle : TaskState()
    data class Running(val step: Int, val maxSteps: Int) : TaskState()
    data class Completed(val success: Boolean, val steps: Int) : TaskState()
}

data class LogEntry(
    val timestamp: Long,
    val type: String, // "info", "observation", "action", "result", "error"
    val content: String
)

class BrowserViewModel(application: Application) : AndroidViewModel(application) {

    private val modelInference = ModelInference(application)

    private val _modelState = MutableStateFlow<ModelState>(ModelState.Idle)
    val modelState: StateFlow<ModelState> = _modelState.asStateFlow()

    private val _taskState = MutableStateFlow<TaskState>(TaskState.Idle)
    val taskState: StateFlow<TaskState> = _taskState.asStateFlow()

    private val _logs = MutableStateFlow<List<LogEntry>>(emptyList())
    val logs: StateFlow<List<LogEntry>> = _logs.asStateFlow()

    private val _currentAXTree = MutableStateFlow("")
    val currentAXTree: StateFlow<String> = _currentAXTree.asStateFlow()

    private var stopRequested = false

    private val systemPrompt: String by lazy {
        try {
            val resources = getApplication<Application>().resources
            val id = resources.getIdentifier(
                "system_prompt", "raw",
                getApplication<Application>().packageName
            )
            InputStreamReader(resources.openRawResource(id)).readText()
        } catch (e: Exception) {
            // 默认系统提示
            """You are a browser automation agent. Given a task goal and the page structure (AXTree),
output a single action to take. Actions: noop(), click('bid'), fill('bid', 'text'),
send_keys('text'), scroll('up'/'down').
Output only the action, nothing else.""".trimIndent()
        }
    }

    fun loadModel() {
        viewModelScope.launch(Dispatchers.Default) {
            try {
                _modelState.value = ModelState.Loading("路径诊断中...")

                // 打印所有路径信息到UI日志
                val cacheFolder = modelInference.getModelCacheFolder()
                val pushPath = modelInference.getRecommendedAdbPushPath()
                addLog("info", "[路径] 内部缓存: ${cacheFolder?.path}")
                addLog("info", "[路径] adb push 目标: $pushPath")

                val localFile = modelInference.findLocalGgufFile()
                if (localFile != null) {
                    addLog("info", "[✓] 找到GGUF: ${localFile.name}")
                    addLog("info", "正在复制到缓存并获取JSON manifest...")
                    _modelState.value = ModelState.Loading("同步模型文件...")
                } else {
                    addLog("error", "[✗] 未找到GGUF文件")
                    addLog("info", "请执行以下命令推送模型（直接推到sdcard，无需run-as）:")
                    addLog("info", "adb push LFM2-350M-Q8_0.gguf /sdcard/LFM2-350M-Q8_0.gguf")
                    addLog("info", "adb push LFM2-350M-Q8_0.json /sdcard/LFM2-350M-Q8_0.json")
                    _modelState.value = ModelState.Loading("等待本地文件...")
                }

                modelInference.loadModel(systemPrompt = systemPrompt)

                _modelState.value = ModelState.Ready
                addLog("info", "✓ 模型加载成功")
            } catch (e: Exception) {
                _modelState.value = ModelState.Error(e.message?.take(60) ?: "未知错误")
                addLog("error", "✗ 加载失败: ${e.message}")
            }
        }
    }

    fun runTask(webView: WebView, goal: String = "Click the button", maxSteps: Int = 5) {
        if (!modelInference.isLoaded()) {
            addLog("error", "模型未加载，请先加载模型")
            return
        }

        stopRequested = false

        viewModelScope.launch(Dispatchers.Default) {
            try {
                _taskState.value = TaskState.Running(0, maxSteps)
                addLog("info", "========== 任务开始 ==========")
                addLog("info", "目标: $goal")

                // 0. 自动点击 MiniWoB START 按钮，初始化 episode
                addLog("info", "初始化 MiniWoB episode...")
                val startResult = withContext(Dispatchers.Main) {
                    startMiniWobEpisode(webView)
                }
                addLog("info", "START 结果: $startResult")
                delay(500) // 等待任务内容渲染

                var lastError: String? = null

                for (step in 0 until maxSteps) {
                    if (stopRequested) {
                        addLog("info", "任务已停止")
                        _taskState.value = TaskState.Idle
                        return@launch
                    }

                    _taskState.value = TaskState.Running(step, maxSteps)
                    addLog("info", "--- Step ${step + 1} / $maxSteps ---")

                    // 1. 提取 accessibility tree（必须在主线程）
                    val axtree = withContext(Dispatchers.Main) {
                        WebViewAccessibility.extractAXTree(webView)
                    }
                    _currentAXTree.value = axtree
                    addLog("observation", "AXTree (${axtree.length} chars):\n${axtree.take(300)}${if (axtree.length > 300) "..." else ""}")

                    // 2. 构建 observation
                    val observation = BrowserObservation(
                        goal = goal,
                        axtree = axtree,
                        error = lastError,
                        step = step
                    )

                    // 3. 格式化 prompt
                    val userPrompt = PromptFormatter.formatUserPrompt(observation)

                    // 4. 模型推理
                    addLog("info", "正在推理...")
                    val responseBuilder = StringBuilder()

                    modelInference.generateAction(userPrompt).collect { chunk ->
                        responseBuilder.append(chunk)
                    }

                    val response = responseBuilder.toString()
                    addLog("action", "模型输出:\n$response")

                    // 5. 解析动作
                    val action = parseAction(response)
                    addLog("action", "解析动作: $action")

                    // 6. 执行动作（必须在主线程）
                    delay(500)
                    val result = withContext(Dispatchers.Main) {
                        ActionExecutor.execute(webView, action)
                    }

                    result.fold(
                        onSuccess = { message ->
                            addLog("result", "✓ $message")
                            lastError = null
                        },
                        onFailure = { error ->
                            addLog("error", "✗ ${error.message}")
                            lastError = error.message
                        }
                    )

                    // 7. 轮询 MiniWoB episode 结束状态（最多等 3s，每 300ms 检查一次）
                    var reward: Double? = null
                    repeat(10) { tick ->
                        delay(300)
                        val r = withContext(Dispatchers.Main) { getMiniWobReward(webView) }
                        if (r != null) {
                            reward = r
                            return@repeat
                        }
                    }
                    addLog("info", "Reward: $reward")

                    when {
                        reward != null && reward!! > 0.0 -> {
                            addLog("info", "========== 任务成功 ✓ reward=$reward ==========")
                            withContext(Dispatchers.Main) { showResultOverlay(webView, reward!!) }
                            _taskState.value = TaskState.Completed(true, step + 1)
                            return@launch
                        }
                        reward != null && reward!! < 0.0 -> {
                            addLog("info", "========== 任务失败 ✗ reward=$reward ==========")
                            withContext(Dispatchers.Main) { showResultOverlay(webView, reward!!) }
                            _taskState.value = TaskState.Completed(false, step + 1)
                            return@launch
                        }
                        // reward == null 或 0.0 → 继续下一步
                    }
                }

                addLog("info", "========== 达到最大步数 ==========")
                _taskState.value = TaskState.Completed(false, maxSteps)

            } catch (e: Exception) {
                addLog("error", "任务执行出错: ${e.message}")
                _taskState.value = TaskState.Idle
            }
        }
    }

    /** 在主线程执行 JS 并返回结果字符串（去掉 JSON 引号）*/
    private suspend fun evaluateJs(webView: WebView, script: String): String? =
        withContext(Dispatchers.Main) {
            suspendCancellableCoroutine { cont ->
                webView.evaluateJavascript(script) { result ->
                    cont.resume(result?.trim('"'))
                }
            }
        }

    /** 自动点击 MiniWoB START 按钮，触发 episode 初始化；同时拦截 endEpisode 捕获 reward */
    private suspend fun startMiniWobEpisode(webView: WebView): String? =
        evaluateJs(webView, """
            (function() {
                try {
                    if (typeof core === 'undefined') return 'no-core';
                    // 重置上次结果标志
                    window._miniWobEpisodeDone = false;
                    window._lastMiniWobReward = undefined;
                    // 拦截 endEpisode —— 在 episode 真正结束前记下 reward
                    // 避免页面重置后 core.getReward() 返回 null
                    if (!core._endEpisodePatched) {
                        var _origEnd = core.endEpisode.bind(core);
                        core.endEpisode = function(r) {
                            window._lastMiniWobReward = r;
                            window._miniWobEpisodeDone = true;
                            _origEnd(r);
                        };
                        core._endEpisodePatched = true;
                    }
                    core.EPISODE_MAX_TIME = 60000;
                    if (core.startEpisodeReal) {
                        core.startEpisodeReal(null);
                    } else {
                        var btn = document.getElementById('start-button')
                               || document.querySelector('button');
                        if (btn) btn.click(); else return 'not-found';
                    }
                    if (core.timer_ && core.timer_.timeoutId_) {
                        clearTimeout(core.timer_.timeoutId_);
                        core.timer_.timeoutId_ = setTimeout(function() {
                            core.endEpisode(-1);
                        }, 60000);
                        return 'started+timer_extended_60s+interceptor_ok';
                    }
                    return 'started+interceptor_ok';
                } catch(e) { return 'error:' + e.message; }
            })()
        """.trimIndent())

    /** 读取 MiniWoB 当前 episode 的 reward
     *  返回 null  = episode 仍在进行
     *  返回 Double = episode 已结束（值来自拦截器缓存，页面重置后仍有效）
     */
    private suspend fun getMiniWobReward(webView: WebView): Double? {
        val raw = evaluateJs(webView, """
            (function() {
                try {
                    // 优先读拦截器缓存（episode 结束后仍有效）
                    if (window._miniWobEpisodeDone === true) {
                        return String(window._lastMiniWobReward);
                    }
                    return 'ongoing';
                } catch(e) { return 'ongoing'; }
            })()
        """.trimIndent()) ?: return null
        return if (raw == "ongoing") null else raw.toDoubleOrNull()
    }

    /** 在 WebView 内注入结果覆盖层，显示评分结果 */
    private suspend fun showResultOverlay(webView: WebView, reward: Double) {
        val color = if (reward > 0) "#4CAF50" else "#F44336"
        val label = if (reward > 0) "✓  任务成功" else "✗  任务失败"
        evaluateJs(webView, """
            (function() {
                var old = document.getElementById('_copilot_result_overlay');
                if (old) old.remove();
                var d = document.createElement('div');
                d.id = '_copilot_result_overlay';
                d.style = 'position:fixed;top:0;left:0;width:100%;height:100%;'
                         + 'background:rgba(0,0,0,0.72);display:flex;'
                         + 'flex-direction:column;align-items:center;'
                         + 'justify-content:center;z-index:99999;';
                d.innerHTML = '<div style="font-size:28px;font-weight:bold;color:$color;">'
                            + '$label</div>'
                            + '<div style="font-size:20px;color:#fff;margin-top:12px;">'
                            + 'Reward: ${'$'}{window._lastMiniWobReward}</div>';
                document.body.appendChild(d);
            })()
        """.trimIndent())
    }

    fun stopTask() {
        stopRequested = true
        addLog("info", "正在停止任务...")
    }

    fun resetTask() {
        _taskState.value = TaskState.Idle
        _logs.value = emptyList()
        _currentAXTree.value = ""
    }

    private fun addLog(type: String, content: String) {
        val entry = LogEntry(
            timestamp = System.currentTimeMillis(),
            type = type,
            content = content
        )
        _logs.value = _logs.value + entry
    }

    override fun onCleared() {
        super.onCleared()
        modelInference.cleanup()
    }
}
