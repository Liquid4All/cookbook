package ai.liquid.browsercontrol.domain

data class BrowserObservation(
    val goal: String,          // 任务目标
    val axtree: String,        // 可访问性树
    val error: String? = null, // 错误信息
    val step: Int = 0          // 当前步骤
)
