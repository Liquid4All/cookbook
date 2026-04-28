package ai.liquid.browsercontrol.ui

import ai.liquid.browsercontrol.AppConfig
import ai.liquid.browsercontrol.viewmodel.BrowserViewModel
import ai.liquid.browsercontrol.viewmodel.LogEntry
import ai.liquid.browsercontrol.viewmodel.ModelState
import ai.liquid.browsercontrol.viewmodel.TaskState
import android.webkit.WebView
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.*
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.lazy.rememberLazyListState
import androidx.compose.material3.*
import androidx.compose.runtime.*
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import java.text.SimpleDateFormat
import java.util.*

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun BrowserControlScreen(viewModel: BrowserViewModel) {
    val modelState by viewModel.modelState.collectAsState()
    val taskState by viewModel.taskState.collectAsState()
    val logs by viewModel.logs.collectAsState()
    val axtree by viewModel.currentAXTree.collectAsState()

    var webView: WebView? by remember { mutableStateOf(null) }
    var showAXTree by remember { mutableStateOf(false) }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Browser Control Demo") },
                colors = TopAppBarDefaults.topAppBarColors(
                    containerColor = MaterialTheme.colorScheme.primaryContainer
                )
            )
        }
    ) { padding ->
        Column(
            modifier = Modifier
                .fillMaxSize()
                .padding(padding)
        ) {
            // 状态栏
            StatusBar(modelState, taskState)

            // 主内容区域 - 上下布局
            Column(
                modifier = Modifier
                    .fillMaxWidth()
                    .weight(1f)
            ) {
                // 上方: WebView
                Box(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1f)
                        .background(Color(0xFF1E1E1E))  // 与日志区同色，消除 WebView 空白灰色
                ) {
                    WebViewCompose(
                        url = AppConfig.taskUrl("click-button"),
                        modifier = Modifier.fillMaxSize(),
                        onWebViewCreated = { webView = it }
                    )
                }

                // 下方: 日志（权重更大，向上占满剩余空间）
                Column(
                    modifier = Modifier
                        .fillMaxWidth()
                        .weight(1.5f)
                        .background(Color(0xFF1E1E1E))
                ) {
                    // 标签栏
                    TabRow(selectedTabIndex = if (showAXTree) 1 else 0) {
                        Tab(
                            selected = !showAXTree,
                            onClick = { showAXTree = false },
                            text = { Text("日志") }
                        )
                        Tab(
                            selected = showAXTree,
                            onClick = { showAXTree = true },
                            text = { Text("AXTree") }
                        )
                    }

                    // 内容
                    if (showAXTree) {
                        AXTreeView(axtree)
                    } else {
                        LogView(logs)
                    }
                }
            }

            // 控制按钮
            ControlButtons(
                modelState = modelState,
                taskState = taskState,
                onLoadModel = { viewModel.loadModel() },
                onStartTask = { webView?.let { viewModel.runTask(it) } },
                onStopTask = { viewModel.stopTask() },
                onResetTask = {
                    viewModel.resetTask()
                    webView?.reload()
                }
            )
        }
    }
}

@Composable
fun StatusBar(modelState: ModelState, taskState: TaskState) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.secondaryContainer
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.SpaceBetween
        ) {
            Text(
                text = when (modelState) {
                    is ModelState.Idle -> "⚪ 模型: 未加载"
                    is ModelState.Loading -> "🟡 模型: ${modelState.progress}"
                    is ModelState.Ready -> "🟢 模型: 就绪"
                    is ModelState.Error -> "🔴 模型: ${modelState.message}"
                },
                style = MaterialTheme.typography.bodyMedium
            )

            Text(
                text = when (taskState) {
                    is TaskState.Idle -> "任务: 待开始"
                    is TaskState.Running -> "任务: Step ${taskState.step + 1}/${taskState.maxSteps}"
                    is TaskState.Completed -> if (taskState.success) {
                        "✓ 任务完成 (${taskState.steps} steps)"
                    } else {
                        "任务未完成 (${taskState.steps} steps)"
                    }
                },
                style = MaterialTheme.typography.bodyMedium
            )
        }
    }
}

@Composable
fun LogView(logs: List<LogEntry>) {
    val listState = rememberLazyListState()

    LaunchedEffect(logs.size) {
        if (logs.isNotEmpty()) {
            listState.animateScrollToItem(logs.size - 1)
        }
    }

    LazyColumn(
        state = listState,
        modifier = Modifier
            .fillMaxSize()
            .padding(8.dp)
    ) {
        items(logs) { log ->
            LogItem(log)
        }
    }
}

@Composable
fun LogItem(log: LogEntry) {
    val color = when (log.type) {
        "info" -> Color(0xFFBBBBBB)
        "observation" -> Color(0xFF64B5F6)
        "action" -> Color(0xFF81C784)
        "result" -> Color(0xFFFFD54F)
        "error" -> Color(0xFFE57373)
        else -> Color.White
    }

    val timeFormat = SimpleDateFormat("HH:mm:ss", Locale.getDefault())
    val timeStr = timeFormat.format(Date(log.timestamp))

    Text(
        text = "[$timeStr] ${log.content}",
        color = color,
        fontSize = 12.sp,
        fontFamily = FontFamily.Monospace,
        modifier = Modifier.padding(vertical = 2.dp)
    )
}

@Composable
fun AXTreeView(axtree: String) {
    LazyColumn(
        modifier = Modifier
            .fillMaxSize()
            .padding(8.dp)
    ) {
        item {
            Text(
                text = axtree.ifEmpty { "未提取" },
                color = Color(0xFF90CAF9),
                fontSize = 11.sp,
                fontFamily = FontFamily.Monospace
            )
        }
    }
}

@Composable
fun ControlButtons(
    modelState: ModelState,
    taskState: TaskState,
    onLoadModel: () -> Unit,
    onStartTask: () -> Unit,
    onStopTask: () -> Unit,
    onResetTask: () -> Unit
) {
    Surface(
        modifier = Modifier.fillMaxWidth(),
        color = MaterialTheme.colorScheme.surface,
        shadowElevation = 4.dp
    ) {
        Row(
            modifier = Modifier.padding(12.dp),
            horizontalArrangement = Arrangement.spacedBy(8.dp)
        ) {
            Button(
                onClick = onLoadModel,
                enabled = modelState is ModelState.Idle || modelState is ModelState.Error
            ) {
                Text("加载模型")
            }

            Button(
                onClick = onStartTask,
                enabled = modelState is ModelState.Ready && taskState is TaskState.Idle
            ) {
                Text("开始任务")
            }

            Button(
                onClick = onStopTask,
                enabled = taskState is TaskState.Running,
                colors = ButtonDefaults.buttonColors(
                    containerColor = MaterialTheme.colorScheme.error
                )
            ) {
                Text("停止")
            }

            Button(
                onClick = onResetTask,
                enabled = taskState !is TaskState.Running
            ) {
                Text("重置")
            }
        }
    }
}
