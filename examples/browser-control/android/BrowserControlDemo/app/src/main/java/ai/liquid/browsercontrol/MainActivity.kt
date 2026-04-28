package ai.liquid.browsercontrol

import ai.liquid.browsercontrol.ui.BrowserControlScreen
import ai.liquid.browsercontrol.viewmodel.BrowserViewModel
import android.content.Intent
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Environment
import android.provider.Settings
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.material3.MaterialTheme
import androidx.lifecycle.viewmodel.compose.viewModel

/**
 * MainActivity - Browser Control Demo 入口
 *
 * 阶段四: 完整UI实现
 * - BrowserControlScreen: WebView + 日志面板 + 控制按钮
 * - BrowserViewModel: 推理主循环
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Android 11+（API 30+）需要 MANAGE_EXTERNAL_STORAGE 才能读取 /sdcard 根目录文件
        requestManageExternalStorageIfNeeded()

        setContent {
            MaterialTheme {
                val viewModel: BrowserViewModel = viewModel()
                BrowserControlScreen(viewModel)
            }
        }
    }

    private fun requestManageExternalStorageIfNeeded() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.R) {
            if (!Environment.isExternalStorageManager()) {
                Log.i("MainActivity", "请求 MANAGE_EXTERNAL_STORAGE 权限")
                try {
                    val intent = Intent(Settings.ACTION_MANAGE_APP_ALL_FILES_ACCESS_PERMISSION).apply {
                        data = Uri.parse("package:$packageName")
                    }
                    startActivity(intent)
                } catch (e: Exception) {
                    // 部分设备不支持精确跳转，退回到通用页面
                    Log.w("MainActivity", "无法直接跳转权限页，尝试通用页面: ${e.message}")
                    startActivity(Intent(Settings.ACTION_MANAGE_ALL_FILES_ACCESS_PERMISSION))
                }
            }
        }
    }
}
