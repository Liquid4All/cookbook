package ai.liquid.browsercontrol.ui

import android.graphics.Color as AndroidColor
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.viewinterop.AndroidView

@Composable
fun WebViewCompose(
    url: String,
    modifier: Modifier = Modifier,
    onWebViewCreated: (WebView) -> Unit = {}
) {
    AndroidView(
        modifier = modifier,
        factory = { context ->
            WebView(context).apply {
                layoutParams = ViewGroup.LayoutParams(
                    ViewGroup.LayoutParams.MATCH_PARENT,
                    ViewGroup.LayoutParams.MATCH_PARENT
                )
                // MiniWoB 内容 210px 高度缩放后可能不填满 WebView 框，
                // 设置与日志区同色避免出现空白灰色区域
                setBackgroundColor(AndroidColor.parseColor("#1E1E1E"))

                settings.apply {
                    javaScriptEnabled = true
                    domStorageEnabled = true
                    // MiniWoB viewport width = 160px，useWideViewPort + loadWithOverview
                    // 让 WebView 原生把 160px 内容缩放铺满屏幕，不影响点击坐标
                    useWideViewPort = true
                    loadWithOverviewMode = true
                    setSupportZoom(false)
                    builtInZoomControls = false
                    displayZoomControls = false
                    textZoom = 100
                }

                setInitialScale(0)

                webViewClient = object : WebViewClient() {
                    override fun onPageFinished(view: WebView, url: String) {
                        // 注入/覆盖 viewport meta，强制声明页面内容宽度 = 160px
                        // 只执行一次（通过 data-vp-set 标记），避免 MiniWoB 多次触发
                        // onPageFinished 时重复叠加
                        view.evaluateJavascript("""
                            (function() {
                                if (document.documentElement.dataset.vpSet) return;
                                document.documentElement.dataset.vpSet = '1';
                                var meta = document.querySelector('meta[name="viewport"]');
                                if (!meta) {
                                    meta = document.createElement('meta');
                                    meta.name = 'viewport';
                                    document.head.appendChild(meta);
                                }
                                meta.content = 'width=160';
                            })();
                        """.trimIndent(), null)
                    }
                }

                loadUrl(url)
                onWebViewCreated(this)
            }
        }
    )
}
