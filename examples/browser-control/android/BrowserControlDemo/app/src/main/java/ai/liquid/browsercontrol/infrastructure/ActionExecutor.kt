package ai.liquid.browsercontrol.infrastructure

import android.webkit.WebView
import ai.liquid.browsercontrol.domain.BrowserAction
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

object ActionExecutor {

    suspend fun execute(webView: WebView, action: BrowserAction): Result<String> {
        val script = when (action) {
            is BrowserAction.Noop -> {
                return Result.success("Noop action executed")
            }

            is BrowserAction.Click -> """
                (function() {
                    var element = document.querySelector('[data-bid="${action.bid}"]');
                    if (element) {
                        element.click();
                        return 'Clicked element ${action.bid}';
                    } else {
                        return 'Error: Element ${action.bid} not found';
                    }
                })();
            """.trimIndent()

            is BrowserAction.Fill -> """
                (function() {
                    var element = document.querySelector('[data-bid="${action.bid}"]');
                    if (element && (element.tagName === 'INPUT' || element.tagName === 'TEXTAREA')) {
                        element.value = '${action.text.replace("'", "\\'")}';
                        element.dispatchEvent(new Event('input', { bubbles: true }));
                        return 'Filled element ${action.bid}';
                    } else {
                        return 'Error: Input element ${action.bid} not found';
                    }
                })();
            """.trimIndent()

            is BrowserAction.SendKeys -> """
                (function() {
                    var activeElement = document.activeElement;
                    if (activeElement && (activeElement.tagName === 'INPUT' || activeElement.tagName === 'TEXTAREA')) {
                        activeElement.value += '${action.text.replace("'", "\\'")}';
                        activeElement.dispatchEvent(new Event('input', { bubbles: true }));
                        return 'Sent keys: ${action.text}';
                    } else {
                        return 'Error: No active input element';
                    }
                })();
            """.trimIndent()

            is BrowserAction.Scroll -> """
                (function() {
                    var scrollAmount = '${action.direction}' === 'down' ? 200 : -200;
                    window.scrollBy(0, scrollAmount);
                    return 'Scrolled ${action.direction}';
                })();
            """.trimIndent()
        }

        return suspendCoroutine { continuation ->
            webView.evaluateJavascript(script) { result ->
                val message = result?.trim('"') ?: "Unknown result"
                if (message.startsWith("Error:")) {
                    continuation.resume(Result.failure(Exception(message)))
                } else {
                    continuation.resume(Result.success(message))
                }
            }
        }
    }
}
