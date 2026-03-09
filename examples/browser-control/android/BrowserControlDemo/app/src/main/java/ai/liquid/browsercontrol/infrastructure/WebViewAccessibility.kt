package ai.liquid.browsercontrol.infrastructure

import android.webkit.WebView
import kotlin.coroutines.resume
import kotlin.coroutines.suspendCoroutine

object WebViewAccessibility {

    private val extractScript = """
        (function() {
            let tree = [];
            let bidCounter = 1;

            function traverse(node, indent) {
                indent = indent || 0;
                if (node.nodeType === Node.ELEMENT_NODE) {
                    if (node.tagName === 'SCRIPT' || node.tagName === 'STYLE') {
                        return;
                    }

                    node.setAttribute('data-bid', bidCounter.toString());

                    let prefix = '  '.repeat(indent);
                    let entry = prefix + '[' + bidCounter + '] ' + node.tagName.toLowerCase();

                    let directText = '';
                    for (let i = 0; i < node.childNodes.length; i++) {
                        let child = node.childNodes[i];
                        if (child.nodeType === Node.TEXT_NODE) {
                            directText += child.textContent.trim();
                        }
                    }
                    if (directText) {
                        entry += " '" + directText.substring(0, 100) + "'";
                    }

                    if (node.id) entry += ' id="' + node.id + '"';
                    if (node.className) entry += ' class="' + node.className + '"';

                    tree.push(entry);
                    bidCounter++;

                    for (let i = 0; i < node.children.length; i++) {
                        traverse(node.children[i], indent + 1);
                    }
                }
            }

            if (document.body) {
                traverse(document.body, 0);
            }

            return tree.join('\n');
        })();
    """.trimIndent()

    suspend fun extractAXTree(webView: WebView, maxLength: Int = 2000): String {
        return suspendCoroutine { continuation ->
            webView.evaluateJavascript(extractScript) { result ->
                val tree = result?.trim('"')?.replace("\\n", "\n") ?: ""
                val truncated = if (tree.length > maxLength) {
                    tree.substring(0, maxLength) + "\n..."
                } else {
                    tree
                }
                continuation.resume(truncated)
            }
        }
    }
}
