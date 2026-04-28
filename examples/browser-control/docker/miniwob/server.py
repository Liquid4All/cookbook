"""MiniWoB++ 本地 HTTP 服务，将 miniwob/html/ 暴露给 Android WebView 访问。

用法:
    python server.py [html_dir] [--port PORT]

html_dir 默认为 /app/html（Docker 容器内路径）。
本地开发可指定 miniwob-plusplus/miniwob/html/ 的实际路径：
    python server.py /path/to/miniwob-plusplus/miniwob/html

调试功能：
    所有 .html 页面加载后会自动在浏览器 console 输出 Android 格式 AXTree。
    在 Android emulator / Chrome DevTools 的 console 中可看到：
      [AXTree] [1] body
      [AXTree]   [2] div id="wrap"
      [AXTree]     [3] div 'Click the button.' id="query"
      ...
    延迟 500ms 确保 MiniWoB core.js 完成任务初始化后再提取。
"""

import functools
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


# 注入到每个 .html 页面 </body> 前的 AXTree 调试脚本
# 与 Android WebViewAccessibility.kt 的 extractScript 逻辑完全一致：
#   - DOM 前序遍历，跳过 SCRIPT/STYLE
#   - bid 从 1 开始自增，写入 data-bid 属性
#   - 格式：[bid] tagName 'directText' id="..." class="..."
#   - 每行缩进 2 空格 × depth
#   - 延迟 500ms 等待 MiniWoB core.js 完成任务初始化（goal 文字注入）
_AXTREE_DEBUG_SCRIPT = """
<script>
(function() {
  function extractAndLog() {
    var bid = 0;
    var lines = [];
    function traverse(node, indent) {
      if (!node) return;
      var name = node.nodeName;
      if (name === 'SCRIPT' || name === 'STYLE') return;
      bid++;
      var myBid = bid;
      node.setAttribute('bid', String(myBid));
      var tag = node.tagName ? node.tagName.toLowerCase() : '#text';
      var directText = '';
      for (var i = 0; i < node.childNodes.length; i++) {
        var ch = node.childNodes[i];
        if (ch.nodeType === 3) {
          var t = ch.textContent.trim();
          if (t) directText += t + ' ';
        }
      }
      directText = directText.trim().slice(0, 100);
      var prefix = '';
      for (var k = 0; k < indent; k++) prefix += '  ';
      var line = prefix + '[' + myBid + '] ' + tag;
      if (directText) line += " '" + directText + "'";
      if (node.id) line += ' id="' + node.id + '"';
      if (node.className) line += ' class="' + node.className + '"';
      lines.push(line);
      for (var j = 0; j < node.children.length; j++) {
        traverse(node.children[j], indent + 1);
      }
    }
    traverse(document.body, 0);
    console.log('[AXTree] === Android-format AXTree (500ms after load) ===');
    for (var i = 0; i < lines.length; i++) {
      console.log('[AXTree] ' + lines[i]);
    }
    console.log('[AXTree] === end ===');
  }
  window.addEventListener('load', function() {
    setTimeout(extractAndLog, 500);
  });
})();
</script>
</body>"""


class CORSRequestHandler(SimpleHTTPRequestHandler):
    """添加 CORS 响应头，支持 Android WebView 跨域请求。
    对 .html 文件动态注入 AXTree 调试脚本（console 输出）。
    """

    def end_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS, HEAD")
        self.send_header("Cache-Control", "no-cache, no-store, must-revalidate")
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

    def do_GET(self):
        """对 .html 文件注入 AXTree 调试脚本；其他文件原样提供。"""
        path = self.path.split("?")[0].split("#")[0]
        if path.endswith(".html"):
            self._serve_html_with_injection()
        else:
            super().do_GET()

    def _serve_html_with_injection(self):
        """读取 .html 文件，将 </body> 替换为注入脚本 + </body>，返回给客户端。"""
        # 解析本地文件路径
        fs_path = self.translate_path(self.path)
        try:
            with open(fs_path, "rb") as f:
                content = f.read()
        except OSError:
            self.send_error(404, "File not found")
            return

        # 注入：将最后一个 </body> 替换为 <script>...</script></body>
        script_bytes = _AXTREE_DEBUG_SCRIPT.encode("utf-8")
        lower = content.lower()
        idx = lower.rfind(b"</body>")
        if idx != -1:
            content = content[:idx] + script_bytes
        else:
            # 没有 </body>，直接追加
            content = content + script_bytes

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(content)))
        self.end_headers()
        self.wfile.write(content)

    def log_message(self, format, *args):
        # 只打印非 200 请求，减少噪音
        if args and str(args[1]) != "200":
            super().log_message(format, *args)


def main():
    html_dir = sys.argv[1] if len(sys.argv) > 1 else "/app/html"
    port = int(os.environ.get("PORT", 8080))

    if not os.path.isdir(html_dir):
        print(f"ERROR: html_dir '{html_dir}' does not exist", file=sys.stderr)
        print(
            "  For Docker: html/ is copied to /app/html at build time.", file=sys.stderr
        )
        print(
            "  For local dev: python server.py /path/to/miniwob-plusplus/miniwob/html",
            file=sys.stderr,
        )
        sys.exit(1)

    handler = functools.partial(CORSRequestHandler, directory=html_dir)
    with ThreadingHTTPServer(("0.0.0.0", port), handler) as httpd:
        print(f"✓ MiniWoB++ server running:")
        print(f"  http://0.0.0.0:{port}/miniwob/click-button.html")
        print(f"  Serving from: {html_dir}")
        print(
            f"  Android emulator URL: http://10.0.2.2:{port}/miniwob/click-button.html"
        )
        print(
            "  AXTree debug: open DevTools console to see [AXTree] output after page load"
        )
        print("  Press Ctrl+C to stop.")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
