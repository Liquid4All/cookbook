FROM python:3.11-slim

# 克隆 miniwob-plusplus，只保留 html/ 静态文件
# Android WebView 充当浏览器，无需安装 chromium
RUN apt-get update && apt-get install -y --no-install-recommends git \
    && git clone --depth=1 https://github.com/Farama-Foundation/miniwob-plusplus.git /tmp/miniwob \
    && mkdir -p /app \
    && cp -r /tmp/miniwob/miniwob/html /app/html \
    && rm -rf /tmp/miniwob /var/lib/apt/lists/*

COPY server.py /app/server.py

EXPOSE 8080

HEALTHCHECK --interval=10s --timeout=3s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/miniwob/click-button.html')" \
    || exit 1

CMD ["python", "/app/server.py", "/app/html"]
