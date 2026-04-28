# BrowserGym环境容器
# 提供本地MiniWoB任务API服务（使用OpenEnv服务器）

FROM python:3.11-slim

ARG http_proxy
ARG https_proxy
ARG HTTP_PROXY
ARG HTTPS_PROXY

ENV DEBIAN_FRONTEND=noninteractive \
    DISPLAY=:99 \
    PYTHONUNBUFFERED=1 \
    BROWSERGYM_BENCHMARK=miniwob \
    BROWSERGYM_HEADLESS=true \
    BROWSERGYM_PORT=8000 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

# 安装系统依赖（Chromium + Playwright + Xvfb + 字体）
RUN if [ -n "${http_proxy}" ]; then \
      echo "Acquire::http::Proxy \"${http_proxy}\";" > /etc/apt/apt.conf.d/99proxy && \
      echo "Acquire::https::Proxy \"${https_proxy:-${http_proxy}}\";" >> /etc/apt/apt.conf.d/99proxy; \
    fi && \
    apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    ca-certificates \
    chromium \
    chromium-driver \
    xvfb \
    x11vnc \
    fluxbox \
    fonts-unifont \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto-core \
    && rm -rf /var/lib/apt/lists/* /etc/apt/apt.conf.d/99proxy

WORKDIR /app

# 分步安装BrowserGym依赖，避免一次性下载太多包
RUN if [ -n "${http_proxy}" ]; then PROXY_ARG="--proxy ${http_proxy}"; else PROXY_ARG=""; fi && \
    pip install --no-cache-dir --default-timeout=300 $PROXY_ARG \
    gymnasium==1.0.0 \
    beautifulsoup4==4.12.3 \
    Pillow==11.1.0 \
    regex==2024.11.6

# 安装BrowserGym核心（不带所有可选依赖）
RUN if [ -n "${http_proxy}" ]; then PROXY_ARG="--proxy ${http_proxy}"; else PROXY_ARG=""; fi && \
    pip install --no-cache-dir --default-timeout=300 $PROXY_ARG browsergym-core==0.14.2

# Patch browsergym-core：将全局 Playwright 单例改为线程本地存储
# 原实现用一个全局 _PLAYWRIGHT 变量，导致 sync_playwright 实例绑定的 event loop
# 与 FastAPI ThreadPoolExecutor 的工作线程不匹配，触发 "no running event loop" 错误。
# 方案四：使用 threading.local() 使每个线程持有独立的 Playwright 实例。
RUN python3 - <<'EOF'
import pathlib, textwrap
target = pathlib.Path("/usr/local/lib/python3.11/site-packages/browsergym/core/__init__.py")
new_content = textwrap.dedent("""
__version__ = "0.14.2"

import threading
import playwright.sync_api

# Thread-local storage: each worker thread owns its own Playwright instance.
# This avoids the "no running event loop" RuntimeError that occurs when
# playwright's sync API (which internally binds to an asyncio event loop)
# is shared across threads inside FastAPI's ThreadPoolExecutor.
_PLAYWRIGHT_LOCAL = threading.local()


def _set_global_playwright(pw: playwright.sync_api.Playwright):
    # Legacy helper kept for compatibility; sets on the calling thread.
    _PLAYWRIGHT_LOCAL.instance = pw


def _get_global_playwright():
    if not getattr(_PLAYWRIGHT_LOCAL, 'instance', None):
        pw = playwright.sync_api.sync_playwright().start()
        _PLAYWRIGHT_LOCAL.instance = pw
    return _PLAYWRIGHT_LOCAL.instance


# register the open-ended task
from .registration import register_task
from .task import OpenEndedTask

register_task(OpenEndedTask.get_task_id(), OpenEndedTask)
""").lstrip()
target.write_text(new_content)
print("Patched", target)
EOF

# 安装MiniWoB支持
RUN if [ -n "${http_proxy}" ]; then PROXY_ARG="--proxy ${http_proxy}"; else PROXY_ARG=""; fi && \
    pip install --no-cache-dir --default-timeout=300 $PROXY_ARG browsergym-miniwob==0.14.2

# 安装MiniWoB++任务集
RUN git config --global http.proxy ${http_proxy:-} && \
    git config --global https.proxy ${https_proxy:-} && \
    git clone --depth 1 https://github.com/Farama-Foundation/miniwob-plusplus.git && \
    cd miniwob-plusplus && \
    pip install --no-cache-dir -e . && \
    git config --global --unset http.proxy || true && \
    git config --global --unset https.proxy || true

# 复制并安装OpenEnv（本地copy，避免网络问题）
COPY OpenEnv /app/OpenEnv
WORKDIR /app/OpenEnv

# 通过 pip install 安装 OpenEnv（自动解析所有依赖，包括 fastmcp 和 playwright 正确版本）
RUN if [ -n "${http_proxy}" ]; then PROXY_ARG="--proxy ${http_proxy}"; else PROXY_ARG=""; fi && \
    pip install --no-cache-dir --default-timeout=300 $PROXY_ARG -e ".[core]"

# 安装与当前 playwright 版本匹配的 Chromium（playwright install 会检测已安装版本）
RUN if [ -n "${http_proxy}" ]; then \
      HTTPS_PROXY=${http_proxy} playwright install chromium; \
    else \
      playwright install chromium; \
    fi

# 手动设置Python路径（同时包含 envs 目录）
ENV PYTHONPATH="/app/OpenEnv/src:/app/OpenEnv/envs:${PYTHONPATH}"

# 回到app目录
WORKDIR /app

# 复制启动脚本
COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

# 暴露端口（8000: OpenEnv API, 9000: MiniWoB 静态文件服务）
EXPOSE 8000 9000

# 健康检查
HEALTHCHECK --interval=10s --timeout=5s --start-period=30s --retries=5 \
    CMD curl -f http://localhost:8000/health || exit 1

# 启动命令
CMD ["/app/start.sh"]
