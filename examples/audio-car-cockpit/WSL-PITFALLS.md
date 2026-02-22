# WSL2 Pitfalls

Known issues when running the audio car cockpit demo under Windows Subsystem for Linux 2 (WSL2).

> **Tip:** Run `bash check_prerequisites.sh` before `make setup` to automatically detect most of these issues.

## 1. `uv` not on PATH after auto-install

The Makefile auto-installs `uv` via `curl` if it's missing, but the newly installed binary at `~/.local/bin/uv` isn't available to the current Make process. The subsequent `uv venv` command fails with:

```
make: uv: No such file or directory
```

**Fix:** Source the env file or add `~/.local/bin` to your PATH, then re-run:

```bash
export PATH="$HOME/.local/bin:$PATH"
make setup
```

To make it permanent, add the export to `~/.bashrc`:

```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
```

## 2. IPv6 breaks network connections (huggingface_hub and llama-server hang)

WSL2 advertises IPv6 support but often cannot route IPv6 traffic. This affects **both** Python (`huggingface_hub`) and native C++ (`llama-server -hf`) since both try IPv6 first. Symptoms:

- `hf download` commands produce no output and never complete
- `make LFM2.5-Audio-1.5B-GGUF` hangs silently
- `llama-server -hf` gets stuck in SYN-SENT on an IPv6 address, blocking `make serve` from ever starting
- `curl` works fine (it falls back to IPv4 faster)

You can verify the issue:

```bash
# This will show IPv6 timing out
uv run --with "huggingface_hub" python -c "
import socket
s = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
s.settimeout(5)
addrs = socket.getaddrinfo('huggingface.co', 443, socket.AF_INET6)
try:
    s.connect(addrs[0][4])
    print('IPv6 OK')
except Exception as e:
    print('IPv6 FAILED:', e)
s.close()
"
```

**Fix:** Disable IPv6 in WSL2:

```bash
sudo sysctl -w net.ipv6.conf.all.disable_ipv6=1 \
               net.ipv6.conf.default.disable_ipv6=1 \
               net.ipv6.conf.lo.disable_ipv6=1
```

**Important:** The `sysctl` settings do not persist across WSL restarts. To make them permanent, add to `/etc/sysctl.conf`:

```bash
sudo tee -a /etc/sysctl.conf <<EOF
net.ipv6.conf.all.disable_ipv6 = 1
net.ipv6.conf.default.disable_ipv6 = 1
net.ipv6.conf.lo.disable_ipv6 = 1
EOF
```

## 3. `llama-server` built without SSL support

The cmake build may fail to find OpenSSL even when `libcurl4-openssl-dev` is installed, because `libssl-dev` is missing. The build silently falls back to no-SSL mode. This causes:

```
'https' scheme is not supported.
```

when `llama-server -hf` tries to download a model over HTTPS. The `make LFM2-1.2B-Tool-GGUF` target appears to succeed because it only runs `--version`, which doesn't need HTTPS.

**Fix:** Install OpenSSL development headers, then rebuild:

```bash
sudo apt install -y libssl-dev
rm -f llama-server
make llama-server
```

Verify the cmake output includes `OpenSSL found` and no longer says `SSL support disabled`.

## 4. Audio server port 8142 shows HTTP 404 in browser

This is expected behavior, not a bug. The audio server (`llama-liquid-audio-server`) on port 8142 is an API-only server with no web UI. The web interface is served by the FastAPI app on port **8000**.

Open http://127.0.0.1:8000 — not :8142.

## 5. Inference warmup times out on CPU

The FastAPI server runs a warmup inference call (`"Turn on the audio."`) during startup. The default 30-second timeout is too short for CPU-only inference of the 1.2B tool-calling model, especially on the first run. The server crashes with:

```
httpx.ReadTimeout: timed out
ERROR:    Application startup failed. Exiting.
```

**Fix:** Increase the timeout in `src/llamacpp_inference.py`. The `_completion` and `_completion_stream` timeouts should be raised from 30s to 300s for CPU use. The `_apply_template` timeout should also be bumped from 3s to 30s.
