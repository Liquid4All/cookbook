"""HuggingFace downloader with IPv4 forced.

Drop-in replacement for 'hf download' that patches socket to use IPv4 only,
working around WSL2 environments where IPv6 is advertised but not routable
(causing huggingface_hub downloads to hang silently).

Usage:
    python hf_download.py <repo_id> [--local-dir DIR] [--include PATTERN...]

When --local-dir is omitted, downloads to the default HuggingFace cache
(~/.cache/huggingface/hub/), which llama-server's -hf flag checks before
attempting a network download.
"""
import argparse
import socket

# Patch socket before any network-touching imports
_orig_getaddrinfo = socket.getaddrinfo
socket.getaddrinfo = lambda h, p, f=0, t=0, pr=0, fl=0: _orig_getaddrinfo(
    h, p, socket.AF_INET, t, pr, fl
)

from huggingface_hub import snapshot_download  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("repo_id")
parser.add_argument("--local-dir", default=None)
parser.add_argument("--include", nargs="*")
args = parser.parse_args()

snapshot_download(
    repo_id=args.repo_id,
    local_dir=args.local_dir,
    allow_patterns=args.include,
)
