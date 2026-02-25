import ast
import json
import signal
import subprocess
from collections.abc import Generator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import AsyncGenerator, overload

import httpx
from httpx_retries import Retry, RetryTransport

from src.utils import find_available_port


def _get_func_name(node: ast.expr):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return _get_func_name(node.value) + "." + node.attr
    raise ValueError(f"Unsupported func node: {ast.dump(node)}")


def function_to_args(call_str: str) -> tuple[str, dict]:
    """Transform a python-like function string to name+args.

    Example:
    some.functionName(arg1=False, arg2=42.0)
    ->
    ("some.functionName", {"arg1": False, "arg2": 42.0})
    """

    tree = ast.parse(call_str, mode="eval")
    call = tree.body

    if not isinstance(call, ast.Call):
        raise ValueError("Not a function call expression")

    func_name = _get_func_name(call.func)

    args = {}
    # only handle keyword args here
    for kw in call.keywords:
        key = kw.arg
        try:
            value = ast.literal_eval(kw.value)
        except (ValueError, TypeError):
            # non-literal value → fall back to source text
            value = ast.unparse(kw.value)
        args[key] = value

    return func_name, args


def spawn_embedding_runtime(
    file_name: str | Path,
) -> tuple[subprocess.Popen, int]:
    """Spawns the llama-server process."""

    port = find_available_port(preferred_port=8989)
    host = "127.0.0.1"
    executable = str((Path.cwd() / "llama-server").resolve())

    command = [
        executable,
        "--host",
        host,
        "--port",
        str(port),
        "--log-disable",
        "--no-perf",
        "--n-gpu-layers",
        "9999",
        "--mlock",
        "--ctx-size",
        "2048",
        # Special tokens output enabled
        "--special",
        "-hf",
        str(file_name),
    ]

    try:
        # Use Popen for non-blocking execution
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)

    except FileNotFoundError:
        raise FileNotFoundError(f"{executable} command not found. Please ensure it is installed and in your PATH.")

    # Wait for server to be ready
    print(f"Waiting for {executable} to start, serving {file_name}...")
    # https://github.com/ggml-org/llama.cpp/tree/master/tools/server#api-endpoints
    health_url = f"http://{host}:{port}/health"
    _tries, _sleep = 300, 0.5
    for _ in range(_tries):
        try:
            r = httpx.get(health_url, timeout=1.0)
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        sleep(_sleep)
    else:
        process.terminate()
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"{executable} failed to become healthy after {_tries * _sleep:.1f}s.\n"
            f"Stdout: {stdout.decode(errors='ignore')}\n"
            f"Stderr: {stderr.decode(errors='ignore')}"
        )

    if process.poll() is not None:  # Process terminated prematurely
        stdout, stderr = process.communicate()
        raise RuntimeError(
            f"{executable} failed to start. Return code: {process.returncode}\n"
            f"Stdout: {stdout.decode(errors='ignore')}\n"
            f"Stderr: {stderr.decode(errors='ignore')}"
        )

    print(f"`{executable}` running on port {port} (PID: {process.pid})")
    return process, port


@contextmanager
def spawn_server(
    file_name: str | Path,
) -> Generator[tuple[subprocess.Popen, int], None, None]:
    embedding_process, embedding_port = spawn_embedding_runtime(file_name)
    try:
        yield embedding_process, embedding_port
    finally:
        embedding_process.terminate()
        try:
            # Wait for graceful shutdown
            embedding_process.wait(timeout=4)
            print("llama-server terminated gracefully.")
        except subprocess.TimeoutExpired:
            try:
                # One more chance
                embedding_process.send_signal(signal.SIGTERM)
                embedding_process.wait(timeout=4)
            except subprocess.TimeoutExpired:
                print("llama-server did not terminate in time, killing...")
                embedding_process.kill()
                embedding_process.wait()


@dataclass(kw_only=True)
class ToolCallingRuntime:
    port: int
    host: str = "localhost"
    max_tokens: int = 4096

    def __post_init__(self):
        self.client = httpx.Client(transport=RetryTransport(retry=Retry(total=3, backoff_factor=0.1)))

        self.default_completion_params: dict[str, float | int | bool] = {
            "temperature": 0.0,
            "n_predict": 512,
        }

        path_functions_def: Path = Path(__file__).parent.parent / "functions.json"
        assert path_functions_def.exists(), f"Function definition file not found: {path_functions_def}"

        self.list_functions: list[dict] = json.loads(path_functions_def.read_bytes())["functions"]

        # Prepare as a string with a flat layout
        self.all_functions_no_indent: str = json.dumps(self.list_functions, indent=2, ensure_ascii=False)

        _instructions = (
            """If you call a function, also output a brief message for the user. The message should be concise."""
        )

        self.system_prompt = f"""List of tools:

<|tool_list_start|>{self.all_functions_no_indent}<|tool_list_end|>

{_instructions}"""

        self._last_messages: list[dict] = []

        # Warmup
        print("Inference warming...", end=" ")
        _ = self.completion("Turn on the audio.")
        print("Done")

    def __del__(self):
        self.client.close()

    def _apply_template(self, content: str) -> str:
        response = self.client.post(
            f"http://{self.host}:{self.port}/apply-template",
            json={
                "messages": [
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": content},
                ]
            },
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        response.raise_for_status()
        formatted_prompt: str = response.json().get("prompt")
        return formatted_prompt

    def _completion(self, content: str) -> tuple[str | None, str]:
        formatted_prompt = self._apply_template(content)

        response = self.client.post(
            f"http://{self.host}:{self.port}/completion",
            json=self.default_completion_params
            | {
                "prompt": formatted_prompt,
            },
            headers={"Content-Type": "application/json"},
            timeout=300.0,
        )
        response.raise_for_status()
        j = response.json()
        rez: str = j.get("content")

        # Separate tool call and response
        if "<|tool_call_start|>" not in rez:
            rez = rez.rstrip("<|im_end|>")
            return None, rez

        rez = rez.split("<|tool_call_start|>", maxsplit=1)[1]
        tool_call, text = rez.split("<|tool_call_end|>", maxsplit=1)
        tool_call = tool_call.lstrip("[").rstrip("]")
        text = text.rstrip("<|im_end|>")

        return tool_call, text

    async def _completion_stream(self, content: str) -> AsyncGenerator[str, None]:
        """SSE response

        https://html.spec.whatwg.org/multipage/server-sent-events.html
        """

        formatted_prompt = self._apply_template(content)

        async with httpx.AsyncClient() as aclient:
            async with aclient.stream(
                "post",
                f"http://{self.host}:{self.port}/completion",
                json=self.default_completion_params
                | {
                    "prompt": formatted_prompt,
                    "stream": True,
                },
                headers={"Content-Type": "application/json"},
                timeout=300.0,
            ) as r:
                async for x in r.aiter_text():
                    yield x

    @overload
    def completion(
        self,
        content: str,
    ) -> tuple[str | None, str]: ...

    @overload
    def completion(self, content: str, stream: bool = True) -> AsyncGenerator[str, None]: ...

    def completion(self, content: str, stream: bool = False) -> tuple[str | None, str] | AsyncGenerator[str, None]:
        try:
            if stream:
                return self._completion_stream(content)
            else:
                return self._completion(content)
        except httpx.HTTPStatusError as e:
            print(f"Failed on:\n{content}\n{e}")
            raise
