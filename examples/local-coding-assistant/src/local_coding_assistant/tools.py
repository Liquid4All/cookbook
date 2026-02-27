import os
import subprocess
from pathlib import Path

# Working directory used by run_bash. Set by the agent at startup.
_working_directory = "."


def set_working_directory(path: str) -> None:
    global _working_directory
    _working_directory = path


def read_file(path: str) -> str:
    """Read the contents of a file. Returns the file content as a string."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except FileNotFoundError:
        return f"[error] File not found: {path}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def write_file(path: str, content: str) -> str:
    """Write content to a file, creating parent directories if needed."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        return f"File written: {path}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def list_directory(path: str = ".") -> str:
    """List files and directories at the given path."""
    try:
        entries = sorted(Path(path).iterdir(), key=lambda e: (e.is_file(), e.name))
        lines = []
        for entry in entries:
            prefix = "  " if entry.is_file() else "/ "
            lines.append(f"{prefix}{entry.name}")
        return "\n".join(lines) if lines else "(empty directory)"
    except FileNotFoundError:
        return f"[error] Directory not found: {path}"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


def run_bash(command: str, timeout: int = 30) -> str:
    """Run a bash command and return combined stdout and stderr."""
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=_working_directory,
        )
        output = result.stdout
        if result.stderr:
            output += f"\n[stderr]\n{result.stderr}"
        if result.returncode != 0:
            output += f"\n[exit code: {result.returncode}]"
        return output.strip() or "(no output)"
    except subprocess.TimeoutExpired:
        return f"[error] Command timed out after {timeout}s"
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"


# JSON Schema definitions sent to the model
TOOLS: list[dict] = [
    {
        "name": "read_file",
        "description": "Read the contents of a file at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to read"},
            },
            "required": ["path"],
        },
    },
    {
        "name": "write_file",
        "description": "Write content to a file, creating it (and any parent directories) if it doesn't exist.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file to write"},
                "content": {"type": "string", "description": "Content to write to the file"},
            },
            "required": ["path", "content"],
        },
    },
    {
        "name": "list_directory",
        "description": "List files and directories at the given path.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Directory path to list (default: current directory)"},
            },
            "required": [],
        },
    },
    {
        "name": "run_bash",
        "description": "Run a bash command and return its output. Use this for searching (grep, find), running tests, git operations, and executing scripts.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The bash command to run"},
                "timeout": {"type": "integer", "description": "Timeout in seconds (default: 30)"},
            },
            "required": ["command"],
        },
    },
]

TOOL_FUNCTIONS: dict[str, object] = {
    "read_file": read_file,
    "write_file": write_file,
    "list_directory": list_directory,
    "run_bash": run_bash,
}


def execute_tool(name: str, inputs: dict) -> str:
    """Dispatch a tool call by name and return the string result."""
    fn = TOOL_FUNCTIONS.get(name)
    if fn is None:
        return f"[error] Unknown tool: {name}"
    try:
        return fn(**inputs)  # type: ignore[operator]
    except Exception as e:
        return f"[error] {type(e).__name__}: {e}"
