import argparse
import logging
import socket
import sys
from typing import Literal

logger = logging.getLogger("utils")


class Args:
    debug: bool
    log_level: Literal["debug", "info", "warning", "error"]


def argparse_setup(description: str, logger: logging.Logger) -> Args:
    """Basic arguments and console logging setup."""

    args = Args()

    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode: process only a few samples",
    )
    parser.add_argument(
        "--log_level",
        type=str.lower,
        default="debug",
        choices=["debug", "info", "warning", "error"],
        help="Logging level.",
    )
    parser.parse_args(namespace=args)

    log_level = args.log_level.upper()

    # Setup logging
    logger.setLevel(log_level)
    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(logging.Formatter("{asctime} {levelname} {name:>14}│ {message}", datefmt="%H:%M:%S", style="{"))
    logger.addHandler(ch)

    # Add colors if stdout is not piped
    if sys.stdout.isatty():
        _m = logging.getLevelNamesMapping()
        for c, lvl in zip([226, 148, 208, 197], ["DEBUG", "INFO", "WARNING", "ERROR"]):
            logging.addLevelName(_m[lvl], f"\x1b[38;5;{c}m{lvl:<7}\x1b[0m")
    else:
        _m = logging.getLevelNamesMapping()
        for lvl in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            logging.addLevelName(_m[lvl], f"{lvl:<7}")

    return args


def is_port_in_use(port: int, host: str = "127.0.0.1") -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex((host, port)) == 0


def find_available_port(preferred_port: int | None, host: str = "127.0.0.1") -> int:
    if preferred_port is not None and not is_port_in_use(preferred_port, host):
        return preferred_port

    # Fallback: let OS pick a port
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind((host, 0))
            return s.getsockname()[1]
    except socket.error as e:
        raise RuntimeError(f"Could not find an available port: {e}")
