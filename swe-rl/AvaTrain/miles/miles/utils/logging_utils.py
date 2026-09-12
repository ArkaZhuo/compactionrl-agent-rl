import logging
import os
import re
import sys
import warnings

_LOGGER_CONFIGURED = False

logger = logging.getLogger(__name__)

_FATAL_ASYNC_PATTERN = "coroutine .* was never awaited"


# ref: SGLang
def configure_logger(prefix: str = ""):
    global _LOGGER_CONFIGURED
    if _LOGGER_CONFIGURED:
        return

    _LOGGER_CONFIGURED = True

    logging.basicConfig(
        level=logging.INFO,
        format=f"[%(asctime)s{prefix}] %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        force=True,
    )

    configure_strict_async_warnings()


def actor_log_path(filename: str) -> str | None:
    log_dir = os.environ.get("LOG_DIR")
    if not log_dir:
        return None
    return os.path.join(log_dir, "actors", filename)


def redirect_process_output(log_path: str | None) -> None:
    if not log_path:
        return

    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    sys.stdout.flush()
    sys.stderr.flush()

    fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        os.dup2(fd, 1)
        os.dup2(fd, 2)
    finally:
        os.close(fd)

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(line_buffering=True)


def configure_strict_async_warnings() -> None:
    """Turn unawaited-coroutine warnings into fatal errors.

    Python emits RuntimeWarning when a coroutine is called but never awaited.
    The warning fires inside __del__, so the resulting exception is swallowed
    by sys.unraisablehook. We override the hook to hard-exit the process.
    """
    warnings.filterwarnings("error", category=RuntimeWarning, message=_FATAL_ASYNC_PATTERN)

    _original_hook = sys.unraisablehook

    def _crash_on_async_misuse(unraisable):
        if isinstance(unraisable.exc_value, RuntimeWarning) and re.search(
            _FATAL_ASYNC_PATTERN, str(unraisable.exc_value)
        ):
            msg = f"Fatal async misuse, aborting: {unraisable.exc_value}"
            logger.error(msg)
            print(msg, file=sys.stderr, flush=True)
            os._exit(1)
        _original_hook(unraisable)

    sys.unraisablehook = _crash_on_async_misuse
