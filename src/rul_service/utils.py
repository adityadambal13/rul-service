"""Small helpers shared across the package."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from functools import wraps

logger = logging.getLogger("rul_service")


def timed[**P, R](func: Callable[P, R]) -> Callable[P, R]:
    """Decorator: log how long the wrapped function took.

    `[**P, R]` is Python 3.12 syntax that tells the type checker the wrapper keeps the
    same arguments and return type as the function it wraps.
    """

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            elapsed_ms = (time.perf_counter() - start) * 1000
            logger.info("timing %s took %.1f ms", func.__qualname__, elapsed_ms)

    return wrapper
