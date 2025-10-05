"""Timeout decorator for functions that may hang."""

import functools
import signal
from typing import Never


def timeout(seconds, default=None, verbose=False):
    """Decorator to apply a timeout to a function call.

    Example #1:
        @timeout(seconds=5, default=0.0)
        def my_function():
            # Some long-running operation
        return result

    Example #2:
        timeout(seconds=5, default=0.0)(my_function)(args, kwargs)
    """

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*a, **kw):
            def _raise(signum, frame) -> Never:
                raise TimeoutError

            old_handler = signal.signal(signal.SIGALRM, _raise)
            signal.setitimer(signal.ITIMER_REAL, seconds)
            try:
                return func(*a, **kw)
            except TimeoutError:
                if verbose:
                    print(f"Function timed out after {seconds} seconds")
                return default
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
                signal.signal(signal.SIGALRM, old_handler)

        return wrapper

    return decorator
