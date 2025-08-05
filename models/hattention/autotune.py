import os
import time
import triton
import inspect
from typing import List, Callable, Any


def autotune(
    configs: List[triton.Config],
    key: List[str],
) -> Callable:

    def decorator(fn: Callable) -> Callable:
        signature = inspect.signature(fn)
        arg_names = list(signature.parameters.keys())

        autotuner = triton.runtime.autotuner.Autotuner(
            RunnableFunction(fn),
            arg_names,
            configs,
            key,
            reset_to_zero=None,
            restore_value=None)

        def _run(*args, **kwargs) -> Any:
            return autotuner.run(*args, **kwargs)

        return _run

    return decorator


class RunnableFunction(object):

    _KWARGS_TO_REMOVE = [
        "num_warps",
        "num_ctas",
        "num_stages",
        "num_buffers_warp_spec",
        "num_consumer_groups",
        "reg_dec_producer",
        "reg_inc_consumer",
    ]

    def __init__(self, fn: Callable) -> None:
        self.fn = fn

    def run(self, *args, **kwargs) -> Any:
        kwargs = {k: v for k, v in kwargs.items() if k not in self._KWARGS_TO_REMOVE}
        return self.fn(*args, **kwargs)


def test_autotune() -> None:
    os.environ["TRITON_PRINT_AUTOTUNING"] = "1"

    @autotune(
        configs=[
            triton.Config({"guess": 16}),
            triton.Config({"guess": 32}),
            triton.Config({"guess": 64}),
        ],
        key=["value"],
    )
    def simple_function(value: int, guess: int) -> int:
        # If values are equal, return immediately
        if value == guess:
            return 0

        # Otherwise sleep for configured time
        time.sleep(0.1)
        return abs(value - guess)

    simple_function(value=64)
    simple_function(value=16)
    simple_function(value=32)
