import asyncio
import subprocess
import sys
import uuid
from typing import Any, AsyncIterator, Coroutine, Tuple, Union

from typing_extensions import TypeVar


def get_cpu_cores_in_container() -> int:
    import multiprocessing

    cpu_count = multiprocessing.cpu_count()
    try:
        with open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us") as fp:
            cfs_quota_us = int(fp.read())
        with open("/sys/fs/cgroup/cpu/cpu.cfs_period_us") as fp:
            cfs_period_us = int(fp.read())
        if cfs_quota_us != -1:
            cpu_count = cfs_quota_us // cfs_period_us
    except FileNotFoundError:
        pass
    return cpu_count


def get_gpu_free_memory():  # pragma: no cover
    """Get GPU free memory using nvidia-smi."""
    try:
        output = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
        ).stdout
        gpu_memory = [int(x) for x in output.strip().split("\n")]
        return gpu_memory
    except Exception as e:
        print(f"Error getting GPU memory: {e}")
        return None


def check_unknown_startup_memory_usage():  # pragma: no cover
    """Check for unknown memory usage at startup."""
    gpu_free_memory = get_gpu_free_memory()
    if gpu_free_memory is not None:
        print(f"GPU free memory at startup in MB: {gpu_free_memory}")
        min_mem = min(gpu_free_memory)
        max_mem = max(gpu_free_memory)
        if max_mem - min_mem > 10:
            print(
                f"WARNING: Unbalanced GPU memory usage at start up. This may cause OOM. Memory usage per GPU in MB: {gpu_free_memory}."
            )
            try:
                output = subprocess.run(
                    ["fuser -v /dev/nvidia*"],
                    shell=True,  # nosemgrep
                    capture_output=True,
                    text=True,
                ).stdout
                print(f"Processes using GPU: {output}")
            except Exception as e:
                print(f"Error getting processes using GPU: {e}")


def random_uuid() -> str:
    return str(uuid.uuid4())


T = TypeVar("T")


class ProducerFinished:
    pass


def await_coroutines(*coroutines: Coroutine[Any, Any, T]) -> AsyncIterator[Tuple[int, T]]:
    """Await multiple coroutines concurrently.

    Returns an async iterator that yields the results of the coroutines as they complete.
    """
    queue: asyncio.Queue[Union[Tuple[int, T], ProducerFinished, Exception]] = asyncio.Queue()

    async def producer(i: int, coroutine: Coroutine[Any, Any, T]):
        try:
            result = await coroutine
            await queue.put((i, result))
        except Exception as e:
            await queue.put(e)
        # Signal to the consumer that we've finished
        await queue.put(ProducerFinished())

    _tasks = [asyncio.create_task(producer(i, coroutine)) for i, coroutine in enumerate(coroutines)]

    async def consumer():
        remaining = len(coroutines)
        try:
            while remaining or not queue.empty():
                item = await queue.get()

                if isinstance(item, ProducerFinished):
                    # Signal that a producer finished- not a real item
                    remaining -= 1
                    continue

                if isinstance(item, Exception):
                    raise item
                yield item
        except (Exception, asyncio.CancelledError) as e:
            for task in _tasks:
                if sys.version_info >= (3, 9):
                    # msg parameter only supported in Python 3.9+
                    task.cancel(e)
                else:
                    task.cancel()
            raise e
        await asyncio.gather(*_tasks)

    return consumer()
