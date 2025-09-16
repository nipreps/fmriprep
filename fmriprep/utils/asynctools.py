import asyncio
from collections.abc import Callable
from typing import TypeVar

R = TypeVar('R')


async def worker(job: Callable[[], R], semaphore: asyncio.Semaphore) -> R:
    async with semaphore:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, job)
