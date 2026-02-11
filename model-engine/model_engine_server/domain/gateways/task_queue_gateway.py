# This is the abstract class defining putting and retrieving tasks into a queue.
import asyncio
import functools
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from model_engine_server.common.dtos.tasks import CreateAsyncTaskV1Response, GetAsyncTaskV1Response


class TaskQueueGateway(ABC):
    """
    Base class for TaskQueue repositories.
    """

    @abstractmethod
    def send_task(
        self,
        task_name: str,
        queue_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        expires: Optional[int] = None,
    ) -> CreateAsyncTaskV1Response:
        """
        Sends a task to the queue.

        Args:
            task_name: The name of the task to submit.
            queue_name: The name of the queue of the task.
            args: Optional arguments for the task.
            kwargs: Optional keyword-arguments for the task.
            expires: Optional number of seconds before the time should time out.

        Returns: The unique identifier for the task.
        """

    async def send_task_async(
        self,
        task_name: str,
        queue_name: str,
        args: Optional[List[Any]] = None,
        kwargs: Optional[Dict[str, Any]] = None,
        expires: Optional[int] = None,
    ) -> CreateAsyncTaskV1Response:
        """
        Non-blocking version of send_task that runs in a thread executor
        to avoid blocking the event loop.
        """
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            functools.partial(
                self.send_task,
                task_name=task_name,
                queue_name=queue_name,
                args=args,
                kwargs=kwargs,
                expires=expires,
            ),
        )

    @abstractmethod
    def get_task(self, task_id: str) -> GetAsyncTaskV1Response:
        """
        Gets a task's status and its final result if it's done.
        """
