# scheduler/base_scheduler.py

from abc import ABC, abstractmethod


class Scheduler(ABC):
    """
    Abstract base class for all schedulers.
    """

    @abstractmethod
    def select_process(self, ready_queue, current_time):
        """
        Select ONE process from the ready queue.

        Args:
            ready_queue: iterable of READY processes
            current_time: current system time

        Returns:
            Process: selected process
        """
        pass

    def should_preempt(self, current_process, ready_queue) -> bool:
        """
        Return True if the current running process should be preempted
        in favour of something in the ready queue.

        Default: non-preemptive (never preempt).
        Override in preemptive schedulers.
        """
        return False