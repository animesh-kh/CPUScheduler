# simulator/ready_queue.py

from collections import deque


class ReadyQueue:
    def __init__(self):
        self._queue = deque()

    # ==========================
    # Core operations
    # ==========================

    def add(self, process):
        """
        Add a process to the ready queue.
        """
        self._queue.append(process)

    def remove(self, process):
        """
        Remove a specific process chosen by the scheduler.
        """
        self._queue.remove(process)

    def is_empty(self) -> bool:
        return len(self._queue) == 0

    # ==========================
    # Introspection helpers
    # ==========================

    def __len__(self):
        return len(self._queue)

    def __iter__(self):
        """
        Allows: for p in ready_queue
        """
        return iter(self._queue)

    def peek_all(self):
        """
        Return a list of all ready processes (read-only use).
        """
        return list(self._queue)

    def __repr__(self):
        pids = [p.pid for p in self._queue]
        return f"ReadyQueue({pids})"
