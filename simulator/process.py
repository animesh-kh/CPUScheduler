# simulator/process.py

from enum import Enum
from typing import List, Optional


class ProcessState(Enum):
    NEW = "NEW"
    READY = "READY"
    RUNNING = "RUNNING"
    WAITING = "WAITING"
    TERMINATED = "TERMINATED"


class Process:
    def __init__(
        self,
        pid: int,
        arrival_time: int,
        cpu_bursts: List[int],
        priority: int = 0
    ):
        # -------------------
        # Identity
        # -------------------
        self.pid = pid
        self.arrival_time = arrival_time
        self.priority = priority

        # -------------------
        # Hidden ground truth (SIMULATOR ONLY)
        # -------------------
        self._cpu_bursts = cpu_bursts.copy()   # future bursts
        self._remaining_burst: Optional[int] = None

        # -------------------
        # Observable history (Scheduler / ML)
        # -------------------
        self.cpu_burst_history: List[int] = []

        # -------------------
        # State & accounting
        # -------------------
        self.state = ProcessState.NEW
        self.waiting_time = 0
        self.total_cpu_time = 0
        self.last_scheduled_time: Optional[int] = None

    # ==================================================
    # Execution lifecycle
    # ==================================================

    def start_execution(self, current_time: int):
        """
        Called when the process is dispatched to the CPU.
        """
        self.state = ProcessState.RUNNING
        self.last_scheduled_time = current_time

        # Load a new CPU burst if needed
        if self._remaining_burst is None:
            self._remaining_burst = self._cpu_bursts.pop(0)

    def execute_one_tick(self):
        """
        Execute the process for ONE time unit.
        """
        if self.state != ProcessState.RUNNING:
            raise RuntimeError("Process is not in RUNNING state")

        self._remaining_burst -= 1
        self.total_cpu_time += 1

    def is_burst_complete(self) -> bool:
        """
        Check whether the current CPU burst is finished.
        """
        return self._remaining_burst == 0

    def complete_burst(self, executed_ticks: int):
        """
        Called when execution stops due to
        preemption or burst completion.
        """
        self.cpu_burst_history.append(executed_ticks)
        self._remaining_burst = None

    def has_more_work(self) -> bool:
        """
        Check if the process still has future CPU bursts.
        """
        return len(self._cpu_bursts) > 0

    def terminate(self):
        """
        Terminate the process.
        """
        self.state = ProcessState.TERMINATED

    # ==================================================
    # Waiting time accounting
    # ==================================================

    def increment_waiting_time(self):
        """
        Increment waiting time if process is READY.
        """
        if self.state == ProcessState.READY:
            self.waiting_time += 1

    # ==================================================
    # Debug helpers
    # ==================================================

    def __repr__(self):
        return (
            f"Process(pid={self.pid}, "
            f"state={self.state.value}, "
            f"waiting={self.waiting_time}, "
            f"cpu_used={self.total_cpu_time})"
        )
