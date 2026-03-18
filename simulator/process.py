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
        # Identity
        self.pid = pid
        self.arrival_time = arrival_time
        self.priority = priority

        # Hidden ground truth (SIMULATOR ONLY)
        self._cpu_bursts = cpu_bursts.copy()
        self._remaining_burst: Optional[int] = None

        # Observable history (Scheduler / ML)
        self.cpu_burst_history: List[int] = []

        # State & accounting
        self.state = ProcessState.NEW
        self.waiting_time = 0
        self.total_cpu_time = 0
        self.last_scheduled_time: Optional[int] = None

        # Completion tracking (for metrics)
        self.finish_time: Optional[int] = None    # set on terminate()
        self.response_time: Optional[int] = None  # first time it got CPU

    # ==================================================
    # Execution lifecycle
    # ==================================================

    def start_execution(self, current_time: int):
        self.state = ProcessState.RUNNING
        self.last_scheduled_time = current_time

        # Record first response time
        if self.response_time is None:
            self.response_time = current_time - self.arrival_time

        if self._remaining_burst is None:
            self._remaining_burst = self._cpu_bursts.pop(0)

    def execute_one_tick(self):
        if self.state != ProcessState.RUNNING:
            raise RuntimeError("Process is not in RUNNING state")
        self._remaining_burst -= 1
        self.total_cpu_time += 1

    def is_burst_complete(self) -> bool:
        return self._remaining_burst == 0

    def complete_burst(self, executed_ticks: int):
        """Called ONLY when a CPU burst actually completes."""
        self.cpu_burst_history.append(executed_ticks)
        self._remaining_burst = None

    def record_preemption(self, executed_ticks: int):
        """Called when execution stops due to preemption."""
        self.cpu_burst_history.append(executed_ticks)

    def has_more_work(self) -> bool:
        return (
            self._remaining_burst is not None
            or len(self._cpu_bursts) > 0
        )

    def terminate(self, current_time: int = None):
        """Terminate and record finish time."""
        self.state = ProcessState.TERMINATED
        if current_time is not None:
            self.finish_time = current_time

    # ==================================================
    # Derived metrics
    # ==================================================

    def turnaround_time(self) -> Optional[int]:
        if self.finish_time is None:
            return None
        return self.finish_time - self.arrival_time

    # ==================================================
    # Waiting time accounting
    # ==================================================

    def increment_waiting_time(self):
        if self.state == ProcessState.READY:
            self.waiting_time += 1

    # ==================================================
    # Debug
    # ==================================================

    def __repr__(self):
        return (
            f"Process(pid={self.pid}, "
            f"state={self.state.value}, "
            f"waiting={self.waiting_time}, "
            f"cpu_used={self.cpu_burst_history}, "
            f"arrival={self.arrival_time}, "
            f"finish={self.finish_time})"
        )