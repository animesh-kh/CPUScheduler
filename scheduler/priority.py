# scheduler/priority.py

from scheduler.base_scheduler import Scheduler


class PreemptivePriorityScheduler(Scheduler):
    """
    Preemptive Priority Scheduler.

    Convention: LOWER priority number = HIGHER urgency
                (e.g. priority 0 beats priority 3)

    Preemption rule: if a newly arrived process has a strictly higher
    priority (lower number) than the currently running process, the
    running process is pushed back to the ready queue and the new one
    takes the CPU immediately.

    Tie-breaking: among equal-priority processes, the one that arrived
    first (smallest arrival_time) is preferred — FCFS within a band.
    """

    def select_process(self, ready_queue, current_time):
        """
        Pick the highest-priority (lowest number) process.
        Ties broken by arrival time (FCFS).
        """
        return min(
            ready_queue,
            key=lambda p: (p.priority, p.arrival_time)
        )

    def should_preempt(self, current_process, ready_queue) -> bool:
        """
        Preempt if ANY ready process has strictly higher priority
        (smaller number) than the currently running process.
        """
        best_ready = min(
            ready_queue,
            key=lambda p: (p.priority, p.arrival_time)
        )
        return best_ready.priority < current_process.priority