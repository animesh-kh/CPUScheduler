# scheduler/fcfs.py

from scheduler.base_scheduler import Scheduler


class FCFSScheduler(Scheduler):
    def select_process(self, ready_queue, current_time):
        """
        First Come First Serve:
        select process with smallest arrival time.
        """
        return min(
            ready_queue,
            key=lambda p: p.arrival_time
        )
