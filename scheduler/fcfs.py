from scheduler.base_scheduler import Scheduler

class FCFSScheduler(Scheduler):
    def select_process(self, ready_queue, current_time):
        # First-come first-served: pick the process with the earliest arrival
        return min(ready_queue, key=lambda p: p.arrival_time)