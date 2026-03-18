from scheduler.base_scheduler import Scheduler

class PreemptivePriorityScheduler(Scheduler):
    def select_process(self, ready_queue, current_time):
        # Lower priority number = higher priority
        return min(ready_queue, key=lambda p: p.priority)

    def should_preempt(self, current_process, ready_queue) -> bool:
        best = min(ready_queue, key=lambda p: p.priority)
        return best.priority < current_process.priority