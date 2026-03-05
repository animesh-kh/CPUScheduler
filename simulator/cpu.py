# simulator/cpu.py

from simulator.process import ProcessState


class CPU:
    def __init__(self, clock, scheduler, ready_queue, time_quantum: int):
        self.clock = clock
        self.scheduler = scheduler
        self.ready_queue = ready_queue
        self.time_quantum = time_quantum

        self.current_process = None
        self.context_switches = 0
        self.ticks_in_current_quantum = 0

        # For CPU utilisation metric
        self.busy_ticks = 0

    # ==================================================
    # Preemption (called by Simulator after new arrivals)
    # ==================================================

    def check_preemption(self):
        """
        Ask the scheduler whether the currently running process should
        be preempted by something in the ready queue.

        Called by the Simulator immediately after admitting new arrivals,
        so that a high-priority newcomer can kick out the running process
        before the next tick executes.
        """
        if self.current_process is None or self.ready_queue.is_empty():
            return

        if self.scheduler.should_preempt(self.current_process, self.ready_queue):
            process = self.current_process

            # Record partial burst
            if self.ticks_in_current_quantum > 0:
                process.record_preemption(self.ticks_in_current_quantum)

            # Send back to ready queue
            process.state = ProcessState.READY
            self.ready_queue.add(process)

            # Release CPU
            self.current_process = None
            self.ticks_in_current_quantum = 0

    # ==================================================
    # Core execution logic (tick-by-tick)
    # ==================================================

    def run_one_tick(self):
        """
        Run ONE tick of execution.
        Called once per clock tick by the Simulator.
        """
        # Step 1: Schedule if CPU is free and queue is not empty
        if self.current_process is None and not self.ready_queue.is_empty():
            self._schedule_next_process()

        # Step 2: Execute or idle
        if self.current_process is not None:
            self._execute_one_tick()
            self.busy_ticks += 1
        else:
            self.clock.tick()

        # Step 3: Increment waiting time for READY processes
        for p in self.ready_queue:
            p.increment_waiting_time()

    # ==================================================
    # Internal helpers
    # ==================================================

    def _schedule_next_process(self):
        """Make a scheduling decision and dispatch the selected process."""
        process = self.scheduler.select_process(
            self.ready_queue,
            self.clock.now()
        )
        self.ready_queue.remove(process)

        self.context_switches += 1
        self.current_process = process
        self.ticks_in_current_quantum = 0

        process.start_execution(self.clock.now())

    def _execute_one_tick(self):
        """Execute the current process for one tick; handle completion/preemption."""
        process = self.current_process

        process.execute_one_tick()
        self.ticks_in_current_quantum += 1

        self.clock.tick()

        burst_complete = process.is_burst_complete()
        quantum_expired = (self.ticks_in_current_quantum >= self.time_quantum)

        if burst_complete or quantum_expired:
            if burst_complete:
                process.complete_burst(self.ticks_in_current_quantum)
            else:
                process.record_preemption(self.ticks_in_current_quantum)

            if process.has_more_work():
                process.state = ProcessState.READY
                self.ready_queue.add(process)
            else:
                process.terminate(current_time=self.clock.now())  # <-- pass time

            self.current_process = None
            self.ticks_in_current_quantum = 0