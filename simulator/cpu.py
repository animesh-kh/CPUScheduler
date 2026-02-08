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

    # ==================================================
    # Core execution logic
    # ==================================================

    def run_one_cycle(self):
        """
        Run ONE scheduling + execution cycle.
        """
        # ----------------------------
        # If no ready process → idle
        # ----------------------------
        if self.ready_queue.is_empty():
            # Idle CPU still advances time
            self.clock.tick()
            return

        # ----------------------------
        # Scheduling decision
        # ----------------------------
        process = self.scheduler.select_process(
            self.ready_queue,
            self.clock.now()
        )

        # Remove selected process from ready queue
        self.ready_queue.remove(process)

        self.context_switches += 1
        self.current_process = process

        # ----------------------------
        # Dispatch
        # ----------------------------
        process.start_execution(self.clock.now())

        executed_ticks = 0

        # ----------------------------
        # Execute (tick by tick)
        # ----------------------------
        while executed_ticks < self.time_quantum:
            # Execute one CPU tick
            process.execute_one_tick()
            executed_ticks += 1

            # Advance system clock
            self.clock.tick()

            # Increment waiting time for all READY processes
            for p in self.ready_queue:
                p.increment_waiting_time()

            # If CPU burst finished, stop execution
            if process.is_burst_complete():
                break

        # ----------------------------
        # Burst accounting
        # ----------------------------
        process.complete_burst(executed_ticks)

        # ----------------------------
        # State transition
        # ----------------------------
        if process.has_more_work():
            process.state = ProcessState.READY
            self.ready_queue.add(process)
        else:
            process.terminate()

        self.current_process = None
