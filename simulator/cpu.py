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

    # ==================================================
    # Core execution logic (tick-by-tick)
    # ==================================================

    def run_one_tick(self):
        """
        Run ONE tick of execution.
        This is called once per clock tick by the simulator.
        """
        # ----------------------------
        # Step 1: Schedule if needed
        # ----------------------------
        if self.current_process is None and not self.ready_queue.is_empty():
            self._schedule_next_process()

        # ----------------------------
        # Step 2: Execute current process (or idle)
        # ----------------------------
        if self.current_process is not None:
            self._execute_one_tick()
        else:
            # CPU is idle - still advance clock
            self.clock.tick()

        # ----------------------------
        # Step 3: Increment waiting time for READY processes
        # ----------------------------
        for p in self.ready_queue:
            p.increment_waiting_time()

    # ==================================================
    # Internal helpers
    # ==================================================

    def _schedule_next_process(self):
        """
        Make a scheduling decision and dispatch the selected process.
        """
        # Ask scheduler to select a process
        process = self.scheduler.select_process(
            self.ready_queue,
            self.clock.now()
        )

        # Remove from ready queue
        self.ready_queue.remove(process)

        # Context switch
        self.context_switches += 1
        self.current_process = process
        self.ticks_in_current_quantum = 0

        # Dispatch to CPU
        process.start_execution(self.clock.now())

    def _execute_one_tick(self):
        """
        Execute the current process for one tick and handle completion/preemption.
        """
        process = self.current_process

        # Execute one tick
        process.execute_one_tick()
        self.ticks_in_current_quantum += 1

        # Advance clock
        self.clock.tick()

        # ----------------------------
        # Check for completion or preemption
        # ----------------------------
        burst_complete = process.is_burst_complete()
        quantum_expired = (self.ticks_in_current_quantum >= self.time_quantum)

        if burst_complete or quantum_expired:
            # Record execution history
            if burst_complete:
                process.complete_burst(self.ticks_in_current_quantum)
            else:
                process.record_preemption(self.ticks_in_current_quantum)

            # State transition
            if process.has_more_work():
                process.state = ProcessState.READY
                self.ready_queue.add(process)
            else:
                process.terminate()
                print(process)

            # Release CPU
            self.current_process = None
            self.ticks_in_current_quantum = 0