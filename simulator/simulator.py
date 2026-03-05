# simulator/simulator.py

from simulator.process import ProcessState


class Simulator:
    def __init__(
            self,
            clock,
            process_generator,
            ready_queue,
            cpu,
            max_time=1000
    ):
        self.clock = clock
        self.process_generator = process_generator
        self.ready_queue = ready_queue
        self.cpu = cpu
        self.max_time = max_time

        self.all_processes = []

    # ==========================
    # Main simulation loop
    # ==========================

    def run(self):
        """
        Main simulation loop.

        Order each tick:
          1. Admit new arrivals.
          2. Check if the scheduler wants to preempt the running process
             (important for preemptive schedulers like Preemptive Priority).
          3. Run one CPU tick.
        """
        while self.clock.now() < self.max_time:
            self._admit_new_processes()

            # Give preemptive schedulers a chance to react to new arrivals
            self.cpu.check_preemption()

            self.cpu.run_one_tick()

            if self._is_system_idle() and not self._has_future_arrivals():
                break

    # ==========================
    # Internal helpers
    # ==========================

    def _admit_new_processes(self):
        arrivals = self.process_generator.get_arrivals(self.clock.now())
        for process in arrivals:
            process.state = ProcessState.READY
            self.ready_queue.add(process)
            self.all_processes.append(process)

    def _is_system_idle(self):
        return (
            self.ready_queue.is_empty()
            and self.cpu.current_process is None
        )

    def _has_future_arrivals(self):
        if hasattr(self.process_generator, 'processes_by_time'):
            return len(self.process_generator.processes_by_time) > 0
        return True