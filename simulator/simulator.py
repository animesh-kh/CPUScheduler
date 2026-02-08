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
        while self.clock.now() < self.max_time:
            self._admit_new_processes()
            self.cpu.run_one_cycle()

    # ==========================
    # Internal helpers
    # ==========================

    def _admit_new_processes(self):
        """
        Ask the process generator for arrivals at current time.
        """
        arrivals = self.process_generator.get_arrivals(self.clock.now())

        for process in arrivals:
            process.state = ProcessState.READY
            self.ready_queue.add(process)
            self.all_processes.append(process)

    def _is_system_idle(self):
        """
        System is idle if:
        - ready queue empty
        - CPU has no current process
        """
        return (
            self.ready_queue.is_empty()
            and self.cpu.current_process is None
        )
