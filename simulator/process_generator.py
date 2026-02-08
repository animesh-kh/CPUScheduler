# simulator/process_generator.py

import random
from typing import List
from simulator.process import Process


class ProcessGenerator:
    def __init__(
        self,
        arrival_probability: float = 0.3,
        max_bursts: int = 5,
        avg_burst_time: int = 5,
        seed: int = 42
    ):
        """
        arrival_probability : probability of a new process at each tick
        max_bursts          : maximum number of CPU bursts per process
        avg_burst_time      : mean CPU burst length
        seed                : random seed for reproducibility
        """
        self.arrival_probability = arrival_probability
        self.max_bursts = max_bursts
        self.avg_burst_time = avg_burst_time

        self.pid_counter = 0
        random.seed(seed)

    # =====================================
    # Public API
    # =====================================

    def get_arrivals(self, current_time: int) -> List[Process]:
        """
        Called once per clock tick by the simulator.
        Returns a list of newly arrived processes.
        """
        arrivals = []

        if random.random() < self.arrival_probability:
            arrivals.append(self._create_process(current_time))

        return arrivals

    # =====================================
    # Internal helpers
    # =====================================

    def _create_process(self, arrival_time: int) -> Process:
        """
        Create a single new process with hidden CPU bursts.
        """
        self.pid_counter += 1

        num_bursts = random.randint(1, self.max_bursts)

        cpu_bursts = [
            max(1, int(random.expovariate(1 / self.avg_burst_time)))
            for _ in range(num_bursts)
        ]

        priority = random.randint(0, 3)

        return Process(
            pid=self.pid_counter,
            arrival_time=arrival_time,
            cpu_bursts=cpu_bursts,
            priority=priority
        )
