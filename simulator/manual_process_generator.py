# simulator/manual_process_generator.py

from simulator.process import Process
from typing import List


class ManualProcessGenerator:
    def __init__(self):
        self.processes_by_time = {}
        self.pid_counter = 0

        self._collect_input()

    def _collect_input(self):
        n = int(input("Enter number of processes: "))

        for _ in range(n):
            arrival = int(input("Arrival time: "))
            bursts = list(map(int, input("CPU bursts (space separated): ").split()))
            priority = int(input("Priority: "))

            self.pid_counter += 1
            p = Process(
                pid=self.pid_counter,
                arrival_time=arrival,
                cpu_bursts=bursts,
                priority=priority
            )

            self.processes_by_time.setdefault(arrival, []).append(p)

    def get_arrivals(self, current_time: int) -> List[Process]:
        return self.processes_by_time.pop(current_time, [])
