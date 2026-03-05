# main.py
"""
Run FCFS and Preemptive Priority back-to-back on the SAME workload
and compare their performance metrics.
"""

from simulator.clock import SystemClock
from simulator.cpu import CPU
from simulator.ready_queue import ReadyQueue
from simulator.simulator import Simulator
from simulator.process_generator import ProcessGenerator

from scheduler.fcfs import FCFSScheduler
from scheduler.priority import PreemptivePriorityScheduler

from metrics.metrics_collector import MetricsCollector, compare_schedulers


# ─────────────────────────────────────────────
# Shared workload config  (same seed = same processes)
# ─────────────────────────────────────────────
WORKLOAD = dict(
    arrival_probability=0.4,
    max_bursts=4,
    avg_burst_time=6,
    seed=42,          # ← change to try different workloads
)
MAX_TIME   = 200
TIME_QUANTUM = 999   # effectively infinite for FCFS / Priority (non-RR)


def build_and_run(scheduler, label):
    clock       = SystemClock()
    ready_queue = ReadyQueue()
    generator   = ProcessGenerator(**WORKLOAD)
    cpu         = CPU(clock, scheduler, ready_queue, time_quantum=TIME_QUANTUM)
    sim         = Simulator(clock, generator, ready_queue, cpu, max_time=MAX_TIME)

    sim.run()

    collector = MetricsCollector(sim.all_processes, clock.now(), cpu.busy_ticks)
    collector.print_report(label)

    result = collector.summary()
    result["scheduler"] = label
    return result


if __name__ == "__main__":
    print("\nRunning FCFS ...")
    fcfs_result = build_and_run(FCFSScheduler(), "FCFS")

    print("Running Preemptive Priority ...")
    prio_result = build_and_run(PreemptivePriorityScheduler(), "Priority (Preemptive)")

    compare_schedulers([fcfs_result, prio_result])