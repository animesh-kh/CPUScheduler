# metrics/metrics_collector.py

from typing import List, Dict, Optional
from simulator.process import Process, ProcessState


class MetricsCollector:
    """
    Computes scheduling performance metrics from a completed simulation.

    Metrics
    -------
    - Average Waiting Time   : mean time processes spent in READY state
    - Average Turnaround Time: mean (finish_time - arrival_time)
    - CPU Utilisation        : fraction of ticks the CPU was busy
    - Throughput             : completed processes per time unit
    """

    def __init__(self, processes: List[Process], total_ticks: int, busy_ticks: int):
        """
        Parameters
        ----------
        processes   : simulator.all_processes after run() completes
        total_ticks : clock.now() after the simulation ends
        busy_ticks  : cpu.busy_ticks after the simulation ends
        """
        self.processes = processes
        self.total_ticks = total_ticks
        self.busy_ticks = busy_ticks

        self._completed = [
            p for p in processes if p.state == ProcessState.TERMINATED
        ]

    # ==================================================
    # Individual metrics
    # ==================================================

    def average_waiting_time(self) -> Optional[float]:
        """Mean time a process spent waiting in the ready queue."""
        if not self._completed:
            return None
        return sum(p.waiting_time for p in self._completed) / len(self._completed)

    def average_turnaround_time(self) -> Optional[float]:
        """Mean (finish_time - arrival_time) for all completed processes."""
        tats = [p.turnaround_time() for p in self._completed if p.turnaround_time() is not None]
        if not tats:
            return None
        return sum(tats) / len(tats)

    def cpu_utilisation(self) -> Optional[float]:
        """Fraction of total simulation time the CPU was executing a process."""
        if self.total_ticks == 0:
            return None
        return self.busy_ticks / self.total_ticks

    def throughput(self) -> Optional[float]:
        """Number of processes completed per time unit."""
        if self.total_ticks == 0:
            return None
        return len(self._completed) / self.total_ticks

    # ==================================================
    # Summary
    # ==================================================

    def summary(self) -> Dict[str, float]:
        """Return all metrics as a dictionary (convenient for comparison)."""
        return {
            "scheduler":              None,          # filled in by caller
            "processes_completed":    len(self._completed),
            "avg_waiting_time":       self.average_waiting_time(),
            "avg_turnaround_time":    self.average_turnaround_time(),
            "cpu_utilisation":        self.cpu_utilisation(),
            "throughput":             self.throughput(),
        }

    def print_report(self, scheduler_name: str = "Unknown"):
        """Print a formatted metrics report to stdout."""
        awt  = self.average_waiting_time()
        ata  = self.average_turnaround_time()
        util = self.cpu_utilisation()
        thru = self.throughput()

        print(f"\n{'='*50}")
        print(f"  Scheduler : {scheduler_name}")
        print(f"  Simulation ticks : {self.total_ticks}")
        print(f"  Processes completed : {len(self._completed)} / {len(self.processes)}")
        print(f"{'='*50}")
        print(f"  Avg Waiting Time    : {awt:.2f}" if awt is not None else "  Avg Waiting Time    : N/A")
        print(f"  Avg Turnaround Time : {ata:.2f}" if ata is not None else "  Avg Turnaround Time : N/A")
        print(f"  CPU Utilisation     : {util*100:.1f}%" if util is not None else "  CPU Utilisation     : N/A")
        print(f"  Throughput          : {thru:.4f} proc/tick" if thru is not None else "  Throughput          : N/A")
        print(f"{'='*50}\n")


# ==================================================
# Comparison helper
# ==================================================

def compare_schedulers(results: List[Dict]) -> None:
    """
    Print a side-by-side comparison table.

    Parameters
    ----------
    results : list of summary dicts produced by MetricsCollector.summary()
              with 'scheduler' key filled in.

    Example
    -------
    compare_schedulers([
        {"scheduler": "FCFS", ...},
        {"scheduler": "Priority (Preemptive)", ...},
    ])
    """
    metrics_keys = [
        ("avg_waiting_time",    "Avg Wait"),
        ("avg_turnaround_time", "Avg TAT"),
        ("cpu_utilisation",     "CPU Util"),
        ("throughput",          "Throughput"),
    ]

    col_w = 22
    name_w = 24

    header = f"{'Metric':<{name_w}}" + "".join(
        f"{r['scheduler']:<{col_w}}" for r in results
    )
    print("\n" + "=" * (name_w + col_w * len(results)))
    print("  SCHEDULER COMPARISON")
    print("=" * (name_w + col_w * len(results)))
    print(header)
    print("-" * (name_w + col_w * len(results)))

    for key, label in metrics_keys:
        row = f"{label:<{name_w}}"
        for r in results:
            val = r.get(key)
            if val is None:
                row += f"{'N/A':<{col_w}}"
            elif key == "cpu_utilisation":
                row += f"{val*100:.1f}%{'':<{col_w-6}}"
            elif key == "throughput":
                row += f"{val:.4f} p/t{'':<{col_w-10}}"
            else:
                row += f"{val:.2f}{'':<{col_w-4}}"
        print(row)

    print("=" * (name_w + col_w * len(results)) + "\n")
