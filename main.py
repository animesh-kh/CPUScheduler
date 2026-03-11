# main.py
"""
Runs FCFS and Preemptive Priority across multiple seeds and workload configs.
Each run logs every scheduling decision via DataCollector (long format).
All rows are merged into: data/scheduling_dataset.csv

Long format: one row per candidate process per decision.
Rows sharing the same decision_id represent the same scheduling moment.

Run:
    python main.py
"""

import csv
import os
from itertools import product

from simulator.clock import SystemClock
from simulator.cpu import CPU
from simulator.ready_queue import ReadyQueue
from simulator.simulator import Simulator
from simulator.process_generator import ProcessGenerator
from simulator.data_collector import DataCollector

from scheduler.fcfs import FCFSScheduler
from scheduler.priority import PreemptivePriorityScheduler

from metrics.metrics_collector import MetricsCollector


# ─────────────────────────────────────────────────────────────────────────────
# Experiment grid
# ─────────────────────────────────────────────────────────────────────────────

SEEDS            = [42, 7, 123, 999]
ARRIVAL_PROBS    = [0.2, 0.4, 0.6]
AVG_BURST_TIMES  = [3, 6, 10]
MAX_TIME         = 500
TIME_QUANTUM     = 999

SCHEDULERS = {
    "FCFS":                FCFSScheduler,
    "Priority_Preemptive": PreemptivePriorityScheduler,
}

OUTPUT_CSV = "data/scheduling_dataset.csv"


# ─────────────────────────────────────────────────────────────────────────────
# Single-run helper
# ─────────────────────────────────────────────────────────────────────────────

def build_and_run(scheduler_name, scheduler_cls,
                  seed, arrival_prob, avg_burst_time):
    collector   = DataCollector(scheduler_name=scheduler_name)
    clock       = SystemClock()
    ready_queue = ReadyQueue()
    generator   = ProcessGenerator(
        arrival_probability=arrival_prob,
        max_bursts=4,
        avg_burst_time=avg_burst_time,
        seed=seed,
    )
    cpu = CPU(
        clock, scheduler_cls(), ready_queue,
        time_quantum=TIME_QUANTUM,
        data_collector=collector,
    )
    sim = Simulator(clock, generator, ready_queue, cpu, max_time=MAX_TIME)
    sim.run()

    metrics = MetricsCollector(sim.all_processes, clock.now(), cpu.busy_ticks)
    summary = metrics.summary()
    summary.update({
        "scheduler":    scheduler_name,
        "seed":         seed,
        "arrival_prob": arrival_prob,
        "avg_burst":    avg_burst_time,
    })

    # Attach run metadata to every data row
    rows = collector.get_rows()
    for row in rows:
        row["seed"]            = seed
        row["arrival_prob"]    = arrival_prob
        row["avg_burst_config"] = avg_burst_time

    return summary, rows


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_rows    = []
    all_results = []

    # Give each run a globally unique decision_id offset so ids don't
    # collide when rows from different runs are merged into one CSV
    global_decision_offset = 0

    configs = list(product(SEEDS, ARRIVAL_PROBS, AVG_BURST_TIMES))
    total   = len(configs) * len(SCHEDULERS)

    print(f"\nRunning {total} simulations "
          f"({len(SCHEDULERS)} schedulers x {len(SEEDS)} seeds x "
          f"{len(ARRIVAL_PROBS)} arrival probs x {len(AVG_BURST_TIMES)} burst configs)\n")

    run_num = 0
    for (seed, arr, burst) in configs:
        for sched_name, sched_cls in SCHEDULERS.items():
            run_num += 1
            print(f"  [{run_num:>3}/{total}] {sched_name:25s} "
                  f"seed={seed:<4} arr={arr:.1f}  burst={burst}", end=" ... ")

            summary, rows = build_and_run(sched_name, sched_cls, seed, arr, burst)

            # Make decision_ids globally unique across all runs
            for row in rows:
                row["decision_id"] += global_decision_offset
            if rows:
                global_decision_offset = rows[-1]["decision_id"]

            all_rows.extend(rows)
            all_results.append(summary)

            decisions = summary.get("processes_completed", "?")
            print(f"{len(rows):>6} rows  ({len(rows)} candidates across decisions)")

    # ── Save merged CSV ───────────────────────────────────────────────
    os.makedirs("data", exist_ok=True)

    if all_rows:
        fieldnames = list(all_rows[0].keys())
        with open(OUTPUT_CSV, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_rows)

        # Count unique decisions
        unique_decisions = len(set(r["decision_id"] for r in all_rows))

        print(f"\n{'='*62}")
        print(f"  Dataset saved        ->  {OUTPUT_CSV}")
        print(f"  Total rows           :   {len(all_rows):,}  (one per candidate per decision)")
        print(f"  Unique decisions     :   {unique_decisions:,}")
        print(f"  Columns              :   {len(fieldnames)}  (was 57, now {len(fieldnames)})")
        print(f"  Avg candidates/step  :   {len(all_rows)/unique_decisions:.2f}")
        print(f"{'='*62}\n")

    # ── Aggregate metrics ─────────────────────────────────────────────
    def avg(key, results):
        vals = [r[key] for r in results if r.get(key) is not None]
        return sum(vals) / len(vals) if vals else None

    fcfs = [r for r in all_results if r["scheduler"] == "FCFS"]
    prio = [r for r in all_results if r["scheduler"] == "Priority_Preemptive"]

    print("Aggregate metrics across all runs:\n")
    print(f"  {'Metric':<28} {'FCFS':>12}  {'Priority':>12}")
    print(f"  {'-'*54}")
    for key, label in [
        ("avg_waiting_time",    "Avg Waiting Time"),
        ("avg_turnaround_time", "Avg Turnaround"),
        ("cpu_utilisation",     "CPU Utilisation"),
    ]:
        fmt = lambda v: f"{v:.3f}" if v is not None else "N/A"
        print(f"  {label:<28} {fmt(avg(key, fcfs)):>12}  {fmt(avg(key, prio)):>12}")
    print()