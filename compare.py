#!/usr/bin/env python3
# compare.py
"""
Scheduler Comparison  —  FCFS vs Priority vs Q-Learning vs DQN
===============================================================
Runs every scheduler on the SAME evaluation seed after training,
then prints a side-by-side metrics table.

Usage (from project root)
-------------------------
  python compare.py

Optional flags (edit the CONFIG block below):
  TRAIN_EPISODES  — number of training episodes for RL schedulers
  MAX_TIME        — simulation ticks per episode / evaluation run
  EVAL_SEED       — random seed for the final evaluation run
  TRAIN_QL        — set False to skip Q-Learning training (use saved policy)
  TRAIN_DQN       — set False to skip DQN training (use saved policy)
"""

# ══════════════════════════════════════════════════════════════════════════════
# CONFIG  — tweak these without touching any other file
# ══════════════════════════════════════════════════════════════════════════════

TRAIN_EPISODES    = 200          # RL training episodes
MAX_TIME          = 500          # ticks per episode / evaluation
TIME_QUANTUM      = 10           # CPU time-slice
ARRIVAL_PROB      = 0.3          # P(new process each tick)
AVG_BURST_TIME    = 5            # mean CPU burst length
EVAL_SEED         = 9999         # SAME seed for every scheduler's evaluation

QL_POLICY_PATH    = "ml/q_policy.pkl"
DQN_POLICY_PATH   = "ml/dqn_policy.pt"

TRAIN_QL          = True         # False → load existing q_policy.pkl
TRAIN_DQN         = True         # False → load existing dqn_policy.pt

# ══════════════════════════════════════════════════════════════════════════════
# Imports
# ══════════════════════════════════════════════════════════════════════════════

from simulator.clock import SystemClock
from simulator.process_generator import ProcessGenerator
from simulator.ready_queue import ReadyQueue
from simulator.cpu import CPU
from simulator.simulator import Simulator
from metrics.metrics_collector import MetricsCollector, compare_schedulers

from scheduler.fcfs import FCFSScheduler
from scheduler.priority import PreemptivePriorityScheduler
from scheduler.q_learning import QLearningScheduler
from scheduler.dqn_scheduler import DQNScheduler

import ml.q_trainer  as q_trainer
import ml.dqn_trainer as dqn_trainer


# ══════════════════════════════════════════════════════════════════════════════
# Helpers
# ══════════════════════════════════════════════════════════════════════════════

def _run_evaluation(scheduler, name: str) -> dict:
    """Run one greedy evaluation episode and return a summary dict."""
    clock       = SystemClock()
    ready_queue = ReadyQueue()
    pg          = ProcessGenerator(
        arrival_probability=ARRIVAL_PROB,
        avg_burst_time=AVG_BURST_TIME,
        seed=EVAL_SEED,
    )
    cpu = CPU(clock, scheduler, ready_queue, TIME_QUANTUM)
    sim = Simulator(clock, pg, ready_queue, cpu, max_time=MAX_TIME)
    sim.run()

    mc = MetricsCollector(sim.all_processes, clock.now(), cpu.busy_ticks)
    summary = mc.summary()
    summary["scheduler"] = name
    return summary


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    results = []

    # ── 1. FCFS (no training needed) ──────────────────────────────────────
    print("\n[1/4] Evaluating FCFS …")
    results.append(_run_evaluation(FCFSScheduler(), "FCFS"))

    # ── 2. Preemptive Priority (no training needed) ────────────────────────
    print("[2/4] Evaluating Preemptive Priority …")
    results.append(_run_evaluation(PreemptivePriorityScheduler(),
                                   "Priority (Preemptive)"))

    # ── 3. Q-Learning ─────────────────────────────────────────────────────
    if TRAIN_QL:
        print("\n[3/4] Training Q-Learning …")
        q_trainer.train(
            episodes=TRAIN_EPISODES,
            max_time=MAX_TIME,
            time_quantum=TIME_QUANTUM,
            arrival_probability=ARRIVAL_PROB,
            avg_burst_time=AVG_BURST_TIME,
            policy_path=QL_POLICY_PATH,
            resume=True,
        )
    else:
        print("\n[3/4] Skipping Q-Learning training (TRAIN_QL=False)")

    print("      Evaluating Q-Learning (greedy) …")
    ql_scheduler = QLearningScheduler(
        policy_path=QL_POLICY_PATH,
        training_mode=False,
    )
    results.append(_run_evaluation(ql_scheduler, "Q-Learning"))

    # ── 4. DQN ────────────────────────────────────────────────────────────
    if TRAIN_DQN:
        print("\n[4/4] Training DQN …")
        dqn_trainer.train(
            episodes=TRAIN_EPISODES,
            max_time=MAX_TIME,
            time_quantum=TIME_QUANTUM,
            arrival_probability=ARRIVAL_PROB,
            avg_burst_time=AVG_BURST_TIME,
            policy_path=DQN_POLICY_PATH,
            resume=True,
        )
    else:
        print("\n[4/4] Skipping DQN training (TRAIN_DQN=False)")

    print("      Evaluating DQN (greedy) …")
    dqn_scheduler = DQNScheduler(
        policy_path=DQN_POLICY_PATH,
        training_mode=False,
    )
    results.append(_run_evaluation(dqn_scheduler, "DQN"))

    # ── Final comparison table ─────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  FINAL COMPARISON  (eval_seed={EVAL_SEED}, max_time={MAX_TIME})")
    print(f"{'═'*70}")
    compare_schedulers(results)

    # ── Winner summary ─────────────────────────────────────────────────────
    _print_winner_summary(results)


def _print_winner_summary(results):
    """Highlight the best scheduler per metric."""
    metrics = [
        ("avg_waiting_time",    "Avg Waiting Time",    "lower"),
        ("avg_turnaround_time", "Avg Turnaround Time", "lower"),
        ("cpu_utilisation",     "CPU Utilisation",     "higher"),
        ("throughput",          "Throughput",          "higher"),
    ]

    print("  WINNERS PER METRIC")
    print(f"  {'Metric':<25}  {'Winner':<28}  Value")
    print(f"  {'-'*65}")

    for key, label, direction in metrics:
        valid = [(r["scheduler"], r[key]) for r in results if r[key] is not None]
        if not valid:
            continue
        if direction == "lower":
            best_name, best_val = min(valid, key=lambda x: x[1])
        else:
            best_name, best_val = max(valid, key=lambda x: x[1])

        if key == "cpu_utilisation":
            val_str = f"{best_val*100:.1f}%"
        elif key == "throughput":
            val_str = f"{best_val:.4f} p/t"
        else:
            val_str = f"{best_val:.2f}"

        print(f"  {label:<25}  {best_name:<28}  {val_str}")

    print()


if __name__ == "__main__":
    main()