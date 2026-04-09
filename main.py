# main.py
"""
CPU Scheduler Simulator — Main Entry Point
==========================================

Workflow
--------
1. TRAIN  : Run the Q-learning agent over many episodes to build the policy.
2. EVALUATE: Reload the policy and run one greedy episode.
3. COMPARE: Benchmark Q-Learning vs FCFS vs Preemptive Priority on an
            identical held-out workload.

Usage
-----
    python main.py
"""

from simulator.clock import SystemClock
from simulator.process_generator import ProcessGenerator
from simulator.ready_queue import ReadyQueue
from simulator.cpu import CPU
from simulator.simulator import Simulator
from metrics.metrics_collector import MetricsCollector, compare_schedulers

from scheduler.fcfs import FCFSScheduler
from scheduler.priority import PreemptivePriorityScheduler
from scheduler.q_learning import QLearningScheduler

from ml.q_trainer import train, evaluate

# ──────────────────────────────────────────────────────────────────────────────
# Shared hyper-parameters — tweak here to affect all experiments
# ──────────────────────────────────────────────────────────────────────────────

POLICY_PATH        = "ml/q_policy.pkl"
TRAIN_EPISODES     = 100        # increase for a better policy (try 500+)
MAX_TIME           = 500        # simulation ticks per episode
TIME_QUANTUM       = 10         # RR-style quantum passed to CPU
ARRIVAL_PROB       = 0.3        # probability of a new process per tick
AVG_BURST          = 5          # mean CPU burst length (exponential dist)
EVAL_SEED          = 9999       # held-out seed — not used during training


# ──────────────────────────────────────────────────────────────────────────────
# Helper: run one simulation and return a metrics summary dict
# ──────────────────────────────────────────────────────────────────────────────

def _run_once(scheduler, scheduler_name: str, seed: int) -> dict:
    clock       = SystemClock()
    ready_queue = ReadyQueue()
    pg          = ProcessGenerator(
        arrival_probability=ARRIVAL_PROB,
        avg_burst_time=AVG_BURST,
        seed=seed,
    )
    cpu = CPU(clock, scheduler, ready_queue, TIME_QUANTUM)
    sim = Simulator(clock, pg, ready_queue, cpu, max_time=MAX_TIME)
    sim.run()

    mc = MetricsCollector(sim.all_processes, clock.now(), cpu.busy_ticks)
    summary = mc.summary()
    summary["scheduler"] = scheduler_name
    return summary


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # ── Step 1: Train the Q-learning scheduler ────────────────────────────────
    trained_scheduler = train(
        episodes=TRAIN_EPISODES,
        max_time=MAX_TIME,
        time_quantum=TIME_QUANTUM,
        arrival_probability=ARRIVAL_PROB,
        avg_burst_time=AVG_BURST,
        policy_path=POLICY_PATH,
        base_seed=0,
        log_every=10,
    )

    # ── Step 2: Evaluate the saved greedy policy ──────────────────────────────
    evaluate(
        policy_path=POLICY_PATH,
        max_time=MAX_TIME,
        time_quantum=TIME_QUANTUM,
        arrival_probability=ARRIVAL_PROB,
        avg_burst_time=AVG_BURST,
        seed=EVAL_SEED,
    )

    # ── Step 3: Compare all three schedulers on the same held-out workload ────
    print("\n  Running comparison on held-out workload (seed=9999)...")

    # Q-Learning — reload policy, act greedy
    ql_scheduler = QLearningScheduler(
        policy_path=POLICY_PATH,
        training_mode=False,
    )

    results = [
        _run_once(FCFSScheduler(),               "FCFS",                  EVAL_SEED),
        _run_once(PreemptivePriorityScheduler(),  "Priority (Preemptive)", EVAL_SEED),
        _run_once(ql_scheduler,                  "Q-Learning (Greedy)",   EVAL_SEED),
    ]

    compare_schedulers(results)

    # ── Step 4 (optional): Inspect what the agent learned ─────────────────────
    trained_scheduler.print_q_table()