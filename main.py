# main.py

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
# Shared hyper-parameters
# ──────────────────────────────────────────────────────────────────────────────

POLICY_PATH    = "ml/q_policy.pkl"
TRAIN_EPISODES = 500        # more episodes = Q-values converge more reliably
EPSILON_DECAY  = 0.9998     # reaches ε=0.05 around episode 250
MAX_TIME       = 500
TIME_QUANTUM   = 10
ARRIVAL_PROB   = 0.3
AVG_BURST      = 5
EVAL_SEED      = 9999


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

    # ── Step 1: Train ─────────────────────────────────────────────────────────
    trained_scheduler = train(
        episodes=TRAIN_EPISODES,
        max_time=MAX_TIME,
        time_quantum=TIME_QUANTUM,
        arrival_probability=ARRIVAL_PROB,
        avg_burst_time=AVG_BURST,
        policy_path=POLICY_PATH,
        base_seed=0,
        resume=True,
        epsilon_decay=EPSILON_DECAY,
        log_every=50,
    )

    # ── Step 2: Evaluate greedy policy ────────────────────────────────────────
    evaluate(
        policy_path=POLICY_PATH,
        max_time=MAX_TIME,
        time_quantum=TIME_QUANTUM,
        arrival_probability=ARRIVAL_PROB,
        avg_burst_time=AVG_BURST,
        seed=EVAL_SEED,
    )

    # ── Step 3: Compare all schedulers on the same held-out workload ──────────
    print("\n  Running comparison on held-out workload (seed=9999)...")

    ql_scheduler = QLearningScheduler(
        policy_path=POLICY_PATH,
        training_mode=False,
    )

    results = [
        _run_once(FCFSScheduler(),                "FCFS",                  EVAL_SEED),
        _run_once(PreemptivePriorityScheduler(),  "Priority (Preemptive)", EVAL_SEED),
        _run_once(ql_scheduler,                   "Q-Learning (Greedy)",   EVAL_SEED),
    ]

    compare_schedulers(results)

    # ── Step 4: Inspect what the agent learned ────────────────────────────────
    trained_scheduler.print_q_table()