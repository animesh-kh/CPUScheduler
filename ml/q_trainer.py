# ml/q_trainer.py
"""
Q-Learning Trainer & Evaluator
===============================
Two public functions:

    train(...)    — Run N simulation episodes with the same QLearningScheduler
                    so the Q-table accumulates experience across episodes.
                    Saves the policy to Pickle at the end.

    evaluate(...) — Load the saved policy, run ONE greedy episode (ε=0),
                    print a full metrics report and return the MetricsCollector.

Typical usage (see main.py for a full example):

    from ml.q_trainer import train, evaluate
    train(episodes=100)
    evaluate()
"""

from simulator.clock import SystemClock
from simulator.process_generator import ProcessGenerator
from simulator.ready_queue import ReadyQueue
from simulator.cpu import CPU
from simulator.simulator import Simulator
from metrics.metrics_collector import MetricsCollector
from scheduler.q_learning import QLearningScheduler


# ──────────────────────────────────────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────────────────────────────────────

def train(
    episodes: int = 100,
    max_time: int = 500,
    time_quantum: int = 10,
    arrival_probability: float = 0.3,
    avg_burst_time: int = 5,
    policy_path: str = "ml/q_policy.pkl",
    base_seed: int = 0,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_decay: float = 0.995,
    log_every: int = 10,
) -> QLearningScheduler:
    """
    Train a Q-learning scheduler over multiple simulation episodes.

    A *single* QLearningScheduler instance is reused across all episodes so
    the Q-table grows continuously (warm-start between episodes).  Each
    episode uses a different random seed so the agent sees varied workloads.

    Parameters
    ----------
    episodes          : number of training simulation runs
    max_time          : maximum clock ticks per episode
    time_quantum      : CPU time quantum (passed to CPU)
    arrival_probability: probability of a new process arriving each tick
    avg_burst_time    : mean CPU burst length
    policy_path       : where to save the Pickle policy
    base_seed         : episode i uses seed = base_seed + i
    alpha / gamma     : Q-learning hyper-parameters
    epsilon_decay     : per-decision epsilon decay rate
    log_every         : print a progress line every N episodes

    Returns
    -------
    The trained QLearningScheduler (policy also saved to disk).
    """
    print(f"\n{'='*55}")
    print(f"  Q-Learning Training")
    print(f"  Episodes={episodes}  max_time={max_time}  α={alpha}  γ={gamma}")
    print(f"{'='*55}")

    scheduler = QLearningScheduler(
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
        policy_path=policy_path,
        training_mode=True,
    )

    for ep in range(1, episodes + 1):
        # Fresh simulation components each episode
        clock       = SystemClock()
        ready_queue = ReadyQueue()
        pg          = ProcessGenerator(
            arrival_probability=arrival_probability,
            avg_burst_time=avg_burst_time,
            seed=base_seed + ep,
        )
        cpu = CPU(clock, scheduler, ready_queue, time_quantum)
        sim = Simulator(clock, pg, ready_queue, cpu, max_time=max_time)

        # Reset per-episode TD bookkeeping so episodes are independent
        scheduler._prev_state  = None
        scheduler._prev_reward = None

        sim.run()

        if ep % log_every == 0 or ep == episodes:
            mc  = MetricsCollector(sim.all_processes, clock.now(), cpu.busy_ticks)
            awt = mc.average_waiting_time()
            s   = scheduler.stats()
            awt_str = f"{awt:.2f}" if awt is not None else "N/A"
            print(
                f"  Ep {ep:>4}/{episodes}  |  "
                f"ε={s['epsilon']:.3f}  |  "
                f"Q-states={s['q_states']:>4}  |  "
                f"AvgWait={awt_str}"
            )

    scheduler.save_policy()
    print(f"\n  Total decisions : {scheduler.stats()['decisions']}")
    print(f"  Explore / Exploit: "
          f"{scheduler.stats()['explorations']} / "
          f"{scheduler.stats()['exploitations']}")
    print(f"{'='*55}\n")

    return scheduler


# ──────────────────────────────────────────────────────────────────────────────
# Evaluation
# ──────────────────────────────────────────────────────────────────────────────

def evaluate(
    policy_path: str = "ml/q_policy.pkl",
    max_time: int = 500,
    time_quantum: int = 10,
    arrival_probability: float = 0.3,
    avg_burst_time: int = 5,
    seed: int = 9999,
) -> MetricsCollector:
    """
    Load the saved policy and run ONE fully-greedy episode (ε = 0).

    Parameters
    ----------
    policy_path : Pickle file written by train()
    seed        : use a seed NOT seen during training for a fair test

    Returns
    -------
    MetricsCollector for the greedy episode (metrics already printed).
    """
    print(f"\n{'='*55}")
    print(f"  Q-Learning Greedy Evaluation  (seed={seed})")
    print(f"{'='*55}")

    scheduler = QLearningScheduler(
        policy_path=policy_path,
        training_mode=False,   # load policy, act greedy, no updates
    )

    clock       = SystemClock()
    ready_queue = ReadyQueue()
    pg          = ProcessGenerator(
        arrival_probability=arrival_probability,
        avg_burst_time=avg_burst_time,
        seed=seed,
    )
    cpu = CPU(clock, scheduler, ready_queue, time_quantum)
    sim = Simulator(clock, pg, ready_queue, cpu, max_time=max_time)
    sim.run()

    mc = MetricsCollector(sim.all_processes, clock.now(), cpu.busy_ticks)
    mc.print_report("Q-Learning (Greedy)")
    return mc
