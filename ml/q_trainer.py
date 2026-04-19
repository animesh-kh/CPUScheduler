# ml/q_trainer.py

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
    resume: bool = True,
    alpha: float = 0.1,
    gamma: float = 0.9,
    epsilon_decay: float = 0.995,
    log_every: int = 10,
) -> QLearningScheduler:

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

    if resume:
        try:
            scheduler.load_policy()
            print(f"  Resuming from existing policy (ε={scheduler.epsilon:.3f})")
        except Exception:
            print("  No existing policy found — starting fresh.")

    for ep in range(1, episodes + 1):
        clock       = SystemClock()
        ready_queue = ReadyQueue()
        pg          = ProcessGenerator(
            arrival_probability=arrival_probability,
            avg_burst_time=avg_burst_time,
            seed=base_seed + ep,
        )
        cpu = CPU(clock, scheduler, ready_queue, time_quantum)
        sim = Simulator(clock, pg, ready_queue, cpu, max_time=max_time)

        # ── Reset all per-episode memory ──────────────────────────────
        # Prevents stale memory from the last decision of the previous
        # episode polluting the first TD update of the new episode.
        scheduler._prev_state         = None
        scheduler._prev_selected_wait = 0.0
        scheduler._prev_avg_wait      = 0.0
        scheduler._prev_priority      = 0

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

    print(f"\n{'='*55}")
    print(f"  Q-Learning Greedy Evaluation  (seed={seed})")
    print(f"{'='*55}")

    scheduler = QLearningScheduler(
        policy_path=policy_path,
        training_mode=False,
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