# ml/dqn_trainer.py
"""
DQN Trainer
===========
Mirrors ml/q_trainer.py exactly — same function signatures, same log format,
same evaluation seed — so the two can be compared side-by-side in compare.py.
"""

from simulator.clock import SystemClock
from simulator.process_generator import ProcessGenerator
from simulator.ready_queue import ReadyQueue
from simulator.cpu import CPU
from simulator.simulator import Simulator
from metrics.metrics_collector import MetricsCollector
from scheduler.dqn_scheduler import DQNScheduler


# ══════════════════════════════════════════════════════════════════════════════
# Training
# ══════════════════════════════════════════════════════════════════════════════

def train(
    episodes: int           = 100,
    max_time: int           = 500,
    time_quantum: int       = 10,
    arrival_probability: float = 0.3,
    avg_burst_time: int     = 5,
    policy_path: str        = "ml/dqn_policy.pt",
    base_seed: int          = 0,
    resume: bool            = True,
    alpha: float            = 1e-3,        # Adam learning rate
    gamma: float            = 0.9,
    epsilon_decay: float    = 0.9998,
    batch_size: int         = 64,
    target_update_freq: int = 50,
    hidden_dim: int         = 64,
    log_every: int          = 10,
) -> DQNScheduler:
    """
    Train a DQN scheduler for `episodes` simulation runs.

    Each episode uses a fresh environment (new clock, queue, generator) but
    the same scheduler instance so the network weights accumulate across
    episodes.  Training is online: one gradient step per scheduling decision.

    Parameters
    ----------
    episodes         : number of simulation episodes to train for
    max_time         : maximum clock ticks per episode
    time_quantum     : CPU time-slice length
    arrival_probability : P(new process arrives each tick)
    avg_burst_time   : mean CPU burst length (exponential distribution)
    policy_path      : where to save / load the .pt checkpoint
    base_seed        : episode `ep` uses seed `base_seed + ep`
    resume           : load existing policy before training if it exists
    alpha            : Adam learning rate
    gamma            : TD discount factor
    epsilon_decay    : multiplicative decay per scheduling decision
    batch_size       : replay mini-batch size
    target_update_freq : gradient steps between target-net syncs
    hidden_dim       : width of each hidden layer in the Q-network
    log_every        : print a progress line every N episodes

    Returns
    -------
    Trained DQNScheduler (policy_net weights updated in-place).
    """

    print(f"\n{'='*60}")
    print(f"  DQN Training")
    print(f"  Episodes={episodes}  max_time={max_time}  lr={alpha}  γ={gamma}")
    print(f"  batch={batch_size}  target_sync_every={target_update_freq} steps")
    print(f"{'='*60}")

    scheduler = DQNScheduler(
        alpha=alpha,
        gamma=gamma,
        epsilon=1.0,
        epsilon_min=0.05,
        epsilon_decay=epsilon_decay,
        batch_size=batch_size,
        target_update_freq=target_update_freq,
        hidden_dim=hidden_dim,
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
        # ── Fresh environment every episode ───────────────────────────
        clock       = SystemClock()
        ready_queue = ReadyQueue()
        pg          = ProcessGenerator(
            arrival_probability=arrival_probability,
            avg_burst_time=avg_burst_time,
            seed=base_seed + ep,
        )
        cpu = CPU(clock, scheduler, ready_queue, time_quantum)
        sim = Simulator(clock, pg, ready_queue, cpu, max_time=max_time)

        # ── Clear per-episode memory on the scheduler ─────────────────
        scheduler.reset_episode()

        sim.run()

        if ep % log_every == 0 or ep == episodes:
            mc      = MetricsCollector(sim.all_processes, clock.now(), cpu.busy_ticks)
            awt     = mc.average_waiting_time()
            s       = scheduler.stats()
            awt_str = f"{awt:.2f}" if awt is not None else "N/A"
            print(
                f"  Ep {ep:>4}/{episodes}  |  "
                f"ε={s['epsilon']:.3f}  |  "
                f"replay={s['replay_size']:>6}  |  "
                f"grad_steps={s['grad_steps']:>5}  |  "
                f"AvgWait={awt_str}"
            )

    scheduler.save_policy()
    s = scheduler.stats()
    print(f"\n  Total decisions    : {s['decisions']}")
    print(f"  Explore / Exploit  : {s['explorations']} / {s['exploitations']}")
    print(f"  Gradient steps     : {s['grad_steps']}")
    print(f"{'='*60}\n")

    return scheduler


# ══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(
    policy_path: str           = "ml/dqn_policy.pt",
    max_time: int              = 500,
    time_quantum: int          = 10,
    arrival_probability: float = 0.3,
    avg_burst_time: int        = 5,
    seed: int                  = 9999,    # identical to q_trainer.evaluate()
) -> MetricsCollector:
    """
    Run one greedy episode with the saved DQN policy and return metrics.
    Uses the same default seed as q_trainer.evaluate() for fair comparison.
    """

    print(f"\n{'='*60}")
    print(f"  DQN Greedy Evaluation  (seed={seed})")
    print(f"{'='*60}")

    scheduler = DQNScheduler(
        policy_path=policy_path,
        training_mode=False,   # ε=0, no buffer writes, no gradient steps
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
    mc.print_report("DQN (Greedy)")
    return mc