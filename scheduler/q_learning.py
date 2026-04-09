# scheduler/q_learning.py
"""
Q-Learning CPU Scheduler
========================
Learns which process to schedule next by building a Q-table over a
discretised state space derived from the four features you chose:

    State = (waiting_bin, priority, avg_burst_bin, queue_len_bin)

At every scheduling decision the agent scores every candidate process
using Q(state_i) and picks the one with the highest Q-value (greedy)
or a random one (exploration).  A standard off-policy TD update is
applied after each decision using the reward signal:

    reward = -mean(waiting_time across all candidates)

The policy (Q-table) is persisted as a Pickle file so it can be saved
after training and reloaded for a greedy evaluation run.
"""

import pickle
import random
import os
from typing import Optional

from scheduler.base_scheduler import Scheduler


# ──────────────────────────────────────────────────────────────────────────────
# State discretisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bin_waiting(w: int) -> int:
    """
    Bin waiting time into 3 levels.
      0 →  0–5 ticks  (fresh / low wait)
      1 →  6–15 ticks (moderate wait)
      2 →  16+ ticks  (high wait — penalise)
    """
    if w <= 5:
        return 0
    if w <= 15:
        return 1
    return 2


def _bin_avg_burst(b: float) -> int:
    """
    Bin average CPU burst length into 3 levels.
      0 →  0–3   (short bursts — I/O bound)
      1 →  4–8   (medium)
      2 →  9+    (long / CPU bound)
    """
    if b <= 3:
        return 0
    if b <= 8:
        return 1
    return 2


def _bin_queue_len(q: int) -> int:
    """
    Bin ready-queue length into 3 levels.
      0 →  1     (only candidate)
      1 →  2–4   (small contention)
      2 →  5+    (heavy contention)
    """
    if q <= 1:
        return 0
    if q <= 4:
        return 1
    return 2


def _state_of(process, queue_length: int) -> tuple:
    """
    Build the 4-tuple state for a single candidate process.

    Dimensions
    ----------
    [0] waiting_bin   : 0/1/2
    [1] priority      : 0/1/2/3  (already discrete in your Process model)
    [2] avg_burst_bin : 0/1/2
    [3] queue_len_bin : 0/1/2
    """
    avg_burst = (
        sum(process.cpu_burst_history) / len(process.cpu_burst_history)
        if process.cpu_burst_history
        else 0.0
    )
    return (
        _bin_waiting(process.waiting_time),  # waiting_bin
        process.priority,                    # priority (0–3)
        _bin_avg_burst(avg_burst),           # avg_burst_bin
        _bin_queue_len(queue_length),        # queue_len_bin
    )


# ──────────────────────────────────────────────────────────────────────────────
# Q-Learning Scheduler
# ──────────────────────────────────────────────────────────────────────────────

class QLearningScheduler(Scheduler):
    """
    Q-Learning CPU scheduler that plugs directly into the existing
    Scheduler / CPU / Simulator stack.

    Parameters
    ----------
    alpha         : learning rate (0 < α ≤ 1)
    gamma         : discount factor (0 ≤ γ < 1)
    epsilon       : initial exploration rate (1.0 = fully random)
    epsilon_min   : floor for exploration after decay
    epsilon_decay : multiplicative decay applied after every decision
    policy_path   : path to save / load the Pickle policy file
    training_mode : True  → epsilon-greedy + TD updates + save on demand
                    False → load policy, act fully greedy (no updates)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        policy_path: str = "ml/q_policy.pkl",
        training_mode: bool = True,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.policy_path = policy_path
        self.training_mode = training_mode

        # Q-table: state (tuple) → float value
        self.q_table: dict = {}

        # TD bookkeeping: remember the state chosen + reward received
        # so we can apply the update at the *next* decision.
        self._prev_state: Optional[tuple] = None
        self._prev_reward: Optional[float] = None

        # Stats
        self.decisions_made: int = 0
        self.explorations: int = 0

        if not training_mode:
            self.load_policy()

    # ──────────────────────────────────────────────────────────────────────
    # Scheduler interface
    # ──────────────────────────────────────────────────────────────────────

    def select_process(self, ready_queue, current_time):
        """
        Called by CPU._schedule_next_process() each time a scheduling
        decision is needed.
        """
        candidates = list(ready_queue)
        queue_length = len(candidates)

        # Reward for the *current* state (used in the TD update below)
        reward = -sum(p.waiting_time for p in candidates) / queue_length

        # ── TD update for the PREVIOUS decision ───────────────────────
        if self.training_mode and self._prev_state is not None:
            self._td_update(
                prev_state=self._prev_state,
                prev_reward=self._prev_reward,
                next_candidates=candidates,
                next_queue_len=queue_length,
            )

        # ── Select next process (ε-greedy or greedy) ──────────────────
        if self.training_mode and random.random() < self.epsilon:
            selected = random.choice(candidates)
            self.explorations += 1
        else:
            selected = self._greedy_select(candidates, queue_length)

        # ── Bookkeeping ───────────────────────────────────────────────
        if self.training_mode:
            self._prev_state = _state_of(selected, queue_length)
            self._prev_reward = reward
            self._decay_epsilon()

        self.decisions_made += 1
        return selected

    # ──────────────────────────────────────────────────────────────────────
    # Q-table operations
    # ──────────────────────────────────────────────────────────────────────

    def _q(self, process, queue_length: int) -> float:
        """Look up Q-value for a process in the current queue context."""
        state = _state_of(process, queue_length)
        return self.q_table.get(state, 0.0)

    def _greedy_select(self, candidates, queue_length):
        """Return the candidate with the highest Q-value."""
        return max(candidates, key=lambda p: self._q(p, queue_length))

    def _td_update(self, prev_state, prev_reward, next_candidates, next_queue_len):
        """
        Standard Q-learning (off-policy) TD update:

            Q(s) ← Q(s) + α · [r + γ · max_a Q(s') - Q(s)]
        """
        max_next_q = (
            max(self._q(p, next_queue_len) for p in next_candidates)
            if next_candidates
            else 0.0
        )

        current_q = self.q_table.get(prev_state, 0.0)
        td_target = prev_reward + self.gamma * max_next_q
        td_error = td_target - current_q
        self.q_table[prev_state] = current_q + self.alpha * td_error

    def _decay_epsilon(self):
        """Decay exploration rate after every decision."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ──────────────────────────────────────────────────────────────────────
    # Policy persistence
    # ──────────────────────────────────────────────────────────────────────

    def save_policy(self):
        """Persist the Q-table and current epsilon to a Pickle file."""
        os.makedirs(os.path.dirname(self.policy_path) or ".", exist_ok=True)
        payload = {
            "q_table":       self.q_table,
            "epsilon":       self.epsilon,
            "alpha":         self.alpha,
            "gamma":         self.gamma,
            "decisions":     self.decisions_made,
        }
        with open(self.policy_path, "wb") as f:
            pickle.dump(payload, f)
        print(
            f"[QLearning] Policy saved → {self.policy_path} "
            f"({len(self.q_table)} states learned)"
        )

    def load_policy(self):
        """Load a previously saved Q-table from disk."""
        try:
            with open(self.policy_path, "rb") as f:
                data = pickle.load(f)
            self.q_table = data["q_table"]
            self.epsilon = data.get("epsilon", self.epsilon_min)
            print(
                f"[QLearning] Policy loaded ← {self.policy_path} "
                f"({len(self.q_table)} states)"
            )
        except FileNotFoundError:
            print(
                f"[QLearning] No saved policy at '{self.policy_path}'. "
                f"Starting with an empty Q-table."
            )

    # ──────────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────────

    def print_q_table(self):
        """Pretty-print the Q-table (useful for small tables / debugging)."""
        if not self.q_table:
            print("[QLearning] Q-table is empty.")
            return

        print(f"\n{'─'*62}")
        print(f"  Q-Table  ({len(self.q_table)} unique states)")
        print(f"  State = (wait_bin, priority, avg_burst_bin, queue_len_bin)")
        print(f"{'─'*62}")
        for state, value in sorted(self.q_table.items(), key=lambda x: -x[1]):
            print(f"  {str(state):<30}  Q = {value:+.4f}")
        print(f"{'─'*62}\n")

    def stats(self) -> dict:
        return {
            "decisions":     self.decisions_made,
            "explorations":  self.explorations,
            "exploitations": self.decisions_made - self.explorations,
            "epsilon":       round(self.epsilon, 4),
            "q_states":      len(self.q_table),
        }
