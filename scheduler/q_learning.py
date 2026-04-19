# scheduler/q_learning.py

import pickle
import random
import os
from typing import Optional

from scheduler.base_scheduler import Scheduler


# ──────────────────────────────────────────────────────────────────────────────
# State discretisation helpers
# ──────────────────────────────────────────────────────────────────────────────

def _bin_waiting(w: int) -> int:
    if w <= 5:
        return 0
    if w <= 15:
        return 1
    return 2


def _bin_avg_burst(b: float) -> int:
    if b <= 3:
        return 0
    if b <= 8:
        return 1
    return 2


def _bin_queue_len(q: int) -> int:
    if q <= 1:
        return 0
    if q <= 4:
        return 1
    return 2


def _state_of(process, queue_length: int) -> tuple:
    avg_burst = (
        sum(process.cpu_burst_history) / len(process.cpu_burst_history)
        if process.cpu_burst_history
        else 0.0
    )
    return (
        _bin_waiting(process.waiting_time),
        process.priority,
        _bin_avg_burst(avg_burst),
        _bin_queue_len(queue_length),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Q-Learning Scheduler
# ──────────────────────────────────────────────────────────────────────────────

class QLearningScheduler(Scheduler):

    def __init__(
        self,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9998,
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

        self.q_table: dict = {}

        # ── Previous-decision memory (for TD update) ──────────────────
        self._prev_state: Optional[tuple] = None
        self._prev_selected_wait: float = 0.0
        self._prev_priority: int = 0

        self.decisions_made: int = 0
        self.explorations: int = 0

        if not training_mode:
            self.load_policy()

    # ──────────────────────────────────────────────────────────────────
    # Scheduler interface
    # ──────────────────────────────────────────────────────────────────

    def select_process(self, ready_queue, current_time):
        candidates = list(ready_queue)
        queue_length = len(candidates)

        # ── Step 1: TD update for the PREVIOUS decision ───────────────
        #
        # Reward design:
        #   Priority term  : -8.0 * priority
        #                    → priority 0 = 0, priority 3 = -24
        #                    → 24-point spread forces clear ordering
        #
        #   Starvation term: small bonus (capped at +2) for picking a
        #                    process that has waited a long time.
        #                    Acts only as a tiebreaker between equal
        #                    priorities — never overrides priority signal.
        #
        # Together: priority 0 with high wait ≈ +2  (best)
        #           priority 0 with low wait  ≈  0
        #           priority 3 with high wait ≈ -22 (always worse than any priority 0)
        #           priority 3 with low wait  ≈ -24 (worst)
        #
        if self.training_mode and self._prev_state is not None:
            priority_term   = -8.0 * self._prev_priority
            starvation_term = min(self._prev_selected_wait * 0.02, 0.5)
            prev_reward     = priority_term + starvation_term

            self._td_update(
                prev_state=self._prev_state,
                prev_reward=prev_reward,
                next_candidates=candidates,
                next_queue_len=queue_length,
            )

        # ── Step 2: Select next process ───────────────────────────────
        if self.training_mode and random.random() < self.epsilon:
            selected = random.choice(candidates)
            self.explorations += 1
        else:
            selected = self._greedy_select(candidates, queue_length)

        # ── Step 3: Save memory for scoring THIS decision next time ───
        if self.training_mode:
            self._prev_state         = _state_of(selected, queue_length)
            self._prev_selected_wait = selected.waiting_time
            self._prev_priority      = selected.priority
            self._decay_epsilon()

        self.decisions_made += 1
        return selected

    # ──────────────────────────────────────────────────────────────────
    # Q-table operations
    # ──────────────────────────────────────────────────────────────────

    def _q(self, process, queue_length: int) -> float:
        state = _state_of(process, queue_length)
        return self.q_table.get(state, 0.0)

    def _greedy_select(self, candidates, queue_length):
        return max(candidates, key=lambda p: self._q(p, queue_length))

    def _td_update(self, prev_state, prev_reward, next_candidates, next_queue_len):
        max_next_q = (
            max(self._q(p, next_queue_len) for p in next_candidates)
            if next_candidates
            else 0.0
        )
        current_q = self.q_table.get(prev_state, 0.0)
        td_target = prev_reward + self.gamma * max_next_q
        td_error  = td_target - current_q
        self.q_table[prev_state] = current_q + self.alpha * td_error

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ──────────────────────────────────────────────────────────────────
    # Policy persistence
    # ──────────────────────────────────────────────────────────────────

    def save_policy(self):
        os.makedirs(os.path.dirname(self.policy_path) or ".", exist_ok=True)
        payload = {
            "q_table":   self.q_table,
            "epsilon":   self.epsilon,
            "alpha":     self.alpha,
            "gamma":     self.gamma,
            "decisions": self.decisions_made,
        }
        with open(self.policy_path, "wb") as f:
            pickle.dump(payload, f)
        print(
            f"[QLearning] Policy saved → {self.policy_path} "
            f"({len(self.q_table)} states learned)"
        )

    def load_policy(self):
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

    # ──────────────────────────────────────────────────────────────────
    # Diagnostics
    # ──────────────────────────────────────────────────────────────────

    def print_q_table(self):
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