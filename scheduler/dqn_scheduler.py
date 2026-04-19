# scheduler/dqn_scheduler.py
"""
Deep Q-Network (DQN) Scheduler
================================
Each candidate process is scored independently by a small neural network;
the highest-scoring candidate is dispatched.

Key differences from tabular Q-Learning
----------------------------------------
- Continuous feature vector instead of discretised state tuple
- Experience replay buffer for decorrelated, stable learning
- Separate target network (synced every N decisions) for stable TD targets
- Gradient clipping to prevent exploding gradients

Reward (identical to QLearningScheduler for a fair apples-to-apples comparison)
---------------------------------------------------------------------------------
  priority_term   = -8.0 * priority        → range [0, -24]
  starvation_term = min(waiting * 0.02, 0.5)  → range [0, +0.5]
  reward          = priority_term + starvation_term

Feature vector (INPUT_DIM = 6, all normalised to roughly [0, 1])
-----------------------------------------------------------------
  0  waiting_time   / WAIT_NORM   (urgency signal)
  1  priority       / PRIO_NORM   (0=high priority → 0.0)
  2  avg_burst      / BURST_NORM  (CPU-intensity proxy)
  3  queue_length   / QUEUE_NORM  (contention context)
  4  burst_count    / COUNT_NORM  (experience proxy)
  5  time_in_system / TIME_NORM   (overall age)
"""

import os
import random
from collections import deque
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim

from scheduler.base_scheduler import Scheduler


# ══════════════════════════════════════════════════════════════════════════════
# Constants
# ══════════════════════════════════════════════════════════════════════════════

INPUT_DIM   = 6
_WAIT_NORM  = 50.0
_PRIO_NORM  = 3.0
_BURST_NORM = 10.0
_QUEUE_NORM = 10.0
_COUNT_NORM = 10.0
_TIME_NORM  = 200.0


# ══════════════════════════════════════════════════════════════════════════════
# Feature Extraction
# ══════════════════════════════════════════════════════════════════════════════

def _extract_features(process, queue_length: int, current_time: int) -> List[float]:
    """Return a normalised 6-dim feature vector for one candidate process."""
    avg_burst = (
        sum(process.cpu_burst_history) / len(process.cpu_burst_history)
        if process.cpu_burst_history else 0.0
    )
    return [
        min(process.waiting_time                     / _WAIT_NORM,  3.0),
        process.priority                             / _PRIO_NORM,
        min(avg_burst                                / _BURST_NORM, 3.0),
        min(queue_length                             / _QUEUE_NORM, 3.0),
        min(len(process.cpu_burst_history)           / _COUNT_NORM, 3.0),
        min((current_time - process.arrival_time)    / _TIME_NORM,  3.0),
    ]


# ══════════════════════════════════════════════════════════════════════════════
# Neural Network
# ══════════════════════════════════════════════════════════════════════════════

class QNetwork(nn.Module):
    """
    Maps a process feature vector → scalar Q-value.
    Architecture:  INPUT_DIM → hidden → hidden → 1
    """

    def __init__(self, input_dim: int = INPUT_DIM, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (batch, input_dim)  →  output: (batch,)"""
        return self.net(x).squeeze(-1)


# ══════════════════════════════════════════════════════════════════════════════
# Replay Buffer
# ══════════════════════════════════════════════════════════════════════════════

class ReplayBuffer:
    """
    Stores transitions  (selected_feat, reward, next_feats, done).

    selected_feat : 6-dim feature vector of the process chosen at time t
    reward        : scalar reward received after the dispatch
    next_feats    : list of 6-dim vectors for all candidates at time t+1
                    (empty list when the episode ends)
    done          : True when there is no follow-up decision in this episode
    """

    def __init__(self, capacity: int = 20_000):
        self._buf = deque(maxlen=capacity)

    def push(
        self,
        selected_feat: List[float],
        reward: float,
        next_feats: List[List[float]],
        done: bool,
    ):
        self._buf.append((selected_feat, reward, next_feats, done))

    def sample(self, batch_size: int):
        return random.sample(self._buf, batch_size)

    def __len__(self) -> int:
        return len(self._buf)


# ══════════════════════════════════════════════════════════════════════════════
# DQN Scheduler
# ══════════════════════════════════════════════════════════════════════════════

class DQNScheduler(Scheduler):
    """
    Deep Q-Network CPU scheduler.

    During training (training_mode=True):
      - Collects transitions into a replay buffer.
      - After each decision, samples a mini-batch and performs one gradient step.
      - Target network is synced every `target_update_freq` gradient steps.
      - ε decays exponentially from 1.0 → epsilon_min.

    During evaluation (training_mode=False):
      - Loads a saved policy and acts greedily (ε = 0).
      - No gradient computation, no buffer writes.
    """

    def __init__(
        self,
        alpha: float          = 1e-3,    # Adam learning rate
        gamma: float          = 0.9,     # discount factor
        epsilon: float        = 1.0,     # initial exploration rate
        epsilon_min: float    = 0.05,
        epsilon_decay: float  = 0.9998,
        batch_size: int       = 64,
        target_update_freq: int = 50,    # gradient steps between target syncs
        replay_capacity: int  = 20_000,
        hidden_dim: int       = 64,
        policy_path: str      = "ml/dqn_policy.pt",
        training_mode: bool   = True,
    ):
        self.gamma            = gamma
        self.epsilon          = epsilon
        self.epsilon_min      = epsilon_min
        self.epsilon_decay    = epsilon_decay
        self.batch_size       = batch_size
        self.target_update_freq = target_update_freq
        self.policy_path      = policy_path
        self.training_mode    = training_mode

        # ── Networks ──────────────────────────────────────────────────
        self.policy_net = QNetwork(INPUT_DIM, hidden_dim)
        self.target_net = QNetwork(INPUT_DIM, hidden_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=alpha)
        self.loss_fn    = nn.MSELoss()

        # ── Replay buffer ─────────────────────────────────────────────
        self.replay = ReplayBuffer(replay_capacity)

        # ── Per-decision memory (for storing the PREVIOUS transition) ─
        self._prev_feat:     Optional[List[float]] = None
        self._prev_priority: int   = 0
        self._prev_wait:     float = 0.0

        # ── Counters ──────────────────────────────────────────────────
        self.decisions_made: int = 0
        self.explorations:   int = 0
        self._grad_steps:    int = 0   # gradient update counter

        if not training_mode:
            self.load_policy()

    # ══════════════════════════════════════════════════════════════════
    # Scheduler interface
    # ══════════════════════════════════════════════════════════════════

    def select_process(self, ready_queue, current_time: int):
        candidates = list(ready_queue)
        q_len      = len(candidates)

        # ── Step 1: Store and learn from the PREVIOUS decision ────────
        if self.training_mode and self._prev_feat is not None:
            priority_term   = -8.0 * self._prev_priority
            starvation_term = min(self._prev_wait * 0.02, 0.5)
            reward          = priority_term + starvation_term

            next_feats = [
                _extract_features(p, q_len, current_time) for p in candidates
            ]
            self.replay.push(self._prev_feat, reward, next_feats, done=False)
            self._learn()

        # ── Step 2: ε-greedy action selection ────────────────────────
        if self.training_mode and random.random() < self.epsilon:
            selected = random.choice(candidates)
            self.explorations += 1
        else:
            selected = self._greedy_select(candidates, q_len, current_time)

        # ── Step 3: Record this decision for the next iteration ───────
        if self.training_mode:
            self._prev_feat     = _extract_features(selected, q_len, current_time)
            self._prev_priority = selected.priority
            self._prev_wait     = selected.waiting_time
            self._decay_epsilon()

        self.decisions_made += 1
        return selected

    # ══════════════════════════════════════════════════════════════════
    # Greedy selection
    # ══════════════════════════════════════════════════════════════════

    def _greedy_select(self, candidates, q_len: int, current_time: int):
        """Score every candidate with the policy network; return the best."""
        feats = torch.tensor(
            [_extract_features(p, q_len, current_time) for p in candidates],
            dtype=torch.float32,
        )                                        # (n_candidates, INPUT_DIM)
        with torch.no_grad():
            q_vals = self.policy_net(feats)      # (n_candidates,)
        return candidates[q_vals.argmax().item()]

    # ══════════════════════════════════════════════════════════════════
    # Learning step
    # ══════════════════════════════════════════════════════════════════

    def _learn(self):
        """One gradient step on a random mini-batch from the replay buffer."""
        if len(self.replay) < self.batch_size:
            return

        batch = self.replay.sample(self.batch_size)
        sel_feats, rewards, next_feats_list, dones = zip(*batch)

        sel_t = torch.tensor(sel_feats, dtype=torch.float32)  # (B, INPUT_DIM)
        rew_t = torch.tensor(rewards,   dtype=torch.float32)  # (B,)

        # Current Q values from policy network
        current_q = self.policy_net(sel_t)   # (B,)

        # Target Q values from frozen target network
        with torch.no_grad():
            next_q_vals = []
            for nf, done in zip(next_feats_list, dones):
                if done or len(nf) == 0:
                    next_q_vals.append(0.0)
                else:
                    nf_t = torch.tensor(nf, dtype=torch.float32)  # (k, INPUT_DIM)
                    next_q_vals.append(self.target_net(nf_t).max().item())
            next_q_t = torch.tensor(next_q_vals, dtype=torch.float32)  # (B,)

        td_targets = rew_t + self.gamma * next_q_t   # Bellman target

        loss = self.loss_fn(current_q, td_targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=1.0)
        self.optimizer.step()

        # Periodically sync target network
        self._grad_steps += 1
        if self._grad_steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def _decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    # ══════════════════════════════════════════════════════════════════
    # Policy persistence
    # ══════════════════════════════════════════════════════════════════

    def save_policy(self):
        os.makedirs(os.path.dirname(self.policy_path) or ".", exist_ok=True)
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "epsilon":    self.epsilon,
                "decisions":  self.decisions_made,
            },
            self.policy_path,
        )
        print(f"[DQN] Policy saved → {self.policy_path}")

    def load_policy(self):
        try:
            data = torch.load(self.policy_path, map_location="cpu")
            self.policy_net.load_state_dict(data["policy_net"])
            self.target_net.load_state_dict(data["target_net"])
            self.epsilon = data.get("epsilon", self.epsilon_min)
            print(f"[DQN] Policy loaded ← {self.policy_path}")
        except FileNotFoundError:
            print(
                f"[DQN] No saved policy at '{self.policy_path}'. "
                "Starting with a randomly initialised network."
            )

    # ══════════════════════════════════════════════════════════════════
    # Episode reset  (called by dqn_trainer between episodes)
    # ══════════════════════════════════════════════════════════════════

    def reset_episode(self):
        """
        Clear per-episode memory so stale transitions from the last
        decision of episode N do not corrupt episode N+1's first update.
        """
        self._prev_feat     = None
        self._prev_priority = 0
        self._prev_wait     = 0.0

    # ══════════════════════════════════════════════════════════════════
    # Diagnostics
    # ══════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        return {
            "decisions":     self.decisions_made,
            "explorations":  self.explorations,
            "exploitations": self.decisions_made - self.explorations,
            "epsilon":       round(self.epsilon, 4),
            "replay_size":   len(self.replay),
            "grad_steps":    self._grad_steps,
        }