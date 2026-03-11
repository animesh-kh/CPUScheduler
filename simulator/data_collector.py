# simulator/data_collector.py
"""
DataCollector  —  Long / Normalised Format
==========================================
One row per PROCESS per scheduling decision (not one row per decision).
Multiple rows sharing the same decision_id represent the same decision moment.

Schema
------
decision_id       : unique integer per scheduling decision (groups rows together)
tick              : system clock at decision time
scheduler         : name of the scheduler that made this decision
queue_length      : total number of candidates in this decision
reward            : -mean(waiting_time) across all candidates — RL reward signal

-- Process features (one row per candidate) --
pid               : process identifier
priority          : priority level (0 = highest, 3 = lowest)
waiting           : ticks spent in READY state so far
burst_count       : number of CPU bursts completed (experience proxy)
avg_burst         : mean length of completed bursts (0.0 if none yet)
time_in_system    : tick - arrival_time  (total age in the system)
was_selected      : 1 if this process was chosen to run, 0 otherwise (label)

-- Run metadata --
seed              : random seed of this simulation run
arrival_prob      : process arrival probability used in this run
avg_burst_config  : avg_burst_time setting used in this run
"""

import csv
import os
from typing import List


class DataCollector:
    def __init__(self, scheduler_name: str):
        self.scheduler_name = scheduler_name
        self._rows: List[dict] = []
        self._decision_counter = 0

    # ------------------------------------------------------------------
    # Public hook — called by CPU._schedule_next_process()
    # ------------------------------------------------------------------

    def record_decision(self, current_time: int, ready_queue, selected_process):
        """
        Emit one row per candidate process in the ready queue.

        Parameters
        ----------
        current_time     : clock.now() at decision time
        ready_queue      : ReadyQueue (iterable); selected process not yet removed
        selected_process : the process the scheduler chose
        """
        self._decision_counter += 1
        decision_id = self._decision_counter

        candidates = list(ready_queue)

        # Reward: negative mean waiting time across all candidates
        reward = -sum(p.waiting_time for p in candidates) / len(candidates)

        for process in candidates:
            avg_burst = (
                sum(process.cpu_burst_history) / len(process.cpu_burst_history)
                if process.cpu_burst_history else 0.0
            )

            self._rows.append({
                # Decision context
                "decision_id":    decision_id,
                "tick":           current_time,
                "scheduler":      self.scheduler_name,
                "queue_length":   len(candidates),
                "reward":         round(reward, 4),

                # Per-process features
                "pid":            process.pid,
                "priority":       process.priority,
                "waiting":        process.waiting_time,
                "burst_count":    len(process.cpu_burst_history),
                "avg_burst":      round(avg_burst, 2),
                "time_in_system": current_time - process.arrival_time,

                # Label
                "was_selected":   1 if process is selected_process else 0,
            })

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save_csv(self, path: str):
        if not self._rows:
            print(f"[DataCollector] No rows to save for '{self.scheduler_name}'.")
            return
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        fieldnames = list(self._rows[0].keys())
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self._rows)
        print(f"[DataCollector] Saved {len(self._rows)} rows -> {path}")

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def row_count(self) -> int:
        return len(self._rows)

    def get_rows(self) -> List[dict]:
        return list(self._rows)

    def clear(self):
        self._rows.clear()
        self._decision_counter = 0