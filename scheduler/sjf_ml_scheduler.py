# scheduler/sjf_ml_scheduler.py
#
# ML-powered Shortest Job First (SJF) Scheduler
#
# Each process carries its own BurstPredictor.
# On every scheduling decision the scheduler asks each ready process
# "how long do you think your next burst will be?" and picks the
# process with the SMALLEST predicted burst — i.e. SJF.
#
# After a burst actually completes the predictor is updated so the
# model learns from real execution data (online learning).

from scheduler.base_scheduler import Scheduler
from scheduler.burst_predictor import BurstPredictor


class SJFMLScheduler(Scheduler):
    """
    Shortest Job First scheduler backed by an ML burst predictor.

    The scheduler:
      1. Attaches a BurstPredictor to each process the first time it sees it.
      2. Calls predictor.estimate() to get the predicted next burst.
      3. Selects the process with the minimum prediction.
      4. Calls predictor.record_actual() after each burst to train the model.

    Because record_actual() must be called by the CPU after a burst
    finishes, a helper method notify_burst_complete() is provided and
    should be called from cpu.py (see integration note below).
    """

    def __init__(self, verbose: bool = False):
        """
        Args:
            verbose: if True, prints prediction table at every scheduling event.
        """
        self._predictors: dict[int, BurstPredictor] = {}  # pid → predictor
        self.verbose = verbose

    # ─────────────────────────────────────────────────────
    # Scheduler interface
    # ─────────────────────────────────────────────────────

    def select_process(self, ready_queue, current_time: int):
        """
        Pick the process with the shortest PREDICTED next CPU burst.
        """
        best_process   = None
        best_estimate  = float('inf')

        if self.verbose:
            print(f"\n[SJF-ML] Scheduling at t={current_time}")
            print(f"  {'PID':>4}  {'History':>30}  {'Prediction':>10}")

        for process in ready_queue:
            predictor = self._get_or_create_predictor(process)
            estimate  = predictor.estimate(process.cpu_burst_history)

            if self.verbose:
                print(f"  {process.pid:>4}  "
                      f"{str(process.cpu_burst_history):>30}  "
                      f"{estimate:>10.2f}")

            if estimate < best_estimate:
                best_estimate  = estimate
                best_process   = process

        if self.verbose and best_process:
            print(f"  → Selected PID {best_process.pid} "
                  f"(predicted burst = {best_estimate:.2f})")

        return best_process

    # ─────────────────────────────────────────────────────
    # Called by CPU after a burst completes  ← INTEGRATION HOOK
    # ─────────────────────────────────────────────────────

    def notify_burst_complete(self, process, actual_burst: int):
        """
        Train the predictor after a real burst finishes.

        Integration note
        ─────────────────
        In cpu.py, inside _execute_one_tick(), just before calling
        process.complete_burst(), add:

            if hasattr(self.scheduler, 'notify_burst_complete'):
                # history BEFORE this burst completes
                self.scheduler.notify_burst_complete(
                    process,
                    self.ticks_in_current_quantum
                )

        This keeps the CPU code backward-compatible with other schedulers.
        """
        history_before_burst = process.cpu_burst_history.copy()
        predictor = self._get_or_create_predictor(process)
        predictor.record_actual(history_before_burst, actual_burst)

    # ─────────────────────────────────────────────────────
    # Diagnostics
    # ─────────────────────────────────────────────────────

    def print_model_weights(self):
        """Print the shared ML model's learned weights."""
        from scheduler.burst_predictor import _global_model
        print("\n[SJF-ML] Global model state:")
        print(f"  Samples trained on : {_global_model.n_samples}")
        labels = ["bias", "last", "second", "mean", "min", "max", "trend", "count"]
        for label, w in zip(labels, _global_model.w):
            print(f"  {label:>8} : {w:+.4f}")

    # ─────────────────────────────────────────────────────
    # Internal helpers
    # ─────────────────────────────────────────────────────

    def _get_or_create_predictor(self, process) -> BurstPredictor:
        if process.pid not in self._predictors:
            self._predictors[process.pid] = BurstPredictor()
        return self._predictors[process.pid]