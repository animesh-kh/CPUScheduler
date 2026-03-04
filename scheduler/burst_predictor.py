# scheduler/burst_predictor.py
#
# Online ML model to predict the NEXT CPU burst length of a process.
#
# Strategy:
#   - Features are extracted from a process's observable burst history.
#   - An online linear regression (SGD) is trained incrementally after
#     every completed burst, so the model improves as the simulation runs.
#   - Falls back to exponential averaging (the classic OS heuristic) when
#     there is not enough data to train (cold-start).
#
# No external libraries required beyond the Python standard library.

from typing import List


# ─────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────

def extract_features(burst_history: List[int]) -> List[float]:
    """
    Build a fixed-length feature vector from a process's burst history.

    Features (7 total + bias):
        0. bias term (always 1.0)
        1. last burst length
        2. second-to-last burst (or last if only one exists)
        3. mean of all past bursts
        4. min of all past bursts
        5. max of all past bursts
        6. trend  = last - second_to_last  (positive → bursts growing)
        7. count  = number of past bursts (saturation-clipped to 10)
    """
    n = len(burst_history)

    if n == 0:
        # No history at all – return zero vector (predictor will use fallback)
        return [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    last    = float(burst_history[-1])
    second  = float(burst_history[-2]) if n >= 2 else last
    mean    = sum(burst_history) / n
    minimum = float(min(burst_history))
    maximum = float(max(burst_history))
    trend   = last - second
    count   = float(min(n, 10))

    return [1.0, last, second, mean, minimum, maximum, trend, count]


# ─────────────────────────────────────────────────────────────
# Online Linear Regression (SGD)
# ─────────────────────────────────────────────────────────────

class OnlineLR:
    """
    Stochastic Gradient Descent linear regression.
    Shared globally across ALL processes so it learns from every burst.
    """

    NUM_FEATURES = 8   # must match len(extract_features(...))

    def __init__(self, learning_rate: float = 0.01, l2: float = 1e-4):
        self.lr  = learning_rate
        self.l2  = l2                                    # L2 regularisation
        self.w   = [0.0] * self.NUM_FEATURES             # weights
        self.n_samples = 0                               # training samples seen

    # ── forward pass ──────────────────────────────────────

    def predict(self, features: List[float]) -> float:
        return max(1.0, sum(w * x for w, x in zip(self.w, features)))

    # ── single SGD update ─────────────────────────────────

    def update(self, features: List[float], actual: float):
        """
        One gradient step: minimise 0.5*(predicted - actual)^2 + L2 penalty.
        """
        predicted = self.predict(features)
        error     = predicted - actual

        for i in range(self.NUM_FEATURES):
            grad      = error * features[i] + self.l2 * self.w[i]
            self.w[i] -= self.lr * grad

        self.n_samples += 1

    @property
    def is_trained(self) -> bool:
        """True once the model has seen at least 5 samples."""
        return self.n_samples >= 5

    def __repr__(self):
        return (f"OnlineLR(samples={self.n_samples}, "
                f"w={[round(x, 3) for x in self.w]})")


# ─────────────────────────────────────────────────────────────
# Burst Predictor (per-process wrapper + global model)
# ─────────────────────────────────────────────────────────────

# One shared model for the entire simulation
_global_model = OnlineLR(learning_rate=0.01)


def reset_global_model():
    """Call this between simulation runs to start fresh."""
    global _global_model
    _global_model = OnlineLR(learning_rate=0.01)


class BurstPredictor:
    """
    Per-process predictor.

    Usage (called by the simulator / scheduler):
        predictor = BurstPredictor()           # one per process
        est = predictor.estimate(history)      # get next-burst estimate
        predictor.record_actual(history, val)  # after burst completes
    """

    ALPHA = 0.5   # exponential-averaging smoothing factor (cold-start)

    def __init__(self):
        self._exp_avg: float = 5.0   # initial guess (tune if needed)

    # ── public API ────────────────────────────────────────

    def estimate(self, burst_history: List[int]) -> float:
        """
        Return predicted next CPU burst length (always >= 1).
        Uses global ML model when trained, else exponential averaging.
        """
        model = _global_model

        if model.is_trained and len(burst_history) > 0:
            features = extract_features(burst_history)
            return model.predict(features)
        else:
            return self._exp_avg

    def record_actual(self, burst_history: List[int], actual_burst: int):
        """
        Called AFTER a burst completes.
        Updates both the exponential average and the shared ML model.

        Args:
            burst_history : history BEFORE the completed burst
            actual_burst  : the burst length that just finished
        """
        # 1. Update exponential average (classic τ_n+1 = α*t_n + (1-α)*τ_n)
        self._exp_avg = (self.ALPHA * actual_burst
                         + (1 - self.ALPHA) * self._exp_avg)

        # 2. Train global model IF we had history to form features
        if len(burst_history) > 0:
            features = extract_features(burst_history)
            _global_model.update(features, float(actual_burst))

    # ── debug ─────────────────────────────────────────────

    def __repr__(self):
        return (f"BurstPredictor(exp_avg={self._exp_avg:.2f}, "
                f"model={_global_model})")