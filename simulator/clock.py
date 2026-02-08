# simulator/clock.py

class SystemClock:
    def __init__(self, start_time: int = 0):
        """
        Discrete system clock for the simulator.
        """
        self._time = start_time

    def tick(self, ticks: int = 1):
        """
        Advance the clock by a given number of ticks.
        """
        if ticks < 0:
            raise ValueError("Clock ticks must be non-negative")
        self._time += ticks

    def now(self) -> int:
        """
        Get the current system time.
        """
        return self._time

    def reset(self):
        """
        Reset the clock to time 0.
        """
        self._time = 0

    def __repr__(self):
        return f"SystemClock(time={self._time})"
