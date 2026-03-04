# main.py  —  Example: run simulator with ML-powered SJF scheduler

from simulator.clock           import SystemClock
from simulator.process         import ProcessState
from simulator.ready_queue     import ReadyQueue
from simulator.cpu             import CPU
from simulator.simulator       import Simulator
from simulator.process_generator import ProcessGenerator

from scheduler.sjf_ml_scheduler import SJFMLScheduler


def main():
    # ── Config ────────────────────────────────────────────
    TIME_QUANTUM   = 100      # large quantum → near non-preemptive SJF
    MAX_TIME       = 500
    ARRIVAL_PROB   = 0.4
    SEED           = 42
    VERBOSE_ML     = True     # print prediction table each scheduling event

    # ── Wire components ───────────────────────────────────
    clock     = SystemClock()
    scheduler = SJFMLScheduler(verbose=VERBOSE_ML)
    rq        = ReadyQueue()
    cpu       = CPU(clock, scheduler, rq, time_quantum=TIME_QUANTUM)
    gen       = ProcessGenerator(
                    arrival_probability=ARRIVAL_PROB,
                    max_bursts=5,
                    avg_burst_time=6,
                    seed=SEED
                )

    sim = Simulator(clock, gen, rq, cpu, max_time=MAX_TIME)

    # ── Run ───────────────────────────────────────────────
    print("=" * 60)
    print("  ML-Powered SJF Process Scheduling Simulator")
    print("=" * 60)

    sim.run()

    # ── Summary ───────────────────────────────────────────
    processes = sim.all_processes
    terminated = [p for p in processes if p.state.value == "TERMINATED"]

    print("\n" + "=" * 60)
    print(f"  Simulation finished at t={clock.now()}")
    print(f"  Total processes    : {len(processes)}")
    print(f"  Completed          : {len(terminated)}")
    print(f"  Context switches   : {cpu.context_switches}")

    if terminated:
        avg_wait = sum(p.waiting_time for p in terminated) / len(terminated)
        avg_cpu  = sum(p.total_cpu_time for p in terminated) / len(terminated)
        print(f"  Avg waiting time   : {avg_wait:.2f}")
        print(f"  Avg CPU time used  : {avg_cpu:.2f}")

    # ── ML model diagnostics ──────────────────────────────
    scheduler.print_model_weights()


if __name__ == "__main__":
    main()