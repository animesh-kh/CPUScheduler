from simulator.clock import SystemClock
from simulator.process_generator import ProcessGenerator
from simulator.ready_queue import ReadyQueue
from simulator.cpu import CPU
from simulator.simulator import Simulator
from scheduler.fcfs import FCFSScheduler
from simulator.manual_process_generator import ManualProcessGenerator

# ----------------------------
# Create core components
# ----------------------------
clock = SystemClock()
generator = ManualProcessGenerator()
ready_queue = ReadyQueue()

scheduler = FCFSScheduler()
cpu = CPU(
    clock=clock,
    scheduler=scheduler,
    ready_queue=ready_queue,
    time_quantum=1
)

simulator = Simulator(
    clock=clock,
    process_generator=generator,
    ready_queue=ready_queue,
    cpu=cpu,
    max_time=20
)

# ----------------------------
# Run simulation
# ----------------------------
simulator.run()

print("\nSimulation finished\n")

for p in simulator.all_processes:
    print(
        f"PID={p.pid}, "
        f"arrival={p.arrival_time}, "
        f"waiting={p.waiting_time}, "
        f"bursts={p.cpu_burst_history}"
    )
