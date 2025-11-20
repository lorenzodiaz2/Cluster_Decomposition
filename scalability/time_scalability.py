from solvers.heuristic_solver import Heuristic_Solver
from utils.environment import Environment

import matplotlib.pyplot as plt


counter = 0
max_count = 10
step = 5
pairs_per_quadrant = 10
grid_side = 30
offset = grid_side // 2
k = 12

results = {}

while counter < max_count:
    print(f"grid side = {grid_side}   total number of pairs = {pairs_per_quadrant * 4}... ", end="")
    env = Environment(grid_side, 0, pairs_per_quadrant, offset, k)
    solver = Heuristic_Solver(env.G, env.od_pairs, env.T)
    solver.solve()

    total_time = sum(m_times for m_times in solver.model_times) + sum(r_time for r_time in solver.resolution_times)
    results[pairs_per_quadrant * 4] = total_time

    counter_updated = "COUNTER UPDATED" if solver.status == "TIME_LIMIT" else ""
    print(f"Solved in {solver.model_times} + {solver.resolution_times} = {total_time}     {counter_updated}")

    if solver.status == "TIME_LIMIT":
        counter += 1

    pairs_per_quadrant += step


od_pairs = []
times = []

with open("../time_scalability_results", "w") as f:
    f.write(f"Grid size = {grid_side * grid_side}   number of pairs = {pairs_per_quadrant * 4}\n\n")
    for pair, time in results.items():
        f.write(f"{pair}:{time}\n")
        od_pairs.append(pair)
        times.append(time)


plt.plot(od_pairs, times, marker="o")
plt.xlabel("od")
plt.ylabel("Time")
plt.grid(True)
plt.tight_layout()

plt.savefig("scalability.png")
