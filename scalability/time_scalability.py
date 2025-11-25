import os
import matplotlib.pyplot as plt

from solvers.heuristic_solver import Heuristic_Solver
from utils.environment import Environment


def save_img(results, i: int | None = None):
    plt.plot(results.keys(), results.values(), marker="o", markersize=3)
    plt.xlabel("od")
    plt.ylabel("Time")
    plt.grid(True)
    plt.tight_layout()

    if i:
        plt.savefig(f"img/scalability_{i}.png")
    else:
        plt.savefig(f"img/scalability.png")


def run():
    max_time_counter = 0
    max_count = 4
    step = 1
    pairs_per_quadrant = 2
    grid_side = 20
    offset = grid_side // 2
    k = 15
    i = 0

    results = {}

    if not os.path.isdir("img"):
        os.makedirs("img")

    while max_time_counter < max_count:
        print(f"i = {i}     grid side = {grid_side}   total number of pairs = {pairs_per_quadrant * 4}... ", end="")
        env = Environment(grid_side, 0, pairs_per_quadrant, offset, k)
        solver = Heuristic_Solver(env.G, env.od_pairs)
        solver.solve()

        total_time = sum(m_times for m_times in solver.model_times) + sum(r_time for r_time in solver.resolution_times)
        results[pairs_per_quadrant * 4] = total_time

        counter_updated = "COUNTER UPDATED" if solver.status == "TIME_LIMIT" else ""
        print(f"Solved in {solver.model_times} + {solver.resolution_times} = {total_time}     {counter_updated}")

        if solver.status == "TIME_LIMIT":
            max_time_counter += 1

        if solver.status == "OPTIMAL":
            max_time_counter = 0

        pairs_per_quadrant += step
        i += 1

        if i % 10 == 0:
            print()
            save_img(results, i)


    with open("time_scalability_results", "w") as f:
        f.write(f"Grid size = {grid_side * grid_side}\n\n")
        for n, time in results.items():
            f.write(f"{n}:{time}\n")

    save_img(results)


run()