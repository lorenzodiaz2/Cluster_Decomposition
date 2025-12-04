import matplotlib.pyplot as plt

from solvers.heuristic_solver import Heuristic_Solver
from elements.environment import Environment


def save_img(results, i: int | None = None):
    plt.plot(results.keys(), results.values(), marker="o", markersize=3)
    plt.xlabel("od")
    plt.ylabel("Time")
    plt.grid(True)
    plt.tight_layout()

    if i:
        plt.savefig(f"scalability_{i}.png")
    else:
        plt.savefig(f"scalability_new.png")


def run():
    grid_side = 20
    offset = 20
    k = 10
    max_count = 3
    step = 5

    for j in range(5):
        print("\n\n==============================================================================================\n\n")
        max_time_counter = 0
        pairs_per_quadrant = 5
        i = 0
        results = {}


        while max_time_counter < max_count:
            print(f"i = {i}     grid side = {grid_side}   total number of pairs = {pairs_per_quadrant}... ", end="")
            seed = 50 * j + i
            env = Environment(grid_side, 0, 4, pairs_per_quadrant, offset, k, seed=seed)
            solver = Heuristic_Solver(env.G, env.od_pairs)
            solver.solve()

            total_time = sum(m_times for m_times in solver.model_times) + sum(r_time for r_time in solver.resolution_times)
            results[pairs_per_quadrant] = total_time

            counter_updated = "COUNTER UPDATED" if solver.status == "TIME_LIMIT" else ""
            print(f"Solved in {solver.model_times} + {solver.resolution_times} = {total_time}     {counter_updated}")

            if solver.status == "TIME_LIMIT":
                max_time_counter += 1

            if solver.status == "OPTIMAL":
                max_time_counter = 0

            pairs_per_quadrant += step
            i += 1


        with open("time_scalability_results", "w") as f:
            f.write(f"Grid size = {grid_side * grid_side}\n\n")
            for n, time in results.items():
                f.write(f"{n}:{time}\n")

        save_img(results)
