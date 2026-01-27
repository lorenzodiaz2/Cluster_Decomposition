import math
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from general.general_environment import General_Environment
from general.general_solver import General_Solver




def save_img(results, name_file):
    xs = sorted(results.keys())
    ys = [results[x] for x in xs]
    plt.figure()
    plt.plot(xs, ys, marker="o", markersize=3)
    plt.xlabel("od")
    plt.ylabel("Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{name_file}.png")
    plt.close()




def run_time_scalability(
    grid_side: int,
    k: int,
    env_maker: Callable[..., General_Environment],
    solver_maker: Callable[..., General_Solver],
    file_results_name: str,
    time_limit: int = 1800,
    n_seeds: int = 5,
    n_start: int = 50,
    step: int = 5,
    stop_after: int = 3,
    tl_threshold: float = 0.6
):
    n = n_start
    consecutive_bad = 0

    median_times = {}
    tl_rates = {}

    while True:
        times = []
        tl = 0

        needed_tl = math.ceil(tl_threshold * n_seeds)

        for j in range(n_seeds):
            seed = 12345 + 100000 * j + n

            env = env_maker(grid_side, k, n, seed)
            print(env)

            solver = solver_maker(env)
            solver.solve()

            total_time = sum(solver.model_times) + sum(solver.resolution_times)
            times.append(total_time)

            if solver.status == "TIME_LIMIT":
                tl += 1

            if tl >= needed_tl:
                break

        tl_rate = tl / n_seeds
        med = float(np.median(times))

        median_times[n] = med
        tl_rates[n] = tl_rate

        print(f"n={n}  median={med:.3f}s  TL_rate={tl_rate:.2f}")

        if tl_rate >= tl_threshold:
            consecutive_bad += 1
        else:
            consecutive_bad = 0

        if consecutive_bad >= stop_after:
            break

        n += step

    with open(f"{file_results_name}.txt", "w") as f:
        f.write(f"Grid side = {grid_side}  (cells={grid_side*grid_side})\n")
        f.write(f"time_limit={time_limit}s, n_seeds={n_seeds}\n\n")
        for n in sorted(median_times):
            f.write(f"{n}: median={median_times[n]:.6f}, tl_rate={tl_rates[n]:.2f}\n")

    save_img(median_times, file_results_name)