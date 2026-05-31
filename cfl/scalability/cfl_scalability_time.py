import os
from datetime import datetime

import pandas as pd

from cfl.elements.cfl_environment import CFL_Environment
from cfl.solver.multi_source.mscfl_heuristic_solver import MSCFL_Heuristic_solver
from cfl.solver.single_source.sscfl_heuristic_solver import SSCFL_Heuristic_solver
from utils.scalability_utils import run_time_scalability


def run_cfl_time_scalability():
    env_maker = lambda grid_side, k, n, seed: CFL_Environment(
        grid_side=grid_side,
        max_cluster_size=0,
        n_quadrants=1,
        n_clients_per_quadrant=n * 5,
        n_facilities_per_quadrant=n,
        offset=0,
        k=k,
        seed=seed
    )

    sscfl_solver_maker = lambda env: SSCFL_Heuristic_solver(env.G, env.elements, env.facilities)
    mscfl_solver_maker = lambda env: MSCFL_Heuristic_solver(env.G, env.elements, env.facilities)

    run_time_scalability(
        50, 10, env_maker, sscfl_solver_maker, "sscfl_scalability_results",
        n_start=10,
        time_limit=1800,
        n_seeds=5,
        tl_bad_count=3,
        consecutive_bad_stop=5,
        # opzionale ma consigliato:
        seed_mode="by_n",  # meno correlazioni tra taglie
        x_label="Number of Clients",
        logy_plot=True
    )

    print("\n\n============================================================\n\n")

    run_time_scalability(
        50, 10, env_maker, mscfl_solver_maker, "mscfl_scalability_results",
         n_start=10,
         time_limit=1800,
         n_seeds=5,
         tl_bad_count=3,
         consecutive_bad_stop=5,
         # opzionale ma consigliato:
         seed_mode="by_n",  # meno correlazioni tra taglie
         x_label="Number of Clients",
         logy_plot=True
    )



def run_sscfl_time_scalability():
    seed = 0
    n = 160
    n_runs = 10
    csv_path = "results/cfl/ss/sscfl_global_solver.csv"

    n_times_out = 0

    while True:
        rows = []
        n_consecutive_time_limit = 0
        for _ in range(n_runs):
            env = CFL_Environment(50, 0, 1, n * 5, n, 0, 10, seed=seed)
            print(datetime.now().strftime(f"%d-%m-%Y   %H:%M:%S    {env}   seed={seed}"))
            global_solver = SSCFL_Heuristic_solver(env.G, env.elements, env.facilities)
            global_solver.solve()
            time = sum(global_solver.model_times) + sum(global_solver.resolution_times)
            rows.append({
                "n": n * 5,
                "seed": seed,
                "time": time
            })
            seed += 1
            if time >= global_solver.time_limit:
                n_consecutive_time_limit += 1

            if n_consecutive_time_limit >= 0.8 * n_runs:
                break

        if n_consecutive_time_limit >= 0.8 * n_runs:
            n_times_out += 1
        else:
            n_times_out = 0
        n += 5

        df = pd.DataFrame(rows)
        file_exists = os.path.isfile(csv_path)
        df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

        if n_times_out >= 3:
            break


def run_mscfl_time_scalability():
    seed = 0
    n = 10
    n_runs = 10
    csv_path = "results/cfl/ms/mscfl_global_solver.csv"

    n_times_out = 0

    while True:
        rows = []
        n_consecutive_time_limit = 0
        for _ in range(n_runs):
            env = CFL_Environment(50, 0, 1, n_clients_per_quadrant=n * 5, n_facilities_per_quadrant=n, offset=0, k=10, seed=seed)
            print(datetime.now().strftime(f"%d-%m-%Y   %H:%M:%S    {env}   seed={seed}"))
            global_solver = MSCFL_Heuristic_solver(env.G, env.elements, env.facilities)
            global_solver.solve()
            time = sum(global_solver.model_times) + sum(global_solver.resolution_times)
            rows.append({
                "n": n * 5,
                "seed": seed,
                "time": time
            })
            seed += 1
            if time >= global_solver.time_limit:
                n_consecutive_time_limit += 1

            if n_consecutive_time_limit >= 0.8 * n_runs:
                break

        if n_consecutive_time_limit >= 0.8 * n_runs:
            n_times_out += 1
        else:
            n_times_out = 0
        n += 5

        df = pd.DataFrame(rows)
        file_exists = os.path.isfile(csv_path)
        df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

        if n_times_out >= 3:
            break
