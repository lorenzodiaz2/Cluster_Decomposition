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

