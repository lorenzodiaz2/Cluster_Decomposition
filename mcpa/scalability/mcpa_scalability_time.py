from mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from mcpa.elements.mcpa_environment import MCPA_Environment
from utils.scalability_utils import run_time_scalability


def run_mcpa_tima_scalability():
    env_maker = lambda grid_side, k, n, seed: MCPA_Environment(
        grid_side=grid_side,
        max_cluster_size=0,
        n_quadrants=1,
        n_pairs_per_quadrant=n,
        offset=0,
        k=k,
        seed=seed
    )
    solver_maker = lambda env: MCPA_Heuristic_Solver(env.G, env.elements)
    run_time_scalability(
        20, 10, env_maker, solver_maker, "mcpa_scalability_results",
        n_start=50,
        time_limit=1800,
        n_seeds=5,
        tl_bad_count=3,
        consecutive_bad_stop=5,
        x_label="Number of OD Pairs"
    )




