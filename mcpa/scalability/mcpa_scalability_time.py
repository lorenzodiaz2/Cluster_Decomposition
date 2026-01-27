from mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from mcpa.elements.mcpa_environment import MCPA_Environment
from utils.scalability_utils import run_time_scalability

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
run_time_scalability(20, 10, env_maker, solver_maker, "mcpa_scalability_results")




