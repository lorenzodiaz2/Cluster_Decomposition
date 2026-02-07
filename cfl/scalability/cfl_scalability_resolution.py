from datetime import datetime
from typing import Callable

from pandas import DataFrame

from cfl.elements.cfl_environment import CFL_Environment
from cfl.solver.cfl_critical_resources import CFL_Critical_Resources
from cfl.solver.cfl_heuristic_solver import CFL_Heuristic_Solver
from cfl.solver.multi_source.mscfl_critical_resources import MSCFL_Critical_Resources
from cfl.solver.multi_source.mscfl_heuristic_solver import MSCFL_Heuristic_solver
from cfl.solver.single_source.sscfl_critical_resources import SSCFL_Critical_Resources
from cfl.solver.single_source.sscfl_heuristic_solver import SSCFL_Heuristic_solver
from utils.cfl_results_io import save_cfl_results


def run_scalability (
    base_grid_side: int,
    n: int,
    max_cluster_size: int,
    offset: int,
    k,
    df: DataFrame,
    starting_seed: int,
    q_range: list[int],
    n_iterations: int,
    env_maker: Callable[..., CFL_Environment],
    complete_solver_maker: Callable[..., CFL_Heuristic_Solver],
    cluster_solver_maker: Callable[..., CFL_Heuristic_Solver],
    critical_resource_maker: Callable[..., CFL_Critical_Resources],
    is_ss: bool,
    file_name: str
):
    if is_ss:
        output_file = f"results/cfl/ss/{file_name}.csv"
    else:
        output_file = f"results/cfl/ms/{file_name}.csv"

    seed = starting_seed

    for n_quadrants in q_range:
        grid_side = 2 * base_grid_side if n_quadrants <= 4 else 3 * base_grid_side
        for i in range(n_iterations):
            print(datetime.now().strftime("%d-%m-%Y   %H:%M:%S    "), end="")

            print(f"n quadrants = {n_quadrants}    n clients per q = {n * 5}    n facilities per q = {n}    offset = {offset}    iteration {i}   ", end="")

            env = env_maker(grid_side, max_cluster_size, n_quadrants, n, offset, k, seed)

            complete_solver = complete_solver_maker(env, None)
            complete_solver.solve()

            env.solve_clusters(cluster_solver_maker)

            critical_resources = None
            final_solver = None
            if all(hs.status == "OPTIMAL" for hs in env.clusters_solvers):
                critical_resources = critical_resource_maker(env)
                if not critical_resources.is_initially_feasible:
                    final_solver = complete_solver_maker(env, critical_resources)
                    final_solver.solve()

            save_cfl_results(env, critical_resources, final_solver, complete_solver, seed, df, is_ss)
            seed += 1
            df.to_csv(output_file, index=False)
        print()




def run_sscfl_scalability_instance(grid_side, n, max_cluster_size, offset, k, df, seed, q_range, n_iterations):
    env_maker = lambda _grid_side, _max_cluster_size, _n_quadrants, _n, _offset, _k, _seed: CFL_Environment(
        grid_side=_grid_side,
        max_cluster_size=_max_cluster_size,
        n_quadrants=_n_quadrants,
        n_clients_per_quadrant=_n * 5,
        n_facilities_per_quadrant=_n,
        offset=_offset,
        k=_k,
        seed=_seed
    )
    complete_solver_maker = lambda env, cr: SSCFL_Heuristic_solver(env.G, env.elements, env.facilities, cr)
    cluster_solver_maker = lambda env, cluster: SSCFL_Heuristic_solver(env.G, cluster.elements, cluster.facilities)
    critical_resources_maker = lambda env: SSCFL_Critical_Resources(env.elements, env.facilities)

    run_scalability(grid_side, n, max_cluster_size, offset, k, df, seed, q_range, n_iterations, env_maker, complete_solver_maker, cluster_solver_maker, critical_resources_maker, True, "sscfl_results")


def run_mscfl_scalability_instance(grid_side, n, max_cluster_size, offset, k, df, seed, q_range, n_iterations):
    env_maker = lambda _grid_side, _max_cluster_size, _n_quadrants, _n, _offset, _k, _seed: CFL_Environment(
        grid_side=_grid_side,
        max_cluster_size=_max_cluster_size,
        n_quadrants=_n_quadrants,
        n_clients_per_quadrant=_n * 5,
        n_facilities_per_quadrant=_n,
        offset=_offset,
        k=_k,
        seed=_seed
    )
    complete_solver_maker = lambda env, cr: MSCFL_Heuristic_solver(env.G, env.elements, env.facilities, cr)
    cluster_solver_maker = lambda env, cluster: MSCFL_Heuristic_solver(env.G, cluster.elements, cluster.facilities)
    critical_resources_maker = lambda env: MSCFL_Critical_Resources(env.elements, env.facilities)

    run_scalability(grid_side, n, max_cluster_size, offset, k, df, seed, q_range, n_iterations, env_maker, complete_solver_maker, cluster_solver_maker, critical_resources_maker, False, "mscfl_results")



















