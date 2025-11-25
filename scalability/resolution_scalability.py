from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from utils.environment import Environment
from utils.results_io import save_results


def run_scalability():
    quadrant_range = range(9, 17)
    for n_quadrants in quadrant_range:
        print()
        for i in range(5):
            print(f"n quadrant = {n_quadrants}    iteration {i} ->  Setting env... ", end="")
            env = Environment(60, 750, n_quadrants, 150, 0, 12, False)
            print("Done.   Solving complete... ", end="")
            complete_solver = Heuristic_Solver(env.G, env.od_pairs)
            complete_solver.solve()
            print("Done.   Solving clusters... ", end="")

            env.compute_clusters()
            cluster_solvers = []
            for cluster in env.clusters:
                hs = Heuristic_Solver(env.G, [od_pair for od_pair in cluster.od_pairs])
                hs.solve()
                hs.assign_solutions()
                cluster_solvers.append(hs)
            print("Done.   ", end="")

            critical_resources = Critical_Resources(env.G, env.od_pairs)
            if critical_resources.is_initially_feasible:
                print("Solution is already feasible.")
                save_results(env, complete_solver, cluster_solvers, critical_resources, i, None)
                continue
            else:
                print("Unassigning agents... ", end="")
                critical_resources.unassign_agents()
                print("Done.   Solving final... ", end="")
                final_solver = Heuristic_Solver(env.G, env.od_pairs, critical_resources)
                final_solver.solve()
                final_solver.assign_solutions()
                print("Done.")
                save_results(env, complete_solver, cluster_solvers, critical_resources, i, final_solver)





