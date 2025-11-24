from scalability.resolution_scalability import run_scalability
from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from utils.environment import Environment
from utils.plot_functions import plot_paths
from utils.results_io import read_scalability_results, read_files


# run_scalability()
# read_scalability_results()
# read_files()

# env = Environment(40, 80, 16, 10, 0, 10)
env = Environment(60, 750, 3, 150, 0, 12)
complete_solver = Heuristic_Solver(env.G, env.od_pairs, env.T)
complete_solver.solve()
complete_solver.assign_solutions()
print(f"complete delay = {sum(a.delay for a in env.agents)}\n")

env.compute_clusters()

print(f"{len(env.clusters)} clusters.\n")

for cluster in env.clusters:
    hs = Heuristic_Solver(env.G, [od_pair for od_pair in cluster.od_pairs], complete_solver.current_T)
    hs.solve()
    hs.assign_solutions()
    plot_paths(env.G, cluster.od_pairs)

print(f"cluster delay = {sum(a.delay for a in env.agents)}\n")


critical_resources = Critical_Resources(env.G, env.od_pairs)
if critical_resources.is_initially_feasible:
    print("Solution is already feasible.")
else:
    critical_resources.unassign_agents()
    final_solver = Heuristic_Solver(env.G, env.od_pairs, complete_solver.current_T, critical_resources)
    final_solver.solve()
    final_solver.assign_solutions()
    print(f"final delay = {sum(a.delay for a in env.agents)}\n")
