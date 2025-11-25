from scalability.resolution_scalability import run_scalability
from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from utils.environment import Environment
from utils.plot_functions import plot_paths
from utils.results_io import read_scalability_results, read_files


# run_scalability()
# read_scalability_results()
# read_files()

env = Environment(20, 50, 5, 20, 0, 10)
complete_solver = Heuristic_Solver(env.G, env.od_pairs)
complete_solver.solve()
complete_solver.assign_solutions()
print(f"complete delay = {sum(a.delay for a in env.agents)}\n")

env.compute_clusters()


for cluster in env.clusters:
    hs = Heuristic_Solver(env.G, [od_pair for od_pair in cluster.od_pairs])
    hs.solve()
    hs.assign_solutions()
    # plot_paths(env.G, cluster.od_pairs)

print(f"cluster delay = {sum(a.delay for a in env.agents)}\n")


critical_resources = Critical_Resources(env.G, env.od_pairs)
if critical_resources.is_initially_feasible:
    print("Solution is already feasible.")
else:
    critical_resources.unassign_agents()
    final_solver = Heuristic_Solver(env.G, env.od_pairs, critical_resources)
    final_solver.solve()
    final_solver.assign_solutions()
    print(f"final delay = {sum(a.delay for a in env.agents)}\n")



# todo usare un orizzonte per coppia, non più globale (trovare il più lungo e fare + 3)
# todo calcolare la similarità usando tutti i path (anche quelli ritardati)
# todo vedere pandas
# todo calcolare gap, varianza, tempo, ... sui risultati