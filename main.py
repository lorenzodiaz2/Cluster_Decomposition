import time

import numpy as np
import pandas as pd

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from utils.environment import Environment

if __name__ == '__main__':

    # iteration, obj, time, other = [], [] ,[], []
    #
    # for i in range(10):
    #     iteration.append(i)
    #     obj.append(2)
    #     time.append(4)
    #     other.append(np.zeros((4, 4)))
    #
    #
    # df = pd.DataFrame({'iter': iteration, 'obj': obj, 'time': time, 'other': other})
    # df.to_csv('test.csv', index=False)
    #
    # df = pd.read_csv("test.csv")

    env = Environment(30, 180, 4, 40, 0, 10)
    start = time.time()
    complete_solver = Heuristic_Solver(env.G, env.od_pairs)
    complete_solver.solve()
    complete_solver.assign_solutions()
    print(f"tempo {time.time() - start}   valore = {sum(a.delay for a in env.agents)}\n")

    start = time.time()
    env.compute_clusters()
    print(f"{time.time() - start} secondi per fare i cluster\n")

    start = time.time()
    for cluster in env.clusters:
        hs = Heuristic_Solver(env.G, cluster.od_pairs)
        hs.solve()
        hs.assign_solutions()

    print(f"tempo {time.time() - start}   valore = {sum(a.delay for a in env.agents)}\n")


    critical_resources = Critical_Resources(env.G, env.od_pairs)

    if critical_resources.is_initially_feasible:
        print("Solution is already feasible.")
    else:
        critical_resources.unassign_agents()
        print(f"removed {len(critical_resources.removed_agents)} agents")

        start = time.time()
        final_solver = Heuristic_Solver(env.G, env.od_pairs, critical_resources)
        final_solver.solve()
        final_solver.assign_solutions()
        print(f"tempo {time.time() - start}   valore = {sum(a.delay for a in env.agents)}\n")


# todo cambiare il salvataggio e salvare su csv usando pandas
# todo fare qualche prova su istanze piccole
# todo iniziare a tirare fuori qualche numero