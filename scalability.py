from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from utils.environment import Environment


def run_scalability():
    GRID_SIZE_values = [10, 30, 50, 70, 90, 100, 120, 150, 180, 200, 250]

    k = 15

    NUM_PAIRS_PER_QUADRANT_values = {
        10: [3, 5, 8, 10],
        30: [10, 15, 25, 30],
        50: [20, 30, 40, 50],
        70: [30, 45, 60, 70],
        90: [40, 60, 80, 90],
        100: [50, 70, 90, 100],
        120: [60, 75, 90, 120],
        150: [70, 90, 125, 150],
        180: [90, 120, 150, 180],
        200: [100, 140, 170, 200],
        250: [120, 160, 200, 250]
    }

    OFFSET_values = {
        10: [-2, 0, 2, 5],
        30: [-7, -3, 0, 3, 7, 15],
        50: [-12, -5, 0, 5, 12, 25],
        70: [-20, -12, -5, 0, 5, 12, 20, 35],
        90: [-25, -18, -12, -5, 0, 5, 12, 18, 25, 45],
        100: [-30, -24, -16, -8, 0, 8, 16, 24, 30, 50],
        120: [-35, -25, -15, -10, -5, 0, 5, 10, 15, 25, 35, 60],
        150: [-45, -30, -15, -6, 0, 6, 15, 30, 45, 75],
        180: [-60, -45, -25, -10, 0, 10, 25, 45, 60, 90],
        200: [-70, -50, -35, -20, -10, 0, 10, 20, 35, 50, 70, 100],
        250: [-80, -65, -30, 10, 0, 10, 30, 65, 80, 125]
    }

    MAX_CLUSTER_SIZE_values = {(g, n): [n * 4 * 12 / 8, n * 4 * 12 / 6, n * 4 * 12 / 4, n * 4 * 12 / 2] for g in GRID_SIZE_values for n in NUM_PAIRS_PER_QUADRANT_values[g]}


    # for g in GRID_SIZE_values:
    #     for n in NUM_PAIRS_PER_QUADRANT_values[g]:
    #         for s in MAX_CLUSTER_SIZE_values[(g, n)]:
    #             for offset in OFFSET_values[g]:
    #                 print(f"SIZE = {g},   pairs per quadrant = {n},   max cluster size = {s},   offset = {offset},   k = {k}     ", end="")
    #                 execute(g, n, s, offset, k)
    #                 print("\n===============================================================================================================================================================================================")
    #                 print("===============================================================================================================================================================================================\n")

    execute(20, 21, 250, 0, 15)


def execute(GRID_SIZE, NUM_PAIRS_PER_QUADRANT, MAX_CLUSTER_SIZE, OFFSET, k):
    env = Environment(GRID_SIZE, MAX_CLUSTER_SIZE, NUM_PAIRS_PER_QUADRANT, OFFSET, k)

    complete_solver = Heuristic_Solver(env.G, env.od_pairs, env.T)
    complete_solver.solve()

    offset = complete_solver.T - complete_solver.starting_T
    for i in range(offset + 1):
        print(f"T = {complete_solver.starting_T + i}  ->   Model created   Time = {complete_solver.model_times[i]}    status = {"INFEASIBLE" if i < offset else complete_solver.status}     Time = {complete_solver.resolution_times[i]}")

    print(f"\nResolution delay by solving all pairs = {sum(a.delay for a in complete_solver.A)}    objVal (LB) = {complete_solver.m.ObjVal}    objBound (UB) = {complete_solver.m.ObjBound}")


    env.compute_clusters()
    cluster_solvers = []
    for cluster in env.clusters:
        hs = Heuristic_Solver(env.G, [od_pair for od_pair in cluster.od_pairs], complete_solver.T)
        hs.solve()
        print(f"Cluster {cluster.id}: T = {hs.starting_T}  ->   Model created   Time = {hs.model_times[0]}    status = {hs.status}     Time = {hs.resolution_times[0]}")
        cluster_solvers.append(hs)

    print(f"Resolution delay by solving clusters = {sum(a.delay for a in env.agents)}     Time = {sum(hs.model_times[0] + hs.resolution_times[0] for hs in cluster_solvers)}\n")



    critical_resources = Critical_Resources(env.G, env.od_pairs, 0)
    print(f"set Critical Resources.  Time {critical_resources.creation_time}")
    if critical_resources.is_initially_feasible:
        print("\nSolution is already feasible")
        return
    critical_resources.unassign_agents()
    print(f"Unassigned {len(critical_resources.removed_agents)} Agents.  Time {critical_resources.unassigning_time}\n")



    final_solver = Heuristic_Solver(env.G, env.od_pairs, complete_solver.T, critical_resources)
    final_solver.solve()

    print(f"Finale solver: T = {final_solver.T}  ->   Model created   Time = {final_solver.model_times[0]}   status = {final_solver.status}     Time = {final_solver.resolution_times[0]}")
    print(f"\nFinal resolution delay = {sum(a.delay for a in env.agents)}   objVal (LB) = {final_solver.m.ObjVal}    objBound (UB) = {final_solver.m.ObjBound}\n")


