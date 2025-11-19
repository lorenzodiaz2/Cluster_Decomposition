import os
import time

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from utils.environment import Environment
from utils.results_io import save_results


def run_scalability():
    update()
    exit(0)
    # GRID_SIDE_values = [10, 20, 30, 50, 70, 90] #, 100, 120, 150, 180, 200, 250]
    GRID_SIDE_values = [50, 70, 90, 100, 120, 150, 180, 200, 250]


    k_values = {
        10: 12,
        20: 15,
        30: 15,
        50: 20,
        70: 25,
        90: 25,
        100: 25,
        120: 25,
        150: 30,
        180: 30,
        200: 30,
        250: 35
    }

    NUM_PAIRS_PER_QUADRANT_values = {
        10: [3, 5, 8, 10, 15],
        20: [8, 12, 15, 20, 30],
        30: [10, 15, 25, 30, 45],
        # 50: [20, 30, 40, 50, 65, 75],
        50: [75],
        70: [30, 45, 60, 70, 80, 90, 100],
        90: [40, 60, 80, 90, 100, 120],
        100: [50, 70, 90, 100, 120, 130],
        120: [60, 75, 90, 120, 140, 150],
        150: [70, 90, 125, 150, 160, 175],
        180: [90, 120, 150, 180, 190, 200],
        200: [100, 140, 170, 200, 210, 225],
        250: [120, 160, 200, 250, 260, 275]
    }

    OFFSET_values = {
        10: [-2, 0, 2, 5],
        20: [-5, -2, 0, 2, 5, 10],
        30: [-7, -3, 0, 3, 7, 15],
        # 50: [-12, -5, 0, 5, 12, 25],
        50: [0, 5, 12, 25],
        70: [-20, -12, -5, 0, 5, 12, 20, 35],
        90: [-25, -18, -12, -5, 0, 5, 12, 18, 25, 45],
        100: [-30, -24, -16, -8, 0, 8, 16, 24, 30, 50],
        120: [-35, -25, -15, -10, -5, 0, 5, 10, 15, 25, 35, 60],
        150: [-45, -30, -15, -6, 0, 6, 15, 30, 45, 75],
        180: [-60, -45, -25, -10, 0, 10, 25, 45, 60, 90],
        200: [-70, -50, -35, -20, -10, 0, 10, 20, 35, 50, 70, 100],
        250: [-80, -65, -30, -10, 0, 10, 30, 65, 80, 125]
    }

    MAX_CLUSTER_SIZE_values = {(g, n): [int(n * 4 * 12 / d) for d in (8, 6, 4, 2)] for g in GRID_SIDE_values for n in NUM_PAIRS_PER_QUADRANT_values[g]}


    for g in GRID_SIDE_values:
        for n in NUM_PAIRS_PER_QUADRANT_values[g]:
            if not os.path.isdir(f"results/grid/{g * g}/{n * 4}"):
                os.makedirs(f"results/grid/{g * g}/{n * 4}")
            for offset in OFFSET_values[g]:
                print()
                env, complete_solver = solve_complete(g, n, offset, k_values[g])
                for s in MAX_CLUSTER_SIZE_values[(g, n)]:
                    print(f"results/grid/{g * g}/{n * 4}/{int(s)}_{offset}_{k_values[g]}")
                    env.max_cluster_size = s
                    solve_clusters(env, complete_solver)




def solve_complete(grid_side, num_pairs_per_quadrant, offset, k):
    print(f"COMPLETA results/grid/{grid_side * grid_side}/{num_pairs_per_quadrant * 4}/offset={offset}...", end=" ")
    start = time.time()
    env = Environment(grid_side, 0, num_pairs_per_quadrant, offset, k)
    complete_solver = Heuristic_Solver(env.G, env.od_pairs, env.T)
    complete_solver.solve()
    print(f"RISOLTA in {time.time() - start}")
    return env, complete_solver


def solve_clusters(env, complete_solver):
    env.compute_clusters()
    cluster_solvers = []
    for cluster in env.clusters:
        hs = Heuristic_Solver(env.G, [od_pair for od_pair in cluster.od_pairs], complete_solver.current_T)
        hs.solve()
        hs.assign_solutions()
        cluster_solvers.append(hs)

    critical_resources = Critical_Resources(env.G, env.od_pairs)
    if critical_resources.is_initially_feasible:
        save_results(env, complete_solver, cluster_solvers, critical_resources, None)
    else:
        critical_resources.unassign_agents()
        final_solver = Heuristic_Solver(env.G, env.od_pairs, complete_solver.current_T, critical_resources)
        final_solver.solve()
        final_solver.assign_solutions()
        save_results(env, complete_solver, cluster_solvers, critical_resources, final_solver)


def update():
    GRID_SIDE_values = [10, 20, 30, 50]

    k_values = {
        10: 12,
        20: 15,
        30: 15,
        50: 20,
        70: 25,
        90: 25,
        100: 25,
        120: 25,
        150: 30,
        180: 30,
        200: 30,
        250: 35
    }

    NUM_PAIRS_PER_QUADRANT_values = {
        10: [3, 5, 8, 10, 15],
        20: [8, 12, 15, 20, 30],
        30: [10, 15, 25, 30, 45],
        50: [20, 30, 40, 50, 65, 75]
    }

    OFFSET_values = {
        10: [-2, 0, 2, 5],
        20: [-5, -2, 0, 2, 5, 10],
        30: [-7, -3, 0, 3, 7, 15],
        50: [-12, -5, 0, 5, 12, 25],
    }

    MAX_CLUSTER_SIZE_values =  {(g, n): int(n * 4 * 12 / 2) for g in GRID_SIDE_values for n in NUM_PAIRS_PER_QUADRANT_values[g]}


    for g in GRID_SIDE_values:
        for n in NUM_PAIRS_PER_QUADRANT_values[g]:
            if g == 50 and n == 75:
                print("\n\n")
                OFFSET_values[g] = [-12, -5]
            for offset in OFFSET_values[g]:
                s = MAX_CLUSTER_SIZE_values[(g, n)]
                file_to_create = f"results/grid/{g * g}/{n * 4}/{int(s)}_{offset}_{k_values[g]}"
                file_to_read = f"results/grid/{g * g}/{n * 4}/{int(s / 2)}_{offset}_{k_values[g]}"

                print(f"{file_to_read}     ->     {file_to_create}")
                with open(file_to_read, "r") as f:
                    lines = f.readlines()
                count = 0
                T = 0
                for i in range(len(lines)):
                    if "Delay by solving all pairs" in lines[i]:
                        count = i
                        T = lines[i - 2].split("T = ")[1].split()[0]
                        break

                with open(file_to_create, "w") as f:
                    for i in range(count + 1):
                        f.write(lines[i])

                    env = Environment(g, s, n, offset, k_values[g])

                    env.compute_clusters()
                    cluster_solvers = []
                    for cluster in env.clusters:
                        hs = Heuristic_Solver(env.G, [od_pair for od_pair in cluster.od_pairs], int(T))
                        hs.solve()
                        hs.assign_solutions()
                        cluster_solvers.append(hs)

                    critical_resources = Critical_Resources(env.G, env.od_pairs)
                    if critical_resources.is_initially_feasible:
                        save(env, cluster_solvers, critical_resources, None, f)
                    else:
                        critical_resources.unassign_agents()
                        final_solver = Heuristic_Solver(env.G, env.od_pairs, int(T), critical_resources)
                        final_solver.solve()
                        final_solver.assign_solutions()
                        save(env, cluster_solvers, critical_resources, final_solver, f)


def save(env, cluster_solvers, critical_resources, final_solver, f):
    hs_cluster_time = 0

    f.write(f"\nCreated {len(env.clusters)} clusters.  Time = {env.cluster_time}\n")
    hs_cluster_time += env.cluster_time
    for cluster in env.clusters:
        f.write(f"     {cluster}\n")

    for i, hs in enumerate(cluster_solvers):
        f.write(f"\nCluster {i}: T = {hs.starting_T}  ->   Model created   Time = {hs.model_times[0]}    status = {hs.status}     Time = {hs.resolution_times[0]}     --->     objVal (LB) = {hs.m.ObjVal}    objBound (UB) = {hs.m.ObjBound}")
        hs_cluster_time += hs.model_times[0] + hs.resolution_times[0]
    f.write(f"\n\nDelay by solving clusters = {sum(hs.m.ObjVal for hs in cluster_solvers)}     Time (without cluster creation) = {sum(hs.model_times[0] + hs.resolution_times[0] for hs in cluster_solvers)}\n")

    f.write(f"\n{critical_resources.critical_resources_per_tol[0]} critical resources  Time = {critical_resources.creation_times[0]}     ")
    hs_cluster_time += critical_resources.creation_times[0]
    if final_solver is None:
        f.write(f"\n\nSolution is already feasible    Total time cluster heuristic = {hs_cluster_time}")
    else:
        f.write(
            f"Unassigned {critical_resources.removed_agents_per_tol[0]} Agents    Time = {critical_resources.unassigning_times[0]}")
        hs_cluster_time += critical_resources.unassigning_times[0]

        diff = critical_resources.current_tol - critical_resources.starting_tol
        for i in range(1, diff + 1):
            f.write(
                f"\nNo solution found     Time = {final_solver.resolution_times[i - 1]}   ->   Augmented tolerance\n")
            f.write(
                f"\n{critical_resources.critical_resources_per_tol[i]} critical resources    Time = {critical_resources.creation_times[i]}     ")
            f.write(
                f"Unassigned {critical_resources.removed_agents_per_tol[i]} Agents    Time = {critical_resources.unassigning_times[i]}")
            hs_cluster_time += critical_resources.creation_times[i] + critical_resources.unassigning_times[i]

        f.write(
            f"\n\nFinale solver: T = {final_solver.current_T}  ->   Model created   Time = {final_solver.model_times[critical_resources.current_tol]}   status = {final_solver.status}     Time = {final_solver.resolution_times[critical_resources.current_tol]}\n")
        for i in range(len(final_solver.model_times)):
            hs_cluster_time += final_solver.model_times[i] + final_solver.resolution_times[i]
        delay_assigned_agents = sum(
            a.delay for a in env.agents if a not in final_solver.critical_resources.removed_agents)
        f.write(f"\nFinal Delay     --->     objVal (UB) = {final_solver.m.ObjVal + delay_assigned_agents}    objBound (LB) = {final_solver.m.ObjBound + delay_assigned_agents}    Total time cluster heuristic = {hs_cluster_time}")





