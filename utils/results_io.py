import numpy as np

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from utils.environment import Environment


def save_results(
    env: Environment,
    complete_solver: Heuristic_Solver,
    cluster_solvers: list[Heuristic_Solver],
    i: int,
    critical_resources: Critical_Resources | None = None,
    final_solver: Heuristic_Solver | None = None,
):
    hs_all_time = 0
    hs_cluster_time = 0

    with open("scalability_results_3", "a") as f:
        f.write(f"NUMBER OF QUADRANTS = {env.n_quadrants}     NUMBER OF TOTAL PAIRS = {env.n_pairs_per_quadrant * env.n_quadrants}     NUMBER OF TOTAL AGENTS = {len(env.agents)}     MAX CLUSTER SIZE = {env.max_cluster_size}     k = {env.k}\n")
        f.write(f"Iteration {i}\n")
        f.write(f"\nEnvironment created     Time: {env.set_time}\n")
        diff = len(complete_solver.model_times)
        for j in range(diff):
            status = "INFEASIBLE" if j < diff - 1 else complete_solver.status
            f.write(f"\nModel created   Time = {complete_solver.model_times[j]}    status = {status}     Time = {complete_solver.resolution_times[j]}")
            hs_all_time += complete_solver.model_times[j] + complete_solver.resolution_times[j]
        f.write(f"\n\nDelay by solving all pairs     --->     objVal (UB) = {complete_solver.m.ObjVal}    objBound (LB) = {complete_solver.m.ObjBound}     Time = {hs_all_time}\n")
        f.write(f"\nCreated {len(env.clusters)} clusters.  Time = {env.matrix_time} + {env.cluster_time} = {env.matrix_time + env.cluster_time}\n")
        hs_cluster_time += env.matrix_time + env.cluster_time
        for cluster in env.clusters:
            f.write(f"     {cluster}\n")

        for j, hs in enumerate(cluster_solvers):
            f.write(f"\nCluster {j}:   Model created   Time = {hs.model_times[0]}    status = {hs.status}     Time = {hs.resolution_times[0]}     --->     objVal (UB) = {hs.m.ObjVal}    objBound (LB) = {hs.m.ObjBound}")
            hs_cluster_time += hs.model_times[0] + hs.resolution_times[0]

        if not critical_resources:
            f.write(f"\n\nClusters are not feasible")
            return

        f.write(f"\n\nDelay by solving clusters = {sum(hs.m.ObjVal for hs in cluster_solvers)}     Time (without cluster creation) = {sum(hs.model_times[0] + hs.resolution_times[0] for hs in cluster_solvers)}\n")



        f.write(f"\n{critical_resources.critical_resources_per_tol[0]} critical resources  Time = {critical_resources.creation_times[0]}     ")
        hs_cluster_time += critical_resources.creation_times[0]
        if final_solver is None:
            f.write(f"\n\nSolution is already feasible    Total time cluster heuristic = {hs_cluster_time}")
        else:
            f.write(f"Unassigned {critical_resources.removed_agents_per_tol[0]} Agents    Time = {critical_resources.unassigning_times[0]}")
            hs_cluster_time += critical_resources.unassigning_times[0]


            diff = critical_resources.current_tol - critical_resources.starting_tol
            for j in range(1, diff + 1):
                f.write(f"\nNo solution found     Time = {final_solver.resolution_times[j - 1]}   ->   Augmented tolerance\n")
                f.write(f"\n{critical_resources.critical_resources_per_tol[j]} critical resources    Time = {critical_resources.creation_times[j]}     ")
                f.write(f"Unassigned {critical_resources.removed_agents_per_tol[j]} Agents    Time = {critical_resources.unassigning_times[j]}")
                hs_cluster_time += critical_resources.creation_times[j] + critical_resources.unassigning_times[j]


            f.write(f"\n\nFinale solver:   Model created   Time = {final_solver.model_times[critical_resources.current_tol]}   status = {final_solver.status}     Time = {final_solver.resolution_times[critical_resources.current_tol]}\n")
            for j in range(len(final_solver.model_times)):
                hs_cluster_time += final_solver.model_times[j] + final_solver.resolution_times[j]
            delay_assigned_agents = sum(a.delay for a in env.agents if a not in final_solver.critical_resources.removed_agents)
            f.write(f"\nFinal Delay     --->     objVal (UB) = {final_solver.m.ObjVal + delay_assigned_agents}    objBound (LB) = {final_solver.m.ObjBound + delay_assigned_agents}    Total time cluster heuristic = {hs_cluster_time}")

        f.write("\n\n========================================================================================\n")
        f.write("========================================================================================\n\n")


def read_scalability_results():
    with open("../scalability_results_2", "r") as f:
        lines = f.readlines()

    n_clusters = 0
    n_unassigned_agents = 0
    status = ""
    iteration = ""
    is_already_feasible = False
    all_cluster_feasible = True

    obj_value_complete = None
    obj_bound_complete = None
    time_complete_solver = None
    final_obj_val = None
    final_obj_bound = None
    total_time = None

    abs_gaps = []
    rel_gaps = []
    time_abs_gaps = []
    time_rel_gaps = []


    for j in range(len(lines)):
        if "NUMBER OF QUADRANTS = " in lines[j]:
            n_quadrants = lines[j].split("NUMBER OF QUADRANTS = ")[1].split()[0]
            n_pairs = lines[j].split("NUMBER OF TOTAL PAIRS = ")[1].split()[0]
            n_agents = lines[j].split("NUMBER OF TOTAL AGENTS = ")[1].split()[0]
            cluster_size = lines[j].split("MAX CLUSTER SIZE = ")[1].split()[0]
            iteration = lines[j + 1].split()[1]
            if iteration == "0":
                print(f"NUMBER OF QUADRANTS = {n_quadrants}     NUMBER OF TOTAL PAIRS = {n_pairs}     NUMBER OF TOTAL AGENTS = {n_agents}     MAX CLUSTER SIZE = {cluster_size}")
            print(f"\n\n    ITERATION {iteration}")


        if "Delay by solving all pairs" in lines[j]:
            status_complete_solver = lines[j - 2].split("status = ")[1].strip().split()[0]
            obj_value_complete = float(lines[j].split("objVal (UB) = ")[1].strip().split()[0])
            obj_bound_complete = float(lines[j].split("objBound (LB) = ")[1].strip().split()[0])
            time_complete_solver = float(lines[j].split("Time = ")[1].strip().split()[0])
            print(f"        Heuristic to all pairs                           ->                                 status = {status_complete_solver}  objValue = {obj_value_complete}  objBound = {obj_bound_complete}  Time = {time_complete_solver}")

        if "Created" in lines[j] and "clusters" in lines[j]:
            n_clusters = lines[j].split()[1]

        if "Cluster 0:" in lines[j]:
            is_already_feasible = False
            all_cluster_feasible = True
            for c in range(int(n_clusters)):
                cluster_status = lines[j + c].split("status = ")[1].split()[0]
                if cluster_status != "OPTIMAL":
                    all_cluster_feasible = False
                    break

        if "Solution is already feasible" in lines[j]:
            is_already_feasible = True

        if "Finale solver" in lines[j]:
            status = lines[j].split("status = ")[1].split()[0]
            n_unassigned_agents = lines[j - 2].split("Unassigned ")[1].split()[0]
        if "Final Delay" in lines[j]:
            final_obj_val = float(lines[j].split("objVal (UB) = ")[1].split()[0])
            final_obj_bound = float(lines[j].split("objBound (LB) = ")[1].split()[0])
            total_time = float(lines[j].split("Total time cluster heuristic = ")[1].split()[0])

            if not is_already_feasible:
                cluster_status = "OK" if all_cluster_feasible else "NOT OK"
                abs_gap = final_obj_val - obj_value_complete
                rel_gap = round(100 * (final_obj_val - obj_value_complete) / obj_value_complete, 2)

                time_abs_gap = round(total_time - time_complete_solver, 2)
                time_rel_gap = round(100 * (total_time - time_complete_solver) / time_complete_solver, 2)

                abs_gaps.append(abs_gap)
                rel_gaps.append(rel_gap)
                time_abs_gaps.append(time_abs_gap)
                time_rel_gaps.append(time_rel_gap)

                print(f"        Heuristic to cluster ({n_clusters} clusters ({cluster_status}))          ->         {n_unassigned_agents} unassigned agents  status = {status}  objValue = {final_obj_val}  objBound = {final_obj_bound}  Time = {total_time}   ")
                print(f"        abs gap = {abs_gap}     rel_gap = {rel_gap} %      ")
                print(f"        abs time gap = {time_abs_gap}     rel time gap = {time_rel_gap} %      ", end="")
            else:
                print(f"        Cluster solution is already feasible")



            if iteration == "4":
                print(f"\n\n    mean abs value gap = {round(np.array(abs_gaps).mean(), 2)}    ->    std dev abs value gap = {round(np.array(abs_gaps).std(ddof=1), 2)}")
                print(f"    mean rel value gap = {round(np.array(rel_gaps).mean(), 2)} %    ->    std dev rel value gap = {round(np.array(rel_gaps).std(ddof=1), 2)} %")

                print(f"    mean abs time gap = {round(np.array(time_abs_gaps).mean(), 2)}    ->    std dev abs time gap = {round(np.array(time_abs_gaps).std(ddof=1), 2)}")
                print(f"    mean rel time gap = {round(np.array(time_rel_gaps).mean(), 2)} %    ->    std dev rel time gap = {round(np.array(time_rel_gaps).std(ddof=1), 2)} %")

                abs_gaps = []
                rel_gaps = []
                time_abs_gaps = []
                time_rel_gaps = []

                print("\n================================================================================================================================================================================\n\n")


read_scalability_results()