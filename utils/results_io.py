import os

from natsort import natsorted


def save_results(env, complete_solver, cluster_solvers, critical_resources, final_solver, i):
    hs_all_time = 0
    hs_cluster_time = 0

    with open("scalability_results", "a") as f:
        f.write(f"NUMBER OF QUADRANTS = {env.n_quadrants}     NUMBER OF TOTAL PAIRS = {env.n_pairs_per_quadrant * env.n_quadrants}     MAX CLUSTER SIZE = {env.max_cluster_size}     k = {env.k}\n")
        f.write(f"Iteration {i}\n")
        f.write(f"\nEnvironment created     Time: {env.set_time}\n")
        diff = complete_solver.current_T - complete_solver.starting_T
        for i in range(diff + 1):
            status = "INFEASIBLE" if i < diff else complete_solver.status
            f.write(f"\nT = {complete_solver.starting_T + i}  ->   Model created   Time = {complete_solver.model_times[i]}    status = {status}     Time = {complete_solver.resolution_times[i]}")
            hs_all_time += complete_solver.model_times[i] + complete_solver.resolution_times[i]
        f.write(f"\n\nDelay by solving all pairs     --->     objVal (UB) = {complete_solver.m.ObjVal}    objBound (LB) = {complete_solver.m.ObjBound}     Time = {hs_all_time}\n")
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
            f.write(f"Unassigned {critical_resources.removed_agents_per_tol[0]} Agents    Time = {critical_resources.unassigning_times[0]}")
            hs_cluster_time += critical_resources.unassigning_times[0]


            diff = critical_resources.current_tol - critical_resources.starting_tol
            for i in range(1, diff + 1):
                f.write(f"\nNo solution found     Time = {final_solver.resolution_times[i - 1]}   ->   Augmented tolerance\n")
                f.write(f"\n{critical_resources.critical_resources_per_tol[i]} critical resources    Time = {critical_resources.creation_times[i]}     ")
                f.write(f"Unassigned {critical_resources.removed_agents_per_tol[i]} Agents    Time = {critical_resources.unassigning_times[i]}")
                hs_cluster_time += critical_resources.creation_times[i] + critical_resources.unassigning_times[i]


            f.write(f"\n\nFinale solver: T = {final_solver.current_T}  ->   Model created   Time = {final_solver.model_times[critical_resources.current_tol]}   status = {final_solver.status}     Time = {final_solver.resolution_times[critical_resources.current_tol]}\n")
            for i in range(len(final_solver.model_times)):
                hs_cluster_time += final_solver.model_times[i] + final_solver.resolution_times[i]
            delay_assigned_agents = sum(a.delay for a in env.agents if a not in final_solver.critical_resources.removed_agents)
            f.write(f"\nFinal Delay     --->     objVal (UB) = {final_solver.m.ObjVal + delay_assigned_agents}    objBound (LB) = {final_solver.m.ObjBound + delay_assigned_agents}    Total time cluster heuristic = {hs_cluster_time}")

        f.write("\n\n========================================================================================\n")
        f.write("========================================================================================\n\n")


def read_files():
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    results_dir = os.path.join(project_dir, "results/grid")

    for dir_grid_size in natsorted(os.listdir(results_dir)):
        for dir_n_od in natsorted(os.listdir(os.path.join(results_dir, dir_grid_size))):
            files_name = sorted(os.listdir(os.path.join(results_dir, dir_grid_size, dir_n_od)), key=key_fun)
            for i in range(int(len(files_name) / 4)):
                with open(os.path.join(results_dir, dir_grid_size, dir_n_od, files_name[i * 4]), "r") as f:
                    lines = f.readlines()

                for j in range(len(lines)):
                    if "Delay by solving all pairs" in lines[j]:
                        status_complete_solver = lines[j - 2].split("status = ")[1].strip().split()[0]
                        obj_value_complete = float(lines[j].split("objVal (UB) = ")[1].strip().split()[0])
                        obj_bound_complete = float(lines[j].split("objBound (LB) = ")[1].strip().split()[0])
                        time_complete_solver = float(lines[j].split("Time = ")[1].strip().split()[0])
                        print(f"Heuristic to all pairs                 ->                                           status = {status_complete_solver}  objValue = {obj_value_complete}  objBound = {obj_bound_complete}  Time = {time_complete_solver}")

                for j in range(i * 4, i * 4 + 4):
                    n_clusters = 0
                    n_unassigned_agents = 0
                    status = ""
                    obj_val = 0.
                    obj_bound = 0.
                    total_time = 0
                    is_already_feasible = False
                    all_cluster_feasible = True

                    with open(os.path.join(results_dir, dir_grid_size, dir_n_od, files_name[j]), "r") as f:
                        lines = f.readlines()
                    for z in range(len(lines)):
                        if "Created" in lines[z] and "clusters" in lines[z]:
                            n_clusters = lines[z].split()[1]
                        if "Cluster 0:" in lines[z]:
                            for c in range(int(n_clusters)):
                                cluster_status = lines[z + c].split("status = ")[1].split()[0]
                                if cluster_status != "OPTIMAL":
                                    all_cluster_feasible = False
                                    break
                        if "Solution is already feasible" in lines[z]:
                            is_already_feasible = True
                            break
                        if "Finale solver" in lines[z]:
                            status = lines[z].split("status = ")[1].split()[0]
                            n_unassigned_agents = lines[z - 2].split("Unassigned ")[1].split()[0]
                        if "Final Delay" in lines[z]:
                            obj_val = lines[z].split("objVal (UB) = ")[1].split()[0]
                            obj_bound = lines[z].split("objBound (LB) = ")[1].split()[0]
                            total_time = lines[z].split("Total time cluster heuristic = ")[1].split()[0]

                    if not is_already_feasible:
                        size = files_name[j].split("_")[0]
                        cluster_status = "OK" if all_cluster_feasible else "NOT OK"
                        print(f"Heuristic to cluster (max size = {size})  ->  {n_clusters} clusters ({cluster_status})  {n_unassigned_agents} unassigned agents  status = {status}  objValue = {obj_val}  objBound = {obj_bound}  Time = {total_time}")





def key_fun(s):
    a1, a2, a3 = map(int, s.split("_"))
    return a2, a1, a3

