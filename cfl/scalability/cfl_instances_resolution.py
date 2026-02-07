import os
from natsort import natsorted

from cfl.elements.cfl_environment import CFL_Environment
from cfl.solver.cfl_heuristic_solver import total_solution_cost
from cfl.solver.single_source.sscfl_critical_resources import SSCFL_Critical_Resources
from cfl.solver.single_source.sscfl_heuristic_solver import SSCFL_Heuristic_solver



def solve_cfl_instances(root):
    for folder_name in natsorted(os.listdir(root)):
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        for file_name in natsorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)

            is_TB_instance = True if file_path.__contains__("TB") else False
            if os.path.isfile(file_path):
                env = CFL_Environment(instance_file=file_path)

                complete_solver = SSCFL_Heuristic_solver(env.G, env.elements, env.facilities)
                complete_solver.solve()

                print(f"file {file_path}    complete = {total_solution_cost(env.elements, env.facilities, is_TB_instance)}")

                n_clients = len(env.elements)
                n_facilities = len(env.facilities)

                k_values = [n_facilities // i for i in [8, 4, 3, 2]]
                max_size_values = [n_clients // i for i in [10, 5, 3, 2]]

                for k in k_values:
                    for s in max_size_values:
                        env = CFL_Environment(max_cluster_size=s, k=k, instance_file=file_path)
                        env.solve_clusters(lambda c: SSCFL_Heuristic_solver(env.G, c.elements, c.facilities))
                        print(f"    k = {k}, max size = {s}, n clusters = {len(env.clusters)}, clusters value = {total_solution_cost(env.elements, env.facilities, is_TB_instance)}", end="   ")

                        critical_resources = SSCFL_Critical_Resources(env.elements, env.facilities)

                        if critical_resources.is_initially_feasible:
                            print("Feasible da subito")
                        else:
                            critical_resources.unassign_items()
                            post_solver = SSCFL_Heuristic_solver(env.G, env.elements, env.facilities, critical_resources)
                            post_solver.solve()
                            print(f"final value = {total_solution_cost(env.elements, env.facilities, is_TB_instance)}   status = {post_solver.status}")

