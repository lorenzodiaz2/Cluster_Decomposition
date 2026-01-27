import os
from natsort import natsorted

import networkx as nx

from cfl.elements.cfl_environment import CFL_Environment
from cfl.solver.single_source.sscfl_critical_resources import SSCFL_Critical_Resources
from cfl.solver.single_source.sscfl_heuristic_solver import SSCFL_Heuristic_solver




def solve_ss_cfl_instances():
    root = "cfl/instances/single_source"

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

                print(f"file {file_path}    complete = {total_solution_cost(env.G, env.elements, env.facilities, is_TB_instance)}")

                n_clients = len(env.elements)
                n_facilities = len(env.facilities)

                k_values = [n_facilities // i for i in [4, 3, 2]]
                max_size_values = [n_clients // i for i in [10, 5, 3, 2]]

                for k in k_values:
                    for s in max_size_values:
                        env = CFL_Environment(max_cluster_size=s, k=k, instance_file=file_path)
                        env.solve_clusters(lambda c: SSCFL_Heuristic_solver(env.G, c.elements, c.facilities))
                        print(f"    k = {k}, max size = {s}, n clusters = {len(env.clusters)}, clusters value = {total_solution_cost(env.G, env.elements, env.facilities, is_TB_instance)} ({"OK" if check_solution(env.elements, env.facilities) else "NOT OK"})", end="   ")

                        critical_resources = SSCFL_Critical_Resources(env.elements, env.facilities)

                        if critical_resources.is_initially_feasible:
                            print("Feasible da subito")
                        else:
                            critical_resources.unassign_items()
                            post_solver = SSCFL_Heuristic_solver(env.G, env.elements, env.facilities, critical_resources)
                            post_solver.solve()
                            print(f"final value = {total_solution_cost(env.G, env.elements, env.facilities, is_TB_instance)} ({"OK" if check_solution(env.elements, env.facilities) else "NOT OK"})   status = {post_solver.status}")




def check_solution(clients, facilities):
    for c in clients:
        shipped = sum(c.shipment_by_facility.values())
        if shipped != c.demand:
            return False
        if len(c.shipment_by_facility) != 1:
            return False

    for f in facilities:
        shipped = sum(f.shipment_by_client.values())
        if shipped > f.capacity:
            return False

    return True






















