import os
from pathlib import Path

import pandas as pd
from natsort import natsorted

from caps.cfl.elements.cfl_environment import CFL_Environment
from caps.cfl.scalability.cfl_scalability_resolution import save_results
from caps.cfl.solver.cfl_heuristic_solver import total_solution_cost
from caps.cfl.solver.single_source.sscfl_critical_resources import SSCFL_Critical_Resources
from caps.cfl.solver.single_source.sscfl_heuristic_solver import SSCFL_Heuristic_Solver



def solve_cfl_instances(root):

    df = pd.read_csv("")


    for folder_name in natsorted(os.listdir(root)):
        folder_path = os.path.join(root, folder_name)
        if not os.path.isdir(folder_path) or "solutions" in folder_path:
            continue

        for file_name in natsorted(os.listdir(folder_path)):
            file_path = os.path.join(folder_path, file_name)

            if "Reame" in file_path or "format" in file_path:
                continue

            is_TB_instance = True if file_path.__contains__("TB") else False # todo sarà da cambiare

            if os.path.isfile(file_path):
                env = CFL_Environment(instance_file=file_path, sim_method=("exponential", 0.95), resource_weights="linear", k=30, max_cluster_size=600)

                global_solver = SSCFL_Heuristic_Solver(env.G, env.elements, env.facilities)
                global_solver.solve()

                print(f"file {file_path}    global = {total_solution_cost(env.elements, env.facilities, is_TB_instance)}   ", end="")

                if "Holmberg" in file_path:
                    sol_path = "Holmberg_Solution_Values.txt"
                elif "OR" in file_path:
                    sol_path = "OR-Library_Solution_Values.txt"
                else:
                    sol_path = "Yang_Solution_Values.txt"

                path = Path("instances/SS/solutions/" + sol_path)
                tokens = path.read_text().split()
                it = iter(tokens)
                instance_name = file_path.split("/")[-1]

                word = next(it)

                while instance_name != word:
                    word = next(it)

                for i in range(5):
                    next(it)

                opt_value = float(next(it))
                print(opt_value, end="   ")

                print("OTTIMO" if opt_value == total_solution_cost(env.elements, env.facilities, is_TB_instance) else "NOT OPTIMAL")






                env.solve_clusters(lambda en, c: SSCFL_Heuristic_Solver(env.G, c.elements, c.facilities))

                print(f"    k = {10}, max size = {600}, n clusters = {len(env.clusters)}, clusters value = {total_solution_cost(env.elements, env.facilities, is_TB_instance)}", end="   ")

                critical_resources = SSCFL_Critical_Resources(env.elements, env.facilities)
                repair_solver = None

                if critical_resources.is_initially_feasible:
                    print("Feasible da subito")
                else:
                    critical_resources.release_units()
                    repair_solver = SSCFL_Heuristic_Solver(env.G, env.elements, env.facilities, critical_resources)
                    repair_solver.solve()
                    print(f"final value = {total_solution_cost(env.elements, env.facilities, is_TB_instance)}   status = {repair_solver.status}")

                save_results(env, global_solver, 42, df, critical_resources, repair_solver)
