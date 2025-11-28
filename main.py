from scalability.resolution_scalability import run_scalability
from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from elements.environment import Environment

if __name__ == '__main__':

    run_scalability()

    exit(0)

    parallel_times = {}
    no_parallel_times = {}

    for n_quadrants in range(3, 17):
        print(f"\nnumero di quadranti = {n_quadrants}")
        parallel_times[n_quadrants] = []
        no_parallel_times[n_quadrants] = []
        for i in range(3):
            print(f"    iterazione {i}     ", end="")
            env = Environment(60, 750, n_quadrants, 150, 0, 12, False)
            print(f"        env: {round(env.set_time, 2)}    ", end="")

            env.compute_clusters()
            parallel_times[n_quadrants].append(env.matrix_time)
            print(f"matrix: {round(env.matrix_time, 2)}     {len(env.clusters)} clusters ({[c.n_agents for c in env.clusters]})     ", end="")

            complete_solver = Heuristic_Solver(env.G, env.od_pairs)
            complete_solver.solve()
            complete_solver.assign_solutions()
            print(f"z completo = {sum(a.delay for a in env.agents)}", end="     ")

            for cluster in env.clusters:
                hs = Heuristic_Solver(env.G, [od_pair for od_pair in cluster.od_pairs])
                hs.solve()
                hs.assign_solutions()
            print(f"z clusters = {sum(a.delay for a in env.agents)}", end="     ")

            critical_resources = Critical_Resources(env.G, env.od_pairs)

            if critical_resources.is_initially_feasible:
                print("Soluzione già feasible")
            else:
                critical_resources.unassign_agents()
                print(f"rimossi {len(critical_resources.removed_agents)}", end="     ")
                final_solver = Heuristic_Solver(env.G, env.od_pairs, critical_resources)
                final_solver.solve()
                final_solver.assign_solutions()
                print(f"z finale = {sum(a.delay for a in env.agents)}")


# todo cambiare il salvataggio e salvare su csv usando pandas
# todo fare qualche prova su istanze piccole
# todo iniziare a tirare fuori qualche numero