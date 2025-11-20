from solvers.heuristic_solver import Heuristic_Solver
from utils.environment import Environment


def run():
    counter = 0
    step = 5
    num_of_pairs = 5
    grid_side = 50
    k = 12

    results: dict[int: float] = {}

    while counter <= 10:
        print(f"number of pairs = {num_of_pairs}... ", end="")
        env = Environment(grid_side, 0, num_of_pairs, grid_side // 2, k)
        solver = Heuristic_Solver(env.G, env.od_pairs, env.T)
        solver.solve()

        total_time = sum(m_times for m_times in solver.model_times) + sum(r_time for r_time in solver.resolution_times)
        results[num_of_pairs] = total_time

        print(f"Solved in {total_time}     {"COUNTER UPDATED" if solver.status == "TIME_LIMIT" else ""}")

        if solver.status == "TIME_LIMIT":
            counter += 1

        num_of_pairs += step

run()