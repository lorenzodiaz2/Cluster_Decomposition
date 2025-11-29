from pandas import DataFrame

from solvers.heuristic_solver import Heuristic_Solver
from solvers.post_processing import Critical_Resources
from elements.environment import Environment


def save_results(
    env: Environment,
    complete_solver: Heuristic_Solver,
    cluster_solvers: list[Heuristic_Solver],
    df: DataFrame,
    i: int,
    critical_resources: Critical_Resources | None = None,
    final_solver: Heuristic_Solver | None = None,
):
    grid_side = env.grid_side
    n_quadrants = env.n_quadrants
    n_pairs = len(env.od_pairs)
    n_agents = len(env.agents)
    max_cluster_size = env.max_cluster_size
    offset = env.offset
    k = env.k
    seed = i
    env_time = env.set_time

    model_times_complete = complete_solver.model_times
    resolution_times_complete = complete_solver.resolution_times
    status_complete = complete_solver.status
    UB_complete = complete_solver.m.ObjVal
    LB_complete = complete_solver.m.ObjBound

    n_clusters = len(env.clusters)
    similarity_matrix = env.similarity_matrix
    similarity_matrix_time = env.matrix_time
    nj_time = env.nj_time
    n_agents_per_cluster = [c.n_agents for c in env.clusters]
    od_pairs_per_cluster = [[od.id for od in c.od_pairs] for c in env.clusters]

    model_times_clusters = []
    resolution_times_clusters = []
    clusters_status = []
    UBs_clusters = []
    LBs_clusters = []
    UB_clusters = 0
    LB_clusters = 0

    critical_resources_creation_times = None
    unassigned_agents = None
    unassigning_times = None
    model_times_final = None
    resolution_times_final = None
    status_final = None
    final_delay = None
    total_time_complete = sum(mt for mt in complete_solver.model_times) + sum(rt for rt in complete_solver.resolution_times)
    total_time_clusters_post = similarity_matrix_time + nj_time

    for hs in cluster_solvers:
        model_times_clusters.append(hs.model_times[0])
        resolution_times_clusters.append(hs.resolution_times[0])
        clusters_status.append(hs.status)
        UBs_clusters.append(hs.m.ObjVal)
        LBs_clusters.append(hs.m.ObjBound)
        total_time_clusters_post += hs.model_times[0] + hs.resolution_times[0]
        UB_clusters += hs.m.ObjVal
        LB_clusters += hs.m.ObjBound

    if critical_resources is not None:
        critical_resources_creation_times = critical_resources.creation_times
        total_time_clusters_post += sum(ct for ct in critical_resources_creation_times)
        if final_solver is not None:
            unassigned_agents = critical_resources.unassigned_agents_per_tol
            unassigning_times = critical_resources.unassigning_times
            total_time_clusters_post += sum(ut for ut in unassigning_times)

            model_times_final = final_solver.model_times
            resolution_times_final = final_solver.resolution_times
            total_time_clusters_post += sum(mt for mt in model_times_final) + sum(rt for rt in resolution_times_final)
            status_final = final_solver.status
            final_delay = sum(a.delay for a in env.agents)

    df.loc[len(df)] = [
        grid_side, n_quadrants, n_pairs, n_agents, max_cluster_size, offset, k, seed, env_time,
        model_times_complete, resolution_times_complete, status_complete,
        n_clusters, similarity_matrix, similarity_matrix_time, nj_time, n_agents_per_cluster, od_pairs_per_cluster, model_times_clusters, resolution_times_clusters, clusters_status, UBs_clusters, LBs_clusters,
        critical_resources_creation_times, unassigned_agents, unassigning_times,
        model_times_final, resolution_times_final, status_final, UB_complete, LB_complete, UB_clusters, LB_clusters, final_delay, total_time_complete, total_time_clusters_post
    ]



def read_results():
    pass