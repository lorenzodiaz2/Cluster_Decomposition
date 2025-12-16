import ast
import pandas as pd

from elements.environment import Environment
from scalability.resolution_scalability import run_scalability

def get_data_frame():
    return pd.DataFrame({
        "grid side": pd.Series(dtype="int"),
        "n quadrants": pd.Series(dtype="int"),
        "n pairs": pd.Series(dtype="int"),
        "n agents": pd.Series(dtype="int"),
        "max cluster size": pd.Series(dtype="int"),
        "offset": pd.Series(dtype="int"),
        "k": pd.Series(dtype="int"),
        "seed": pd.Series(dtype="int"),
        "restrict paths to quadrant": pd.Series(dtype="bool"),
        "env time": pd.Series(dtype="float"),
        "model times complete": pd.Series(dtype="object"),
        "resolution times complete": pd.Series(dtype="object"),
        "status complete": pd.Series(dtype="object"),
        "n clusters": pd.Series(dtype="int"),
        "similarity index": pd.Series(dtype="float"),
        "cluster similarity indexes": pd.Series(dtype="object"),
        "cluster congestion indexes": pd.Series(dtype="object"),
        "similarity matrix time": pd.Series(dtype="float"),
        "nj time": pd.Series(dtype="float"),
        "n agents per cluster": pd.Series(dtype="object"),
        "od pairs per cluster": pd.Series(dtype="object"),
        "model times clusters": pd.Series(dtype="object"),
        "resolution times clusters": pd.Series(dtype="object"),
        "clusters status": pd.Series(dtype="object"),
        "UBs clusters": pd.Series(dtype="object"),
        "LBs clusters": pd.Series(dtype="object"),
        "critical resources creation times": pd.Series(dtype="object"),
        "unassigned agents": pd.Series(dtype="object"),
        "unassigning agents times": pd.Series(dtype="object"),
        "model times final": pd.Series(dtype="object"),
        "resolution times final": pd.Series(dtype="object"),
        "status final": pd.Series(dtype="object"),
        "UB complete": pd.Series(dtype="float"),
        "LB complete": pd.Series(dtype="float"),
        "UB clusters": pd.Series(dtype="float"),
        "LB clusters": pd.Series(dtype="float"),
        "final delay": pd.Series(dtype="int"),
        "total time complete": pd.Series(dtype="float"),
        "total time clusters + post": pd.Series(dtype="float")
    })

if __name__ == '__main__':
    df = pd.read_csv("results/20_test_2-9.csv")
    E_abs_list = []
    R_max_list = []

    for index, row in df.iterrows():
        grid_side = int(row["grid side"])
        n_quadrants = int(row["n quadrants"])
        n_pairs_per_quadrant = int(row["n pairs"]) // n_quadrants
        max_cluster_size = int(row["max cluster size"])
        offset = int(row["offset"])
        k = int(row["k"])
        seed = int(row["seed"])
        restrict_paths_to_quadrants = row["restrict paths to quadrant"]
        total_time_complete = round(float(row["total time complete"]), 2)
        total_time_clusters_complete = round(float(row["total time clusters + post"]), 2)
        cluster_congestion_indexes = ast.literal_eval(row["cluster congestion indexes"])
        round_indexes = [round(x, 3) for x in cluster_congestion_indexes]

        resolution_times_clusters = ast.literal_eval(row["resolution times clusters"])
        round_times_clusters = [round(x, 3) for x in resolution_times_clusters]

        augmented_T_times = len(ast.literal_eval(row["resolution times complete"]))

        final_delay = int(row["final delay"])

        print(f"{grid_side}  {n_quadrants}  {n_pairs_per_quadrant}  {restrict_paths_to_quadrants}  {offset}  {seed}")
        env = Environment(grid_side, max_cluster_size, n_quadrants, n_pairs_per_quadrant, offset, k, seed=seed, restrict_paths_to_quadrant=restrict_paths_to_quadrants)
        for i in range(augmented_T_times - 1):
            for od in env.od_pairs:
                od.delay_shortest_paths(od.T + 1)
                od.T += 1
        env.compute_clusters(refinement_levels=0)
        E_abs = env.cluster_congestion_indexes_absolute
        R_max = env.cluster_congestion_ratio_max

        E_abs_list.append(E_abs)
        R_max_list.append(R_max)
        df.at[index, "nj time"] = env.nj_time

    df.insert(17, "cluster congestion indexes absolute", E_abs_list)
    df.insert(18, "cluster congestion ratio max", R_max_list)
    df.to_csv("results/20_test_2-9.csv", index=False)


    exit(0)



    run_scalability(20, 132, True, -1, df)
    run_scalability(20, 134, True, -1, df)
    run_scalability(20, 136, True, -1, df)
    run_scalability(20, 138, True, -1, df)
    run_scalability(20, 140, True, -1, df)


    offset_values = [0, 2, 4, 6, 8, 10]
    n_pairs_per_quadrant_values = [100, 108, 116, 124, 132, 134, 136, 138, 140]


    for offset in offset_values:
        for restrict_paths_to_quadrants in [False, True]:
            for n_pairs_per_quadrant in n_pairs_per_quadrant_values:
                run_scalability(20, n_pairs_per_quadrant, restrict_paths_to_quadrants, offset, df)





# todo creare una classe results che tenga il dataframe e la matrice di similarità (...)
# todo pensare alla metrica di bontà dei clusters  ---->  IN TEORIA FATTO

# todo vedere se ha senso parallelizzare il calcolo degli indici di bontà dei clusters (non penso...)
# todo pensare alla metrica di sovrapposizione spazio (spazio-temporale) dentro i clusters