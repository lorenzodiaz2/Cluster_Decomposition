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
    """
    df = pd.read_csv("results/20_test_2-9.csv")
    values_to_insert = []

    for index, row in df.iterrows():
        grid_side = int(row["grid side"])
        n_quadrants = int(row["n quadrants"])
        n_pairs_per_quadrant = int(row["n pairs"]) // n_quadrants
        max_cluster_size = int(row["max cluster size"])
        offset = int(row["offset"])
        k = int(row["k"])
        seed = int(row["seed"])
        restrict_paths_to_quadrants = bool(row["restrict paths to quadrant"])

        env = Environment(grid_side, max_cluster_size, n_quadrants, n_pairs_per_quadrant, offset, k, seed=seed, restrict_paths_to_quadrant=restrict_paths_to_quadrants)
        env.compute_clusters()
        values_to_insert.append(env.cluster_congestion_indexes)

        print(f"{grid_side}   {n_quadrants}   {n_pairs_per_quadrant}   {offset}   {restrict_paths_to_quadrants}   {env.cluster_congestion_indexes}")

    df.insert(16, "cluster congestion indexes", values_to_insert)
    df.to_csv("results/20_test_2-9.csv", index=False)
    """


    df = pd.read_csv("results/20_test_2-9.csv")

    run_scalability(20, 134, False, -1, df)
    run_scalability(20, 136, False, -1, df)
    run_scalability(20, 138, False, -1, df)
    run_scalability(20, 140, False, -1, df)


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
# todo pensare alla metrica di bontà dei clusters  ---->  IN TEORIA FATTO, DA PROVARE (VEDERE SOTTO)

# todo vedere se ha senso parallelizzare il calcolo degli indici di bontà dei clusters (non penso...)
# todo pensare alla metrica di sovrapposizione spazio (spazio-temporale) dentro i clusters
# todo trovare una soglia e, per ogni cluster vedere se la metrica di sovrapposizione supera la soglia e se supera fare un'ulteriore separazione