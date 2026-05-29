import pandas as pd
from scipy.stats import spearmanr


from cfl.elements.cfl_environment import CFL_Environment
from cfl.elements.client import Client
from utils.plot_functions import scatter_plot


def compute_spearman(df, x, y):
    coef, p_value = spearmanr(df[f"{x}"], df[f"{y}"])
    print(f"{round(coef, 2)}", end="")



def plot_similarity_scatter():
    df = pd.read_csv("results/mcpa/150_mcpa_results.csv")
    df = df[df['UB final'] == df['LB final']]
    df["speedup"] = df["total time complete"] / df["total time heuristic"]
    scatter_plot(df["similarity index"], df["speedup"], df["gap"], df["total time heuristic"], "results/mcpa/img/scatter_SI-gap.pdf", "Speedup")
    compute_spearman(df, "similarity index", "gap")

    df = pd.read_csv("results/cfl/ss/175_sscfl_results.csv")
    df = df[df['UB final'] == df['LB final']]
    df["speedup"] = df["total time complete"] / df["total time heuristic"]
    scatter_plot(df["similarity index"], df["speedup"], df["gap"], df["total time heuristic"], "results/cfl/ss/img/scatter_SI-gap.pdf", "Speedup")
    compute_spearman(df, "similarity index", "gap")

    df = pd.read_csv("results/cfl/ms/175_mscfl_results.csv")
    df = df[df['UB final'] == df['LB final']]
    df["speedup"] = df["total time complete"] / df["total time heuristic"]

    scatter_plot(df["similarity index"], df["speedup"], df["gap"], df["total time heuristic"], "results/cfl/ms/img/scatter_SI-gap.pdf", "Speedup")
    compute_spearman(df, "similarity index", "gap")



def compute_additional_metrics(df):
    silhouette_list = []
    max_mean_list = []

    for index, row in df.iterrows():

        _grid_side = int(row["grid side"])
        _n_quadrants = int(row["n quadrants"])
        _n_clients_per_quadrant = int(row["n clients per quadrant"])
        _n_facilities_per_quadrant = int(row["n facilities per quadrant"])
        _max_cluster_size = int(row["max cluster size"])
        _offset = int(row["offset"])
        _k = int(row["k"])
        _seed = int(row["seed"])
        n = len(str(row["model times complete"]).split(",")) - 1

        env = CFL_Environment(_grid_side, _max_cluster_size, _n_quadrants, _n_clients_per_quadrant, _n_facilities_per_quadrant, _offset, _k, seed=_seed)

        for i in range(n - 1):
            for client in env.elements:
                client.add_facility()

        env.compute_clusters(Client.compute_similarity)

        val_silhouette = env.silhouette_score
        val_max_mean = env.max_mean_intercluster_similarity

        silhouette_list.append(val_silhouette)
        max_mean_list.append(val_max_mean)

    df['silhouette_score'] = silhouette_list
    df['max_mean_intercluster_sim'] = max_mean_list

    return df



if __name__ == '__main__':
    print()














# todo single source sono apposto
# todo TB sono apposto quando uso Multi Source, quando uso Single Source sembrerebbe apposto quando modifico SSCFL_Heuristic_solver -> _compute_shipping_cost -> vedere commento
# todo multi source (TEST_BED_C) sono da trovare le soluzioni. Esistono anche TEST_BED_A e TEST_BED_C a questo link: http://wpage.unina.it/sforza/test/