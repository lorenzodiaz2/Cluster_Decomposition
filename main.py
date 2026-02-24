import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

from mcpa.elements.mcpa_environment import MCPA_Environment
from mcpa.elements.pair import OD_Pair
from utils.read_instance import get_min_from_array_string


def compute_spearman(df, x, y):
    coef, p_value = spearmanr(df[f"{x}"], df[f"{y}"])
    print(f"Coefficiente di Spearman tra {x} e {y}: {coef}")
    print(f"P-value: {p_value}")




def scatter_plot(x, y):
    plt.scatter(x, y, color='blue', marker='o')
    # m, q = np.polyfit(x, y, 1)
    # plt.plot(x, m*np.array(x) + q, color='red', label=f'Trend (y={m:.2f}x + {q:.2f})')

    plt.title("Scatter Plot")
    plt.xlabel("Asse X")
    plt.ylabel("Asse Y")
    # plt.legend()
    plt.grid(True)
    plt.show()



def plot_similarity_scatter():
    df = pd.read_csv("results/mcpa/mcpa_results.csv")
    df = df[df['UB final'] == df['LB final']]
    df = df[df['n pairs per quadrant'] == 150]
    scatter_plot(df["similarity index"], df["gap"])
    compute_spearman(df, "similarity index", "gap")

    df = pd.read_csv("results/cfl/ss/sscfl_results.csv")
    df = df[df['UB final'] == df['LB final']]
    df = df[df['n clients per quadrant'] == 175]
    scatter_plot(df["similarity index"], df["gap"])
    compute_spearman(df, "similarity index", "gap")

    df = pd.read_csv("results/cfl/ms/mscfl_results.csv")
    df = df[df['UB final'] == df['LB final']]
    df = df[df['n clients per quadrant'] == 175]
    scatter_plot(df["similarity index"], df["gap"])
    compute_spearman(df, "similarity index", "gap")



def compute_additional_metrics(df):
    silhouette_list = []
    max_mean_list = []

    for index, row in df.iterrows():

        _grid_side = int(row["grid side"])
        _n_quadrants = int(row["n quadrants"])
        _n_pairs_per_quadrant = int(row["n pairs per quadrant"])
        _max_cluster_size = int(row["max cluster size"])
        _offset = int(row["offset"])
        _k = int(row["k"])
        _seed = int(row["seed"])
        n = len(str(row["model times complete"]).split(",")) - 1

        env = MCPA_Environment(_grid_side, _max_cluster_size, _n_quadrants,
                               _n_pairs_per_quadrant, _offset, _k, seed=_seed)
        print(index)

        for i in range(n - 1):
            for od in env.elements:
                od.delay_shortest_paths(od.T)

        env.compute_clusters(OD_Pair.compute_similarity)

        val_silhouette = env.silhouette_score
        val_max_mean = env.max_mean_intercluster_similarity

        silhouette_list.append(val_silhouette)
        max_mean_list.append(val_max_mean)

    df['silhouette_score'] = silhouette_list
    df['max_mean_intercluster_sim'] = max_mean_list

    return df



if __name__ == '__main__':
    df = pd.read_csv("results/mcpa/150_mcpa_results.csv")
    new_df = compute_additional_metrics(df)
    new_df.to_csv("results/mcpa/new_150_mcpa_results.csv", index=False)


    exit(0)
    plot_similarity_scatter()

    print("\n\n==============================\n\n")

    df = pd.read_csv("results/mcpa/mcpa_results.csv")
    df = df[df['n pairs per quadrant'] == 150]
    df = df[df['UB final'] == df['LB final']]
    df["min similarity index"] = df["cluster similarity indexes"].apply(get_min_from_array_string)
    scatter_plot(df["min similarity index"], df["gap"])
    compute_spearman(df, "min similarity index", "gap")


    df = pd.read_csv("results/cfl/ss/sscfl_results.csv")
    df = df[df['n clients per quadrant'] == 175]
    df = df[df['UB final'] == df['LB final']]
    df["min similarity index"] = df["cluster similarity indexes"].apply(get_min_from_array_string)
    scatter_plot(df["min similarity index"], df["gap"])
    compute_spearman(df, "min similarity index", "gap")


    df = pd.read_csv("results/cfl/ms/mscfl_results.csv")
    df = df[df['n clients per quadrant'] == 175]
    df = df[df['UB final'] == df['LB final']]
    df["min similarity index"] = df["cluster similarity indexes"].apply(get_min_from_array_string)
    scatter_plot(df["min similarity index"], df["gap"])
    compute_spearman(df, "min similarity index", "gap")










    exit(0)
    print("SPACE-TIME CAPACITATED PATH ASSIGNMENT\n")

    df = pd.read_csv("results/mcpa/mcpa_results.csv")
    compute_spearman(df, "n agents", "total time complete")
    compute_spearman(df, "global congestion absolute", "total time complete")
    compute_spearman(df, "cross congestion absolute", "total time heuristic")
    compute_spearman(df, "similarity index", "total time heuristic")
    df = df[df['UB final'] == df['LB final']]
    compute_spearman(df, "similarity index", "gap")


    print("\n\n============================================\n\n")
    print("CAPACITATED FACILITY LOCATION PROBLEM - SINGLE SOURCE\n")

    df = pd.read_csv("results/cfl/ss/sscfl_results.csv")
    df["instance size"] = df['n clients per quadrant'] * df["n facilities per quadrant"] * df["n quadrants"]
    compute_spearman(df, "instance size", "total time complete")
    compute_spearman(df, "global congestion absolute", "total time complete")
    compute_spearman(df, "similarity index", "total time heuristic")
    compute_spearman(df, "cross congestion absolute", "total time heuristic")
    df = df[df['UB final'] == df['LB final']]
    compute_spearman(df, "similarity index", "gap")


    print("\n\n============================================\n\n")
    print("CAPACITATED FACILITY LOCATION PROBLEM - SPLITTABLE DEMAND\n")

    df = pd.read_csv("results/cfl/ms/mscfl_results.csv")
    df["instance size"] = df['n clients per quadrant'] * df["n facilities per quadrant"] * df["n quadrants"]
    compute_spearman(df, "instance size", "total time complete")
    compute_spearman(df, "global congestion absolute", "total time complete")
    compute_spearman(df, "similarity index", "total time heuristic")
    compute_spearman(df, "cross congestion absolute", "total time heuristic")
    df = df[df['UB final'] == df['LB final']]
    compute_spearman(df, "similarity index", "gap")











# todo single source sono apposto
# todo TB sono apposto quando uso Multi Source, quando uso Single Source sembrerebbe apposto quando modifico SSCFL_Heuristic_solver -> _compute_shipping_cost -> vedere commento
# todo multi source (TEST_BED_C) sono da trovare le soluzioni. Esistono anche TEST_BED_A e TEST_BED_C a questo link: http://wpage.unina.it/sforza/test/