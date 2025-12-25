import pandas as pd

from scalability.resolution_scalability import run_scalability

def get_data_frame_total():
    return pd.DataFrame({
        # =================================================================
        # PARAMETRI DI MODELLO
        "grid side": pd.Series(dtype="int"),
        "n quadrants": pd.Series(dtype="int"),
        "n pairs per quadrant": pd.Series(dtype="int"),
        "n agents": pd.Series(dtype="int"),
        "max cluster size": pd.Series(dtype="int"),
        "offset": pd.Series(dtype="int"),
        "k": pd.Series(dtype="int"),
        "seed": pd.Series(dtype="int"),
        "time limit": pd.Series(dtype="int"),
        "restrict paths to quadrant": pd.Series(dtype="bool"),
        "env time": pd.Series(dtype="float"),

        # =================================================================
        # VALORI SOLUZIONE COMPLETA
        "model times complete": pd.Series(dtype="object"),
        "resolution times complete": pd.Series(dtype="object"),
        "status complete": pd.Series(dtype="object"),
        "UB complete": pd.Series(dtype="float"),
        "LB complete": pd.Series(dtype="float"),
        "gap complete": pd.Series(dtype="float"),
        "incumbent times": pd.Series(dtype="object"),
        "incumbent solutions": pd.Series(dtype="object"),
        "total time complete": pd.Series(dtype="float"),

        # =================================================================
        # VALORI SOLUZIONE CLUSTERS SENZA RECLUSTERIZZAZIONE
        "n clusters": pd.Series(dtype="int"),
        "similarity index": pd.Series(dtype="float"),
        "cluster similarity indexes": pd.Series(dtype="object"),
        "cluster congestion indexes": pd.Series(dtype="object"),
        "cluster congestion indexes absolute": pd.Series(dtype="object"),
        "cluster congestion ratio max": pd.Series(dtype="object"),
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
        "UB clusters": pd.Series(dtype="float"),
        "LB clusters": pd.Series(dtype="float"),
        "final delay": pd.Series(dtype="float"),
        "total time clusters + post": pd.Series(dtype="float"),

        # =================================================================
        # VALORI SOLUZIONE CLUSTERS CON RECLUSTERIZZAZIONE
        # "refinement levels 2": pd.Series(dtype="int"),
        # "E threshold 2": pd.Series(dtype="float"),
        # "R threshold 2": pd.Series(dtype="float"),
        # "n clusters 2": pd.Series(dtype="int"),
        # "similarity index 2": pd.Series(dtype="float"),
        # "cluster similarity indexes 2": pd.Series(dtype="object"),
        # "cluster congestion indexes 2": pd.Series(dtype="object"),
        # "cluster congestion indexes absolute 2": pd.Series(dtype="object"),
        # "cluster congestion ratio max 2": pd.Series(dtype="object"),
        # "similarity matrix time 2": pd.Series(dtype="float"),
        # "nj time 2": pd.Series(dtype="float"),
        # "n agents per cluster 2": pd.Series(dtype="object"),
        # "od pairs per cluster 2": pd.Series(dtype="object"),
        # "model times clusters 2": pd.Series(dtype="object"),
        # "resolution times clusters 2": pd.Series(dtype="object"),
        # "clusters status 2": pd.Series(dtype="object"),
        # "UBs clusters 2": pd.Series(dtype="object"),
        # "LBs clusters 2": pd.Series(dtype="object"),
        # "critical resources creation times 2": pd.Series(dtype="object"),
        # "unassigned agents 2": pd.Series(dtype="object"),
        # "unassigning agents times 2": pd.Series(dtype="object"),
        # "model times final 2": pd.Series(dtype="object"),
        # "resolution times final 2": pd.Series(dtype="object"),
        # "status final 2": pd.Series(dtype="object"),
        # "UB clusters 2": pd.Series(dtype="float"),
        # "LB clusters 2": pd.Series(dtype="float"),
        # "final delay 2": pd.Series(dtype="int"),
        # "total time clusters + post 2": pd.Series(dtype="float")
    })

GRID_SIDE = 20
n = 150
MAX_SIZE = n * 5

if __name__ == '__main__':
    # run_time_scalability(26, 10) # todo capire meglio la scalabilità del tempo

    # df = get_data_frame_total()  # todo questo è da modificare se si prendono i risultati da fuori -------------------
    df = pd.read_csv("results/test_20_750.csv")

    offset_values = [0, 2, 5]
    n_pairs_per_quadrant_values = [n, n + 5, n + 10]
    seed = 120

    for n_pairs_per_quadrant in n_pairs_per_quadrant_values:
        for offset in offset_values:
            run_scalability(GRID_SIDE, n_pairs_per_quadrant, MAX_SIZE, -1, df, seed)
            seed += 10 if offset == 0 and n_pairs_per_quadrant == 150 else 40
            print()
        print()
