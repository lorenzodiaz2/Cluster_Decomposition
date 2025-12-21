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
        "refinement levels 1": pd.Series(dtype="int"),
        "E threshold 1": pd.Series(dtype="float"),
        "R threshold 1": pd.Series(dtype="float"),
        "n clusters 1": pd.Series(dtype="int"),
        "similarity index 1": pd.Series(dtype="float"),
        "cluster similarity indexes 1": pd.Series(dtype="object"),
        "cluster congestion indexes 1": pd.Series(dtype="object"),
        "cluster congestion indexes absolute 1": pd.Series(dtype="object"),
        "cluster congestion ratio max 1": pd.Series(dtype="object"),
        "similarity matrix time 1": pd.Series(dtype="float"),
        "nj time 1": pd.Series(dtype="float"),
        "n agents per cluster 1": pd.Series(dtype="object"),
        "od pairs per cluster 1": pd.Series(dtype="object"),
        "model times clusters 1": pd.Series(dtype="object"),
        "resolution times clusters 1": pd.Series(dtype="object"),
        "clusters status 1": pd.Series(dtype="object"),
        "UBs clusters 1": pd.Series(dtype="object"),
        "LBs clusters 1": pd.Series(dtype="object"),
        "critical resources creation times 1": pd.Series(dtype="object"),
        "unassigned agents 1": pd.Series(dtype="object"),
        "unassigning agents times 1": pd.Series(dtype="object"),
        "model times final 1": pd.Series(dtype="object"),
        "resolution times final 1": pd.Series(dtype="object"),
        "status final 1": pd.Series(dtype="object"),
        "UB clusters 1": pd.Series(dtype="float"),
        "LB clusters 1": pd.Series(dtype="float"),
        "final delay 1": pd.Series(dtype="int"),
        "total time clusters + post 1": pd.Series(dtype="float"),

        # =================================================================
        # VALORI SOLUZIONE CLUSTERS CON RECLUSTERIZZAZIONE
        "refinement levels 2": pd.Series(dtype="int"),
        "E threshold 2": pd.Series(dtype="float"),
        "R threshold 2": pd.Series(dtype="float"),
        "n clusters 2": pd.Series(dtype="int"),
        "similarity index 2": pd.Series(dtype="float"),
        "cluster similarity indexes 2": pd.Series(dtype="object"),
        "cluster congestion indexes 2": pd.Series(dtype="object"),
        "cluster congestion indexes absolute 2": pd.Series(dtype="object"),
        "cluster congestion ratio max 2": pd.Series(dtype="object"),
        "similarity matrix time 2": pd.Series(dtype="float"),
        "nj time 2": pd.Series(dtype="float"),
        "n agents per cluster 2": pd.Series(dtype="object"),
        "od pairs per cluster 2": pd.Series(dtype="object"),
        "model times clusters 2": pd.Series(dtype="object"),
        "resolution times clusters 2": pd.Series(dtype="object"),
        "clusters status 2": pd.Series(dtype="object"),
        "UBs clusters 2": pd.Series(dtype="object"),
        "LBs clusters 2": pd.Series(dtype="object"),
        "critical resources creation times 2": pd.Series(dtype="object"),
        "unassigned agents 2": pd.Series(dtype="object"),
        "unassigning agents times 2": pd.Series(dtype="object"),
        "model times final 2": pd.Series(dtype="object"),
        "resolution times final 2": pd.Series(dtype="object"),
        "status final 2": pd.Series(dtype="object"),
        "UB clusters 2": pd.Series(dtype="float"),
        "LB clusters 2": pd.Series(dtype="float"),
        "final delay 2": pd.Series(dtype="int"),
        "total time clusters + post 2": pd.Series(dtype="float")
    })

if __name__ == '__main__':
    # df = pd.read_csv("results/new_test.csv")
    df = get_data_frame_total()
    run_scalability(20, 20, -1, df, 0)
    run_scalability(20, 20, -1, df, 1)

    exit(0)
    offset_values = [-1, 0, 2, 5, 8, 10, 12, 15]
    n_pairs_per_quadrant_values = [110, 118, 126, 132, 134, 136, 138, 140, 145, 150]
    seed = 0

    for offset in offset_values:
        for n_pairs_per_quadrant in n_pairs_per_quadrant_values:
            run_scalability(20, n_pairs_per_quadrant, offset, df, seed)
            seed += 80
            print()
        print()
