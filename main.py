import pandas as pd

from scalability.resolution_scalability import run_scalability

def get_data_frame():
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

        # =================================================================

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
        "final delay": pd.Series(dtype="int"),
        "total time complete": pd.Series(dtype="float"),
        "total time clusters + post": pd.Series(dtype="float")
    })

if __name__ == '__main__':
    df = get_data_frame()
    seed = 0
    run_scalability(20, 80, 0, df, seed)
    seed += 80
    run_scalability(20, 90, 0, df, seed)



    exit(0)
    offset_values = [0, 2, 4, 6, 8, 10]
    n_pairs_per_quadrant_values = [100, 108, 116, 124, 132, 134, 136, 138, 140]
    seed = 0

    for offset in offset_values:
        for restrict_paths_to_quadrants in [False, True]:
            for n_pairs_per_quadrant in n_pairs_per_quadrant_values:
                run_scalability(20, n_pairs_per_quadrant, offset, df, seed)
                seed += 80
