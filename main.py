import pandas as pd

from mcpa.scalability.mcpa_scalability_resolution import run_mcpa_scalability

if __name__ == '__main__':
    df = pd.read_csv("results/mcpa/results.csv")
    run_mcpa_scalability(20, 150, 750, -1, df, 0, [2, 3], 3)
    run_mcpa_scalability(20, 150, 750, 0, df, 0, [2, 3], 3)
    run_mcpa_scalability(20, 150, 750, 1, df, 0, [2, 3], 3)
    run_mcpa_scalability(20, 150, 750, 2, df, 0, [2, 3], 3)



    columns = [
        "grid side",
        "n quadrants",
        "n pairs per quadrant",
        "n agents",
        "max cluster size",
        "offset",
        "k",
        "seed",

        "time global",
        "status global",
        "number of resolution",
        "LB global",
        "UB global",

        "n clusters",
        "n agents per cluster",
        "similarity index",
        "min cluster similarity index",
        "max mean intercluster similarity",
        "silhouette score",
        "cluster congestion ratio max",
        "global congestion absolute",
        "cross congestion absolute",
        "cross congestion rate",
        "cross congestion share",
        "model times clusters",
        "resolution time clusters",
        "unassigned agents",
        "final tolerance",
        "LB heuristic",
        "UB heuristic",
        "time heuristic",
        "gap"
    ]

    df = pd.DataFrame(columns=columns)
    df.to_csv("results/mcpa/results.csv", index=False)















# todo single source sono apposto
# todo TB sono apposto quando uso Multi Source, quando uso Single Source sembrerebbe apposto quando modifico SSCFL_Heuristic_solver -> _compute_shipping_cost -> vedere commento
# todo multi source (TEST_BED_C) sono da trovare le soluzioni. Esistono anche TEST_BED_A e TEST_BED_C a questo link: http://wpage.unina.it/sforza/test/