import pandas as pd

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
    df = get_data_frame()

    offset_values = [-1, 0, 1, 2, 3, 4, 5,6 ,7, 8, 9, 10]
    n_pairs_per_quadrant_values = [100, 108, 116, 124, 132]

    for offset in offset_values:
        for restrict_paths_to_quadrants in [False, True]:
            for n_pairs_per_quadrant in n_pairs_per_quadrant_values:
                run_scalability(20, n_pairs_per_quadrant, restrict_paths_to_quadrants, offset, df)





# todo creare una classe results che tenga il dataframe e la matrice di similarità (...)
# todo pensare alla metrica di bontà dei clusters  ---->  IN TEORIA FATTO, DA PROVARE (VEDERE SOTTO)
# todo provare a creare i clusters con od random (prese da qualsiasi quadrante) e quindi vedere il valore degli indici di similarità. Dovrebbero essere vicino lo zero
# todo mettere restric_paths_to_quadrant a False e fare le stesse prove (magari direttamente con 10 istanze)

# todo fare altre 5 prove sui test già fatti, recuperare il seed e aumentarlo
# todo vedere se ha senso parallelizzare il calcolo degli indici di bontà dei clusters (non penso...)
# todo pensare alla metrica di sovrapposizione spazio (spazio-temporale) dentro i clusters