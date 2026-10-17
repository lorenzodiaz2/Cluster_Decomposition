import pandas as pd
import numpy as np
from isa import ISA


def test_isa(df, dir):
    print(f"Starting df length: {len(df)}")
    df = df[df["status heuristic"] == "OPTIMAL"].copy()
    print(f"Truncated non optimal instances df length: {len(df)}")


    df["instance"] = np.array(range(df.shape[0])).astype(str)

    feat_names = [
        'grid side', 'n agents', 'max cluster size', 'mean resource capacity', 'n clusters', 'similarity index',
        'min cluster similarity index', 'global congestion absolute', 'cross congestion absolute', 'global congestion ratio max',
        'cross congestion rate', 'cross congestion share', 'silhouette score', 'max mean intercluster similarity',
        'normalized macro components', 'max eigengap', 'clustering distortion ratio', 'absorption rate', 'forced fragmentation',
        "interaction density", "similarity gini", "similarity skewness", "mass cv", "boundary entities ratio", 'partition entropy', 'critical resources ratio',
        'congestion gini', 'cross cluster footprint density', 'average footprint breadth', 'average guaranteed slack', 'space to mass ratio', 'spatial density'
    ]

    feat_names = [f for f in feat_names if df[f].nunique() > 1]

    # df['heuristic_fails'] = np.where(df['gap'] <= 10.0, 50.0, 0.0)


    '''
    df["perf_global"] = df["time global"]
    # Per l'Euristica: se il gap <= 10%, la performance è il suo tempo
    # se il gap > 10%, imponi una penalità di tempo elevata
    df['perf_heuristic'] = np.where(df['gap'] <= 10.0, df['time heuristic'], 3600.0)
    alg_names = ['perf_heuristic', 'perf_global']
    '''

    df['heuristic_fails'] = np.where(df['gap'] <= 10.0, 50.0, 0.0)
    alg_names = ['gap', 'heuristic_fails']

    custom_options = {
        'perf': {
            'max_perf': False,
            'abs_perf': True,
            'epsilon': 10.0
        },
        'parallel': {'n_cores': 4},
        'sifted': {'rho': 0.3, 'k': 50}
    }

    instance_col_name = 'instance'

    isa = ISA(df, instance_col_name, feat_names, alg_names, custom_options)
    isa.run(verbose=True)

    print("Accuracy:", isa.isa.model.pythia.accuracy)
    print("Summary:", isa.isa.model.pythia.summary)
    isa.save_plots(f"{dir}/img")
    isa.save_full_csv(df, f"{dir}/csv")




df = pd.read_csv("../../results/mcpa/big/big_instances_EXP_LIN_results.csv")
test_isa(df, "../test/big/exp_lin")
