import pandas as pd
import numpy as np
from isa import ISA


def test_isa(df, dir):
    print(f"Starting df length: {len(df)}")
    df = df[df["status heuristic"] == "OPTIMAL"]
    print(f"Truncated non optimal instances df length: {len(df)}")
    # df = df[df["mean resource capacity"].isin([5, 6])]
    # print(f"Truncated 7, 8, 9 capacities instances df length: {len(df)}")
    df = df[df["final unassigned agents"] / df["n agents"] < 0.65]
    print(f"Truncated too many removed agents instances df length: {len(df)}")

    df["instance"] = np.array(range(df.shape[0])).astype(str)

    feat_names = [ # aggiungere la grid side
        'n agents', 'max cluster size', 'mean resource capacity', 'n clusters', 'similarity index',
        'min cluster similarity index', 'global congestion absolute', 'cross congestion absolute', 'global congestion ratio max',
        'cross congestion rate', 'cross congestion share', 'silhouette score', 'max mean intercluster similarity',
        'normalized macro components', 'max eigengap', 'clustering distortion ratio', 'absorption rate', 'forced fragmentation'
    ]

    feat_names = [f for f in feat_names if df[f].nunique() > 1]

    df['dummy_solver'] = np.where(df['gap'] <= 10.0, 50.0, 0.0)
    alg_names = ['gap', 'dummy_solver']

    custom_options = {
        'perf': {
            'max_perf': False,
            'abs_perf': True,
            'epsilon': 10.0
        },
        'parallel': {'n_cores': 4},
        'sifted': {'rho': 0.3, 'k': 10}
    }

    instance_col_name = 'instance'

    isa = ISA(df, instance_col_name, feat_names, alg_names, custom_options)
    isa.run(verbose=True)

    print("Accuracy:", isa.isa.model.pythia.accuracy)
    print("Summary:", isa.isa.model.pythia.summary)
    isa.save_plots(f"{dir}/img")
    isa.save_full_csv(df, f"{dir}/csv")



# df = pd.read_csv("../../results/mcpa/small_instances_uniform.csv")
# condizione = pd.to_numeric(df['gap'], errors='coerce') <= 10
# conteggio_per_offset = condizione.groupby(df['offset']).sum().astype(int)
# percentuale_per_offset = condizione.groupby(df['offset']).mean() * 100
# print(conteggio_per_offset, percentuale_per_offset)
#
#
# print("\n")
#
# df = pd.read_csv("../../results/mcpa/small_instances_inverse.csv")
# condizione = pd.to_numeric(df['gap'], errors='coerce') <= 10
# conteggio_per_offset = condizione.groupby(df['offset']).sum().astype(int)
# percentuale_per_offset = condizione.groupby(df['offset']).mean() * 100
# print(conteggio_per_offset, percentuale_per_offset)
#
# print("\n")
#
# df = pd.read_csv("../../results/mcpa/small_instances_exponential.csv")
# condizione = pd.to_numeric(df['gap'], errors='coerce') <= 10
# conteggio_per_offset = condizione.groupby(df['offset']).sum().astype(int)
# percentuale_per_offset = condizione.groupby(df['offset']).mean() * 100
# print(conteggio_per_offset, percentuale_per_offset)
#
#
# exit(0)



df = pd.read_csv("../../results/mcpa/big/big_instances_EXP_LIN.csv")
test_isa(df, "../test/big/exp_lin")
