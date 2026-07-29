import pandas as pd
import numpy as np
from isa import ISA


def test_isa(df, dir):
    df["instance"] = np.array(range(df.shape[0])).astype(str)

    feat_names = [ # aggiungere la grid side
        'n agents', 'max cluster size', 'mean resource capacity', 'n clusters', 'similarity index',
        'min cluster similarity index', 'global congestion absolute', 'cross congestion absolute', 'global congestion ratio max',
        'cross congestion rate', 'cross congestion share',
        'silhouette score', 'max mean intercluster similarity'
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



df = pd.read_csv("../../results/mcpa/small/small_instances_exponential.csv")
test_isa(df, "../test/exponential")
print("\n\n\n\n===========================================================================================================\n\n\n\n")
df = pd.read_csv("../../results/mcpa/small/small_instances_uniform.csv")
test_isa(df, "../test/uniform")
print("\n\n\n\n===========================================================================================================\n\n\n\n")
df = pd.read_csv("../../results/mcpa/small/small_instances_inverse.csv")
test_isa(df, "../test/inverse")