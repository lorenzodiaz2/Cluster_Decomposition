import pandas as pd
import numpy as np
from isa import ISA

df = pd.read_csv("../../results/mcpa/small_instances.csv")
df["instance"] = np.array(range(df.shape[0])).astype(str)

# Definisci la label: 1 = buono (gap ≤ 5), 0 = cattivo (gap > 5)
df['is_good'] = (df['gap'] <= 10).astype(float)

# Crea due colonne algoritmo:
# - 'good_label': performance alta quando is_good=1, bassa quando is_good=0
# - 'bad_label': performance bassa quando is_good=1, alta quando is_good=0
# In pratica, rendi le performance opposte in modo che ISA veda una differenza netta.

df['good_label'] = df['is_good'] * 10   # se è buono, performance=10; se cattivo, performance=0
df['bad_label'] = (1 - df['is_good']) * 10  # se è cattivo, performance=10; se buono, performance=0

feat_names = [
    'grid side', 'n agents', 'max cluster size', 'mean resource capacity', 'n clusters', 'similarity index',
    'min cluster similarity index', 'global congestion absolute', 'cross congestion absolute', 'global congestion ratio max',
    'cross congestion rate', 'cross congestion share',
    'silhouette score', 'max mean intercluster similarity'
]

instance_col_name = 'instance'
alg_names = ['good_label', 'bad_label']  # due algoritmi fittizi che separano le classi

custom_options = {
    'perf': {
        'max_perf': True,   # massimizziamo la performance (10 è meglio di 0)
        'abs_perf': True,
        'epsilon': 0.1      # soglia stretta per distinguere 0 da 10
    },
    'parallel': {'n_cores': 4},
    'sifted': {'rho': 0.3, 'k': 10}
}

isa = ISA(df, instance_col_name, feat_names, alg_names, custom_options)
isa.run(verbose=True)
print("Accuracy:", isa.isa.model.pythia.accuracy)
print("Summary:", isa.isa.model.pythia.summary)

isa.save_plots("../test/img")