import pandas as pd

from isa.NN_Isa.nn import NN_Isa


def test_nn_isa(df):
    feat_names = [ # aggiungere la grid side
        'n agents', 'max cluster size', 'mean resource capacity', 'n clusters', 'similarity index',
        'min cluster similarity index', 'global congestion absolute', 'cross congestion absolute', 'global congestion ratio max',
        'cross congestion rate', 'cross congestion share',
        'silhouette score', 'max mean intercluster similarity'
    ]

    nn_isa = NN_Isa(df, feat_names, alg_col='gap', epsilon=10.0, max_perf=False)
    nn_isa.run(epochs=1000, lr=0.0001, train_test_ratio=0.5)
    nn_isa.plot()
    print(nn_isa.accuracy)

#
# df = pd.read_csv("../../results/mcpa/small_instances_uniform.csv")
# df = df.reset_index(drop=True)
# test_nn_isa(df)
#
# print("\n\n\n\n===========================================================================================================\n\n\n\n")
#
# df = pd.read_csv("../../results/mcpa/small_instances_inverse.csv")
# df = df.reset_index(drop=True)
# test_nn_isa(df)
#
# print("\n\n\n\n===========================================================================================================\n\n\n\n")

df = pd.read_csv("../../results/mcpa/small/small_instances_exponential.csv")
df = df.reset_index(drop=True)
test_nn_isa(df)