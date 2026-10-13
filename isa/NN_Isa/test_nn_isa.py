import numpy as np
import pandas as pd

from isa.NN_Isa.nn import NN_Isa


def test_nn_isa(df):
    # print(f"Starting df length: {len(df)}")
    # df = df[df["status heuristic"] == "OPTIMAL"]
    # print(f"Truncated non optimal instances df length: {len(df)}")
    #
    # df["instance"] = np.array(range(df.shape[0])).astype(str)
    #
    # feat_names = [ # aggiungere la grid side
    #     'n agents', 'max cluster size', 'mean resource capacity', 'n clusters', 'similarity index',
    #     'min cluster similarity index', 'global congestion absolute', 'cross congestion absolute', 'global congestion ratio max',
    #     'cross congestion rate', 'cross congestion share', 'silhouette score', 'max mean intercluster similarity',
    #     'normalized macro components', 'max eigengap', 'clustering distortion ratio', 'absorption rate', 'forced fragmentation'
    # ]
    #
    # nn_isa = NN_Isa(df, feat_names, alg_col='gap', epsilon=10.0, max_perf=False)
    # nn_isa.run(epochs=1000, lr=0.0001, train_test_ratio=0.5)
    # nn_isa.plot()
    # print(nn_isa.accuracy)


    print(f"Starting df length: {len(df)}")
    df = df[df["status heuristic"] == "OPTIMAL"]
    df = df.reset_index(drop=True)
    print(f"Truncated non optimal instances df length: {len(df)}")


    cols = [ # aggiungere la grid side
        'n agents', 'max cluster size', 'mean resource capacity', 'n clusters', 'similarity index',
        'min cluster similarity index', 'global congestion absolute', 'cross congestion absolute', 'global congestion ratio max',
        'cross congestion rate', 'cross congestion share', 'silhouette score', 'max mean intercluster similarity',
        'normalized macro components', 'max eigengap', 'clustering distortion ratio', 'absorption rate', 'forced fragmentation', 'gap'
    ]

    df = df[cols]
    # time result 'speedup'

    df['inv_gap'] = np.where(df['gap'] <= 10.0, 50.0, 0.0)

    feat_names = df.columns.tolist()[:-2]


    baseline_alg = "inv_gap"
    target_alg = "gap"

    isa = NN_Isa(df=df, feat_names=feat_names, target_alg=target_alg, baseline_alg=baseline_alg)

    isa.run(epochs=1000)
    isa.model(isa.X_tensor)
    isa.plot(0.01)





df = pd.read_csv("../../results/mcpa/big/big_instances_EXP_LIN.csv")
df = df.reset_index(drop=True)
test_nn_isa(df)