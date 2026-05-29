import pandas as pd
from isa.NN_Isa.nn import NN_Isa

df = pd.read_csv("../../results/mcpa/150_mcpa_results.csv")

feat_names = [
    'grid side', 'n agents', 'max cluster size', 'n clusters', 'similarity index',
    'global congestion absolute', 'cross congestion absolute',
    'cross congestion rate', 'cross congestion share',
    'silhouette score', 'max mean intercluster similarity'
]

# 1. Definisci la Verità (la tua condizione di business)
df['is_good'] = (df['gap'] <= 5).astype(float)

# 2. Il Trucco Matematico per la classe NN_Isa:
# Creiamo una baseline fissa che non sia MAI zero per evitare crash.
df['fake_baseline'] = 10.0

# Vogliamo ingannare la formula: 100 * (baseline - target) / baseline
# - Se l'istanza è Buona (is_good=1): fake_target = 0.
#   La formula calcolerà: 100 * (10 - 0) / 10 = 100% di miglioramento.
# - Se l'istanza è Cattiva (is_good=0): fake_target = 10.
#   La formula calcolerà: 100 * (10 - 10) / 10 = 0% di miglioramento.
df['fake_target'] = (1 - df['is_good']) * 10.0


# 3. Passiamo i nostri algoritmi fittizi alla rete
isa = NN_Isa(
    df=df,
    feat_names=feat_names,
    target_alg="fake_target",    # Algoritmo da valutare
    baseline_alg="fake_baseline" # L'asticella fissa
)

# 4. Eseguiamo il training (puoi variare le epoche per velocizzare i test)
isa.run(epochs=1000)

# 5. Visualizziamo i risultati
isa.plot()

#
# threshold = 0.05
# colors = ['b' if el <= 0.5 - threshold else ('k' if 0.5 - threshold < el <= 0.5 + threshold else 'r')  for el in y_]
#
# colors_quadratic = (y_ - 0.5) ** 2
# colors_quadratic = (colors_quadratic - colors_quadratic.min()) / (colors_quadratic.max() - colors_quadratic.min())
#
# plt.scatter(z[:, 0], z[:, 1], marker="o", c=colors, s=2)
# plt.show()
#
# plt.scatter(z[:, 0], z[:, 1], marker="o", c=y_, s=4, cmap='autumn')
# plt.show()
#
# np.unique_counts(colors)
#

#
# res.shape[0]