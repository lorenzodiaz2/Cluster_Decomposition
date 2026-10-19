import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, rankdata, wilcoxon

from caps.mcpa.scalability.mcpa_runner import load_results


def _label(row) -> str:
    beta = "" if row["sim_weighting"] == "uniform" else f"(beta={row['sim_beta']})"
    return f"opzioni {row['sim_weighting']}{beta} / risorse {row['resource_weights']}"


def analyze_tuning(out_dir: str) -> pd.DataFrame:
    """
    Confronta le varianti di similarità sulle stesse istanze di taratura.
    Tutte le varianti di un'istanza usano lo stesso riferimento globale, quindi l'ordinamento si fa direttamente
    sull'UB dell'euristica; un run fallito (soluzione non ammissibile o errore) è considerato il peggiore.
    """
    df = load_results(out_dir)
    df["variant"] = df.apply(_label, axis=1)
    df["ub_for_rank"] = np.where(df["heuristic_feasible"].fillna(False).astype(bool) & df["error"].isna(),
                                 df["heuristic_UB"], np.inf)

    ub = df.pivot_table(index="instance_id", columns="variant", values="ub_for_rank", aggfunc="first")
    ub = ub.dropna()                                     # solo istanze con tutte le varianti eseguite
    variants = list(ub.columns)
    ranks = pd.DataFrame(np.apply_along_axis(rankdata, 1, ub.values), index=ub.index, columns=variants)

    table = []
    for v in variants:
        runs = df[(df["variant"] == v) & df["instance_id"].isin(ub.index)]
        table.append({
            "variante": v,
            "rango medio": ranks[v].mean(),
            "gap mediano %": runs.loc[np.isfinite(runs["ub_for_rank"]), "gap"].median(),
            "gap medio %": runs.loc[np.isfinite(runs["ub_for_rank"]), "gap"].mean(),
            "falliti": int((~np.isfinite(runs["ub_for_rank"])).sum()),
            "tempo euristica mediano s": runs["heuristic_time"].median(),
            "agenti rilasciati mediani": runs["released_agents"].median(),
        })
    table = pd.DataFrame(table).sort_values("rango medio").reset_index(drop=True)

    pd.set_option("display.width", 250)
    print(f"\nTaratura della similarità: {len(ub)} istanze, {len(variants)} varianti (rango 1 = migliore)\n")
    print(table.round(3).to_string(index=False))

    if len(variants) >= 3 and len(ub) >= 2:
        stat, p = friedmanchisquare(*[ub[v].values for v in variants])
        print(f"\nTest di Friedman (le varianti sono equivalenti?): chi2 = {stat:.2f}, p = {p:.4f}")

    best = table.loc[0, "variante"]
    print(f"\nConfronti appaiati (Wilcoxon) della variante migliore contro le altre, sulle istanze riuscite per entrambe:")
    for v in variants:
        if v == best:
            continue
        both = ub[np.isfinite(ub[best]) & np.isfinite(ub[v])]
        diff = both[v] - both[best]
        if len(both) < 5 or (diff == 0).all():
            print(f"  {v}: non testabile (istanze utili: {len(both)}, differenze non nulle: {int((diff != 0).sum())})")
            continue
        _, p = wilcoxon(both[best], both[v])
        print(f"  {v}: differenza mediana di UB = {diff.median():+.1f}, p = {p:.4f}  ({len(both)} istanze)")

    print("\nRegola di scelta concordata: variante con rango medio migliore; se la differenza con le successive non è "
          "significativa, scegliere la più semplice (uniforme prima di esponenziale, w = 1 prima dei pesi lineari).")
    return table
