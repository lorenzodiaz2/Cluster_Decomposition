"""
Analisi della prova di scala (Task 3 di workflow/specs/scale-test.md): confronto tra l'euristica e il modello completo
allo stesso tempo t^H, cioè quando l'euristica termina.

Per ogni run:
  - Γ_H: primal gap di Berthold della soluzione dell'euristica (1 se il run è fallito);
  - Γ_F(t^H): primal gap dell'ultimo incumbent del modello completo trovato entro t^H (1 se non ce n'è nessuno);
    se t^H supera il time limit del modello completo si usa il suo ultimo incumbent;
  - vittoria dell'euristica: run riuscito, t^H <= time limit, Γ_H <= 1/11 e Γ_H < Γ_F(t^H) (o nessun incumbent del
    modello completo entro t^H, anche perché il modello non si è costruito);
  - t_F_buona: primo istante in cui il modello completo ha un incumbent con Γ <= 1/11.
I Γ sono calcolati rispetto alla migliore soluzione nota del run: il migliore incumbent del modello completo o la
soluzione dell'euristica.

Tempi del modello completo: euristica e modello completo si confrontano sulla stessa istanza definitiva, quella con le
opzioni già estese se il modello completo era infeasible con le opzioni iniziali (l'euristica parte da lì senza
pagarne il tempo). Il CFL registra già gli incumbent così (solo l'ultimo tentativo); nel CPA i tempi degli incumbent
includono i tentativi infeasible, che qui si tolgono: dalla colonna global_infeasible_time (run dal 28/11/2026) o,
se manca, da infeasible_attempts.csv della cartella. I run del CPA senza nessuna delle due restano non corretti e
vengono contati nella stampa.

Per ogni configurazione (gruppo di run) stampa le mediane e la quota di vittorie: la configurazione è un regime utile
se le vittorie sono almeno la metà dei run (criterio di successo delle specifiche).

Uso (da code/):
    python analysis/scale_test.py results/mcpa/campaign_A
    python analysis/scale_test.py results/cfl/ss/campaign_A --by n_quadrants offset --sim-weighting exponential_unscaled
    python analysis/scale_test.py <cartella> --csv <file.csv>
"""
import argparse
import csv
import json
import os
import statistics as st
from collections import defaultdict

GOOD = 1 / 11
DEFAULT_TIME_LIMIT = 1800


def primal_gap(value, best):
    """Primal gap di Berthold: |v - best| / max(|v|, |best|), 0 se entrambi nulli, 1 se manca la soluzione."""
    if value is None:
        return 1.0
    den = max(abs(value), abs(best))
    return 0.0 if den == 0 else abs(value - best) / den


def load_runs(folder: str) -> list[dict]:
    with open(os.path.join(folder, "results.jsonl"), encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_incumbents(folder: str) -> dict[str, list[tuple[float, float]]]:
    """
    Incumbent per id del run con il globale. I run interrotti e ripresi possono lasciare più serie per lo stesso id
    (note.md): una serie ricomincia quando il tempo torna indietro, e si tiene l'ultima.
    """
    series: dict[str, list[list[tuple[float, float]]]] = defaultdict(list)
    path = os.path.join(folder, "incumbents.csv")
    if not os.path.isfile(path):
        return {}
    with open(path, encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            t, obj = float(rec["time"]), float(rec["obj"])
            runs = series[rec["id"]]
            if not runs or t < runs[-1][-1][0]:
                runs.append([])
            runs[-1].append((t, obj))
    return {run_id: sorted(runs[-1]) for run_id, runs in series.items()}


def load_infeasible_times(folder: str) -> dict[str, float]:
    """Tempo dei tentativi infeasible per id del run con il globale, da infeasible_attempts.csv (CPA), se c'è."""
    path = os.path.join(folder, "infeasible_attempts.csv")
    if not os.path.isfile(path):
        return {}
    out = {}
    with open(path, encoding="utf-8") as f:
        for rec in csv.DictReader(f):
            if not rec.get("error") and rec.get("infeasible_time") not in (None, ""):
                out[rec["id"]] = float(rec["infeasible_time"])
    return out


def infeasible_shift(r: dict, infeasible_times: dict[str, float]) -> float | None:
    """
    Secondi da togliere ai tempi degli incumbent del modello completo: 0 nel CFL (già esclusi), il tempo dei tentativi
    infeasible nel CPA; None se nel CPA non è noto (run non corretto).
    """
    if "variant" in r:   # CFL
        return 0.0
    if r.get("global_infeasible_time") is not None:
        return float(r["global_infeasible_time"])
    if (r.get("global_n_resolutions") or 1) <= 1:
        return 0.0       # un solo tentativo: nessun tempo da togliere
    return infeasible_times.get(r.get("global_reference") or r["id"])


def heuristic_ok(r: dict) -> bool:
    if r.get("error") or r.get("heuristic_UB") is None:
        return False
    if "heuristic_solution_ok" in r:          # CFL
        return bool(r["heuristic_solution_ok"])
    return bool(r.get("heuristic_feasible"))  # CPA


def analyse_run(r: dict, incumbents: dict, infeasible_times: dict[str, float]) -> dict | None:
    ref = r.get("global_reference") or r["id"]
    shift = infeasible_shift(r, infeasible_times)
    seq = [(t - (shift or 0.0), obj) for t, obj in incumbents.get(ref, [])]
    ok = heuristic_ok(r)
    ub_h = r["heuristic_UB"] if ok else None
    t_h = r.get("heuristic_time") if ok else None
    tl = r.get("global_time_limit") or DEFAULT_TIME_LIMIT

    candidates = [obj for _, obj in seq] + ([ub_h] if ub_h is not None else [])
    if not candidates:
        return None   # nessuna soluzione né dall'euristica né dal modello completo: il run non dice nulla sul confronto
    best = min(candidates)

    g_h = primal_gap(ub_h, best)
    slower_than_tl = ok and t_h > tl
    if ok:
        horizon = min(t_h, tl)
        before = [obj for t, obj in seq if t <= horizon]
        g_f_th = primal_gap(before[-1] if before else None, best)
    else:
        g_f_th = None
    t_f_good = next((t for t, obj in seq if primal_gap(obj, best) <= GOOD), None)
    win = bool(ok and not slower_than_tl and g_h <= GOOD and g_h < g_f_th)
    return {"ok": ok, "g_h": g_h, "g_f_th": g_f_th, "win": win, "slower_than_tl": slower_than_tl,
            "t_h": t_h, "t_f_good": t_f_good, "f_good_before_th": (ok and t_f_good is not None and t_f_good <= t_h),
            "uncorrected": shift is None}


def median_or_none(values):
    values = [v for v in values if v is not None]
    return st.median(values) if values else None


def summarise(groups: dict) -> list[dict]:
    rows = []
    for key in sorted(groups, key=lambda k: tuple(str(x) for x in k)):
        res = groups[key]
        n = len(res)
        ok = [x for x in res if x["ok"]]
        rows.append({
            "config": key,
            "n": n,
            "succ_pct": 100 * len(ok) / n,
            "good_pct": 100 * sum(x["g_h"] <= GOOD for x in res) / n,
            "gamma_h_med": median_or_none(x["g_h"] for x in res),
            "gamma_f_th_med": median_or_none(x["g_f_th"] for x in ok),
            "f_good_before_th_pct": 100 * sum(x["f_good_before_th"] for x in ok) / len(ok) if ok else None,
            "win_pct": 100 * sum(x["win"] for x in res) / n,
            "slower_than_tl_pct": 100 * sum(x["slower_than_tl"] for x in res) / n,
            "t_h_med": median_or_none(x["t_h"] for x in ok),
            "t_f_good_med": median_or_none(x["t_f_good"] for x in res),
            "useful_regime": 100 * sum(x["win"] for x in res) / n >= 50,
        })
    return rows


def fmt(v, spec):
    return "-" if v is None else format(v, spec)


def main():
    parser = argparse.ArgumentParser(description="Euristica contro modello completo allo stesso tempo t^H")
    parser.add_argument("folder", help="cartella con results.jsonl e incumbents.csv")
    parser.add_argument("--by", nargs="+", default=["n_quadrants", "offset"], help="campi che definiscono una configurazione")
    parser.add_argument("--partitioner", default="nj", help="partizione da analizzare ('all' per tutte)")
    parser.add_argument("--sim-weighting", default=None, help="tiene solo i run con questa similarità (default: tutti)")
    parser.add_argument("--csv", default=None, help="scrive la tabella anche in questo file CSV")
    args = parser.parse_args()

    runs = load_runs(args.folder)
    incumbents = load_incumbents(args.folder)
    infeasible_times = load_infeasible_times(args.folder)

    groups = defaultdict(list)
    skipped = uncorrected = 0
    for r in runs:
        if args.partitioner != "all" and r.get("partitioner") != args.partitioner:
            continue
        if args.sim_weighting and r.get("sim_weighting") != args.sim_weighting:
            continue
        res = analyse_run(r, incumbents, infeasible_times)
        if res is None:
            skipped += 1
            continue
        uncorrected += res["uncorrected"]
        groups[tuple(r.get(k) for k in args.by)].append(res)

    rows = summarise(groups)
    header = ("config", "n", "succ%", "good%", "Γ_H med", "Γ_F(tH) med", "F buona prima di tH %", "vittorie %",
              "tH>TL %", "tH med (s)", "t_F_buona med (s)", "regime utile")
    print(f"{args.folder}  (configurazione = {', '.join(args.by)}; run senza alcuna soluzione esclusi: {skipped}; "
          f"run del CPA con tentativi infeasible di durata non nota, tempi non corretti: {uncorrected})")
    print(" | ".join(header))
    for x in rows:
        print(" | ".join([str(x["config"]), str(x["n"]), fmt(x["succ_pct"], ".0f"), fmt(x["good_pct"], ".0f"),
                          fmt(x["gamma_h_med"], ".3f"), fmt(x["gamma_f_th_med"], ".3f"), fmt(x["f_good_before_th_pct"], ".0f"),
                          fmt(x["win_pct"], ".0f"), fmt(x["slower_than_tl_pct"], ".0f"), fmt(x["t_h_med"], ".1f"),
                          fmt(x["t_f_good_med"], ".1f"), "sì" if x["useful_regime"] else "no"]))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["config"])
            writer.writeheader()
            for x in rows:
                writer.writerow({**x, "config": "/".join(str(v) for v in x["config"])})


if __name__ == "__main__":
    main()
