import os
import traceback
from datetime import datetime

import pandas as pd

from caps.cfl.scalability.cfl_runner import VARIANTS
from caps.cfl.scalability.cfl_stress_test import _stress_settings


def calibrate_opening_cost_scale(
    out_dir: str,
    *,
    variants: tuple[str, ...] = ("ss", "sd"),
    scales: tuple[float, ...] = (0.5, 0.75, 1.0, 1.5),
    capacity_ratios: tuple[float, ...] = (1.5, 3.0, 5.0),
    n_clients: int = 200,
    n_seeds: int = 5,
    density: float = 0.07,
    clients_per_facility: int = 5,
    k: int = 10,
    time_limit: int = 300,
    threads: int | None = None,
    target: tuple[float, float] = (0.30, 0.60),   # quota dei costi di apertura sul costo totale dell'ottimo globale
    verbose: bool = True,
) -> dict:
    """
    Taratura della scala dei costi di apertura del generatore di Cornuéjols (step 1).
    Istanze generate come nello stress test (densità costante, Cornuéjols, una sola zona). Una riga per run in calibration_runs.csv
    (si riprende da dove ci si era fermati). Scelta: tra le scale con quota mediana dentro target per tutti i valori di r e per tutte
    le varianti, quella con quota media più vicina al centro dell'intervallo.
    """
    os.makedirs(out_dir, exist_ok=True)
    runs_file = os.path.join(out_dir, "calibration_runs.csv")
    if threads is not None:
        import gurobipy as gp
        gp.setParam("Threads", threads)

    runs = pd.read_csv(runs_file) if os.path.isfile(runs_file) else pd.DataFrame()
    done = set(zip(runs["variant"], runs["scale"], runs["capacity_ratio"], runs["replica"])) if len(runs) else set()

    for variant in variants:
        Solver = VARIANTS[variant][0]
        for scale in scales:
            for r in capacity_ratios:
                for replica in range(n_seeds):
                    if (variant, scale, r, replica) in done:
                        continue
                    st = _stress_settings(variant, n_clients, replica, density, clients_per_facility, k, r, scale)
                    row = {"variant": variant, "scale": scale, "capacity_ratio": r, "replica": replica, "seed": st.seed,
                           "n_clients": n_clients, "n_facilities": st.n_facilities_per_quadrant, "status": None, "time": None,
                           "UB": None, "opening_cost_share": None, "open_facilities_share": None, "error": None}
                    try:
                        env = st.make_environment()
                        solver = Solver(env.G, env.elements, env.facilities, time_limit=time_limit)
                        solver.solve()
                        row["status"] = solver.status
                        row["time"] = sum(solver.model_times) + sum(solver.resolution_times)
                        if solver.m.SolCount > 0:
                            ub = solver.m.getAttr("ObjVal")
                            open_ids = {fid for fid, var in solver.x.items() if var.X > 0.5}
                            opening = sum(f.activation_cost for f in env.facilities if f.id in open_ids)
                            row.update({"UB": ub, "opening_cost_share": opening / ub if ub else None,
                                        "open_facilities_share": len(open_ids) / len(env.facilities)})
                    except Exception:
                        row["error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
                    pd.DataFrame([row]).to_csv(runs_file, mode="a", index=False, header=not os.path.isfile(runs_file))
                    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True) if len(runs) else pd.DataFrame([row])
                    if verbose:
                        print(f"{datetime.now():%d-%m-%Y %H:%M:%S}  {variant} scala={scale} r={r} replica={replica}  {row['status']}  "
                              f"aperture {row['opening_cost_share']}", flush=True)

    sel = runs[runs["variant"].isin(variants) & runs["scale"].isin(scales) & runs["capacity_ratio"].isin(capacity_ratios)]
    summary = (sel.groupby(["scale", "capacity_ratio", "variant"])
                  .agg(runs=("replica", "count"), median_opening_share=("opening_cost_share", "median"),
                       median_open_facilities=("open_facilities_share", "median"), median_time=("time", "median"))
                  .reset_index())
    summary.to_csv(os.path.join(out_dir, "calibration_summary.csv"), index=False)

    lo, hi = target
    center = (lo + hi) / 2
    candidates = []
    for scale, g in summary.groupby("scale"):
        if g["median_opening_share"].between(lo, hi).all():
            candidates.append((abs(g["median_opening_share"].mean() - center), scale))
    chosen = min(candidates)[1] if candidates else None

    if verbose:
        pd.set_option("display.width", 200)
        print("\nQuota mediana dei costi di apertura nell'ottimo (righe: scala; colonne: r, variante):")
        print(summary.pivot_table(index="scale", columns=["capacity_ratio", "variant"], values="median_opening_share").round(3).to_string())
        print("\nFrazione mediana di facility aperte:")
        print(summary.pivot_table(index="scale", columns=["capacity_ratio", "variant"], values="median_open_facilities").round(3).to_string())
        print(f"\nScale con quota dentro [{lo}, {hi}] per tutti gli r e le varianti: {sorted(s for _, s in candidates)}  ->  scelta: {chosen}")
    return {"summary": summary, "chosen_scale": chosen}
