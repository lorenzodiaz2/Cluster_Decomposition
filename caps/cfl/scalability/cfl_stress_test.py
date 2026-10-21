import os
from datetime import datetime

import pandas as pd

from caps.cfl.elements.cfl_environment import CFL_Environment
from caps.cfl.solver.multi_source.mscfl_heuristic_solver import MSCFL_Heuristic_solver
from caps.cfl.solver.single_source.sscfl_heuristic_solver import SSCFL_Heuristic_Solver
from utils.scalability_utils import run_time_scalability


def run_cfl_time_scalability():
    env_maker = lambda grid_side, k, n, seed: CFL_Environment(
        grid_side=grid_side,
        max_cluster_size=0,
        n_quadrants=1,
        n_clients_per_quadrant=n * 5,
        n_facilities_per_quadrant=n,
        offset=0,
        k=k,
        seed=seed
    )

    sscfl_solver_maker = lambda env: SSCFL_Heuristic_Solver(env.G, env.elements, env.facilities)
    mscfl_solver_maker = lambda env: MSCFL_Heuristic_solver(env.G, env.elements, env.facilities)

    run_time_scalability(
        50, 10, env_maker, sscfl_solver_maker, "sscfl_scalability_results",
        n_start=10,
        time_limit=1800,
        n_seeds=5,
        tl_bad_count=3,
        consecutive_bad_stop=5,
        # opzionale ma consigliato:
        seed_mode="by_n",  # meno correlazioni tra taglie
        x_label="Number of Clients",
        logy_plot=True
    )

    print("\n\n============================================================\n\n")

    run_time_scalability(
        50, 10, env_maker, mscfl_solver_maker, "mscfl_scalability_results",
         n_start=10,
         time_limit=1800,
         n_seeds=5,
         tl_bad_count=3,
         consecutive_bad_stop=5,
         # opzionale ma consigliato:
         seed_mode="by_n",  # meno correlazioni tra taglie
         x_label="Number of Clients",
         logy_plot=True
    )



def run_sscfl_time_scalability():
    seed = 0
    n = 160
    n_runs = 10
    csv_path = "results/cfl/ss/sscfl_global_solver.csv"

    n_times_out = 0

    while True:
        rows = []
        n_consecutive_time_limit = 0
        for _ in range(n_runs):
            env = CFL_Environment(50, 0, 1, n * 5, n,120,0,10, seed=seed)
            print(datetime.now().strftime(f"%d-%m-%Y   %H:%M:%S    {env}   seed={seed}"))
            global_solver = SSCFL_Heuristic_Solver(env.G, env.elements, env.facilities)
            global_solver.solve()
            time = sum(global_solver.model_times) + sum(global_solver.resolution_times)
            rows.append({
                "n": n * 5,
                "seed": seed,
                "time": time
            })
            seed += 1
            if time >= global_solver.time_limit:
                n_consecutive_time_limit += 1

            if n_consecutive_time_limit >= 0.8 * n_runs:
                break

        if n_consecutive_time_limit >= 0.8 * n_runs:
            n_times_out += 1
        else:
            n_times_out = 0
        n += 5

        df = pd.DataFrame(rows)
        file_exists = os.path.isfile(csv_path)
        df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

        if n_times_out >= 3:
            break


def run_mscfl_time_scalability():
    seed = 0
    n = 10
    n_runs = 10
    csv_path = "results/cfl/ms/mscfl_global_solver.csv"

    n_times_out = 0

    while True:
        rows = []
        n_consecutive_time_limit = 0
        for _ in range(n_runs):
            env = CFL_Environment(50, 0, 1, n_clients_per_quadrant=n * 5, n_facilities_per_quadrant=n, offset=0, k=10, seed=seed)
            print(datetime.now().strftime(f"%d-%m-%Y   %H:%M:%S    {env}   seed={seed}"))
            global_solver = MSCFL_Heuristic_solver(env.G, env.elements, env.facilities)
            global_solver.solve()
            time = sum(global_solver.model_times) + sum(global_solver.resolution_times)
            rows.append({
                "n": n * 5,
                "seed": seed,
                "time": time
            })
            seed += 1
            if time >= global_solver.time_limit:
                n_consecutive_time_limit += 1

            if n_consecutive_time_limit >= 0.8 * n_runs:
                break

        if n_consecutive_time_limit >= 0.8 * n_runs:
            n_times_out += 1
        else:
            n_times_out = 0
        n += 5

        df = pd.DataFrame(rows)
        file_exists = os.path.isfile(csv_path)
        df.to_csv(csv_path, mode='a', index=False, header=not file_exists)

        if n_times_out >= 3:
            break



# =====================================================================================================================
# Stress test per M* (Cornuéjols, densità costante): misura il tempo del solver globale al crescere del numero di clienti
# =====================================================================================================================

import hashlib
import json
import math
import traceback

import numpy as np
from matplotlib import pyplot as plt

from caps.cfl.scalability.cfl_runner import CFL_Instance_Settings, VARIANTS


def _stress_seed(variant: str, n_clients: int, replica: int) -> int:
    key = f"cfl-stress-{variant}-{n_clients}-{replica}"
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2 ** 31)


def _stress_settings(variant: str, n_clients: int, replica: int, density: float, clients_per_facility: int, k: int,
                     capacity_ratio: float, opening_cost_scale: float) -> CFL_Instance_Settings:
    # densità costante: il lato della griglia cresce con la radice del numero di clienti
    grid_side = math.ceil(math.sqrt(n_clients / density))
    return CFL_Instance_Settings(variant=variant, grid_side=grid_side, n_quadrants=1, n_clients_per_quadrant=n_clients,
                                 n_facilities_per_quadrant=max(1, round(n_clients / clients_per_facility)), offset=0,
                                 seed=_stress_seed(variant, n_clients, replica), max_cluster_size=n_clients, k=k,
                                 capacity_ratio=capacity_ratio, opening_cost_scale=opening_cost_scale, family="uniform")


def _summarize_size(runs: pd.DataFrame, time_limit: float, n_seeds: int, p90_bad_frac: float, tl_bad_share: float) -> dict:
    times = np.minimum(runs["time"].to_numpy(dtype=float), time_limit)
    tl_count = int(runs["time_limit_reached"].astype(bool).sum())
    p90 = float(np.quantile(times, 0.9)) if len(times) else 0.0
    bad = tl_count >= math.ceil(tl_bad_share * n_seeds) or p90 >= p90_bad_frac * time_limit
    return {"runs": len(runs), "time_limit_count": tl_count, "median_time": float(np.median(times)) if len(times) else 0.0,
            "p90_time": p90, "max_time": float(times.max()) if len(times) else 0.0, "bad": bool(bad),
            "median_opening_cost_share": float(pd.to_numeric(runs["opening_cost_share"], errors="coerce").median()),
            "median_mip_gap": float(pd.to_numeric(runs["mip_gap"], errors="coerce").median())}


def _save_stress_plot(summary: pd.DataFrame, out_dir: str, variant: str):
    for logy in (False, True):
        plt.figure(figsize=(14, 10))
        for col, label in (("median_time", "median"), ("p90_time", "p90"), ("max_time", "max")):
            plt.plot(summary["n_clients"], summary[col], marker="o", markersize=8, linewidth=3, label=label)
        plt.xlabel("Number of Clients", fontsize=30)
        plt.ylabel("Time (s)", fontsize=30)
        plt.tick_params(axis="both", which="major", labelsize=24)
        if logy:
            plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"stress_{variant}{'_log' if logy else ''}.png"), dpi=200)
        plt.close()


def run_cfl_stress_test(
    variant: str,
    out_dir: str,
    *,
    opening_cost_scale: float,
    capacity_ratio: float = 3.0,
    clients_per_facility: int = 5,
    density: float = 0.07,            # clienti per nodo della griglia (quella della tesi: 175 clienti su 50 x 50)
    k: int = 10,
    n_start: int = 50,
    step: int = 25,
    n_max: int = 3000,
    n_seeds: int = 10,
    time_limit: int = 1800,
    threads: int | None = None,
    p90_bad_frac: float = 0.8,        # dimensione difficile se p90 >= 0.8 * time_limit ...
    tl_bad_share: float = 0.5,        # ... oppure se almeno metà dei seed va in time limit
    window: int = 5,                  # ci si ferma quando almeno window_bad delle ultime window dimensioni sono difficili
    window_bad: int = 4,
    verbose: bool = True,
) -> dict:
    """
    Una riga per run in stress_runs.csv (si riprende da dove ci si era fermati), riepilogo per dimensione in stress_summary.csv,
    grafici e stress_result.json con M* = ultima dimensione buona prima della prima dimensione difficile della finestra di arresto.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant '{variant}' non supportata (usare 'ss' o 'sd')")
    os.makedirs(out_dir, exist_ok=True)
    runs_file = os.path.join(out_dir, "stress_runs.csv")
    if threads is not None:
        import gurobipy as gp
        gp.setParam("Threads", threads)

    runs = pd.read_csv(runs_file) if os.path.isfile(runs_file) else pd.DataFrame()
    Solver = VARIANTS[variant][0]
    summary_rows, bad_flags = [], []
    stop_reason, m_star = f"raggiunto n_max = {n_max}", None

    n = n_start
    while n <= n_max:
        for replica in range(n_seeds):
            done = runs[runs["n_clients"] == n] if len(runs) else runs
            if len(done) and replica in set(done["replica"]):
                continue
            # la dimensione è già difficile per il numero di time limit: gli altri seed non cambierebbero l'esito
            if len(done) and done["time_limit_reached"].astype(bool).sum() >= math.ceil(tl_bad_share * n_seeds):
                break

            st = _stress_settings(variant, n, replica, density, clients_per_facility, k, capacity_ratio, opening_cost_scale)
            row = {"variant": variant, "n_clients": n, "n_facilities": st.n_facilities_per_quadrant, "grid_side": st.grid_side,
                   "replica": replica, "seed": st.seed, "capacity_ratio": capacity_ratio, "opening_cost_scale": opening_cost_scale,
                   "density": density, "k": k, "time_limit": time_limit, "threads": threads, "error": None,
                   "status": None, "time": None, "n_resolutions": None, "time_limit_reached": None, "UB": None, "LB": None,
                   "mip_gap": None, "n_open_facilities": None, "opening_cost_share": None}
            try:
                env = st.make_environment()
                solver = Solver(env.G, env.elements, env.facilities, time_limit=time_limit)
                solver.solve()
                total = sum(solver.model_times) + sum(solver.resolution_times)
                has_sol = solver.m.SolCount > 0
                ub = solver.m.getAttr("ObjVal") if has_sol else None
                open_ids = {fid for fid, var in solver.x.items() if var.X > 0.5} if has_sol else set()
                opening = sum(f.activation_cost for f in env.facilities if f.id in open_ids)
                row.update({"status": solver.status, "time": total, "n_resolutions": len(solver.resolution_times),
                            "time_limit_reached": solver.status == "TIME_LIMIT" or total >= time_limit,
                            "UB": ub, "LB": solver.m.getAttr("ObjBound"), "mip_gap": solver.m.getAttr("MIPGap") if has_sol else None,
                            "n_open_facilities": len(open_ids), "opening_cost_share": opening / ub if ub else None})
            except Exception:
                row.update({"status": "ERROR", "time": float(time_limit), "time_limit_reached": True,
                            "error": traceback.format_exc(limit=3).strip().splitlines()[-1]})
            pd.DataFrame([row]).to_csv(runs_file, mode="a", index=False, header=not os.path.isfile(runs_file))
            runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True) if len(runs) else pd.DataFrame([row])
            if verbose:
                print(f"{datetime.now():%d-%m-%Y %H:%M:%S}  {variant} n={n} replica={replica}  {row['status']}  {row['time']:.1f}s"
                      f"{'  ERROR: ' + row['error'] if row['error'] else ''}", flush=True)

        s = {"n_clients": n, **_summarize_size(runs[runs["n_clients"] == n], time_limit, n_seeds, p90_bad_frac, tl_bad_share)}
        summary_rows.append(s)
        bad_flags.append(s["bad"])
        if verbose:
            print(f"--- n={n}: mediana {s['median_time']:.1f}s  p90 {s['p90_time']:.1f}s  max {s['max_time']:.1f}s  "
                  f"time limit {s['time_limit_count']}/{s['runs']}  difficile={s['bad']}", flush=True)

        if len(bad_flags) >= window and sum(bad_flags[-window:]) >= window_bad:
            first_bad = next(r["n_clients"] for r in summary_rows[-window:] if r["bad"])
            good_before = [r["n_clients"] for r in summary_rows if r["n_clients"] < first_bad and not r["bad"]]
            m_star = good_before[-1] if good_before else None
            stop_reason = f"almeno {window_bad} delle ultime {window} dimensioni difficili (prima difficile della finestra: {first_bad})"
            break
        n += step

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "stress_summary.csv"), index=False)
    if len(summary):
        _save_stress_plot(summary, out_dir, variant)
    result = {"variant": variant, "m_star_clients": m_star, "stop_reason": stop_reason, "density": density,
              "clients_per_facility": clients_per_facility, "capacity_ratio": capacity_ratio, "opening_cost_scale": opening_cost_scale,
              "k": k, "n_seeds": n_seeds, "time_limit": time_limit, "window": window, "window_bad": window_bad,
              "p90_bad_frac": p90_bad_frac, "tl_bad_share": tl_bad_share}
    with open(os.path.join(out_dir, "stress_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        side = math.ceil(math.sqrt(m_star / density)) if m_star else None
        print(f"\nStress test {variant}: {stop_reason}  ->  M* = {m_star} clienti (lato del quadrante per la campagna: {side})")
    return {"summary": summary, **result}
