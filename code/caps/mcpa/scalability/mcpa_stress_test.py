import os
from datetime import datetime

import pandas as pd

from caps.mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
from caps.mcpa.elements.mcpa_environment import MCPA_Environment
from utils.scalability_utils import run_time_scalability


def run_mcpa_time_scalability():
    env_maker = lambda grid_side, k, n, seed: MCPA_Environment(
        grid_side=grid_side,
        max_cluster_size=0,
        n_quadrants=1,
        n_pairs_per_quadrant=n,
        offset=0,
        k=k,
        seed=seed
    )
    solver_maker = lambda env: MCPA_Heuristic_Solver(env.G, env.elements)
    run_time_scalability(
        20, 10, env_maker, solver_maker, "mcpa_scalability_results",
        n_start=50,
        time_limit=1800,
        n_seeds=5,
        tl_bad_count=3,
        consecutive_bad_stop=5,
        x_label="Number of OD Pairs"
    )



def run_mcpa_time_scalability_():
    seed = 0
    n = 50
    n_runs = 10
    csv_path = "results/mcpa/mcpa_global_solver.csv"

    n_times_out = 0
    while True:
        rows = []
        n_consecutive_time_limit = 0
        for _ in range(n_runs):
            env = MCPA_Environment(20, 0, 1, n, 0, 10, seed=seed)
            print(datetime.now().strftime(f"%d-%m-%Y   %H:%M:%S    {env}   seed={seed}"))
            global_solver = MCPA_Heuristic_Solver(env.G, env.elements)
            global_solver.solve()
            time = sum(global_solver.model_times) + sum(global_solver.resolution_times)
            rows.append({
                "n": n,
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

        if n_times_out >= 5:
            break



# =====================================================================================================================
# Stress test per M* (versione definitiva): tempo del solver globale al crescere del numero di coppie OD in un quadrante
# =====================================================================================================================

import hashlib
import json
import math
import traceback

import numpy as np
from matplotlib import pyplot as plt

from caps.mcpa.scalability.mcpa_runner import MCPA_Instance_Settings


def _stress_seed(n_pairs: int, replica: int, capacity: int) -> int:
    key = f"mcpa-stress-c{capacity}-{n_pairs}-{replica}"
    return int(hashlib.sha1(key.encode()).hexdigest(), 16) % (2 ** 31)


def _save_mcpa_stress_plot(summary: pd.DataFrame, out_dir: str):
    for logy in (False, True):
        plt.figure(figsize=(14, 10))
        for col, label in (("median_time", "median"), ("p90_time", "p90"), ("max_time", "max")):
            plt.plot(summary["n_pairs"], summary[col], marker="o", markersize=8, linewidth=3, label=label)
        plt.xlabel("Number of OD Pairs", fontsize=30)
        plt.ylabel("Time (s)", fontsize=30)
        plt.tick_params(axis="both", which="major", labelsize=24)
        if logy:
            plt.yscale("log")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend(fontsize=24)
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f"stress_mcpa{'_log' if logy else ''}.png"), dpi=200)
        plt.close()


def run_mcpa_stress_test(
    out_dir: str,
    *,
    resources_capacity: int = 3,
    grid_side: int = 20,              # un quadrante della campagna (n0 = 20)
    demand: int = 5,                  # agenti per coppia OD
    k: int = 10,
    n_start: int = 20,
    step: int = 5,
    n_max: int = 400,
    n_seeds: int = 10,                # run per dimensione
    tl_stop_runs: int = 4,            # una dimensione è "in time limit" quando 4 run raggiungono il limite: i restanti si saltano
    consecutive_tl_stop: int = 3,     # ci si ferma dopo 3 dimensioni consecutive in time limit
    time_limit: int = 1800,
    threads: int | None = None,
    verbose: bool = True,
) -> dict:
    """
    Una riga per run in stress_runs.csv (si riprende da dove ci si era fermati), riepilogo per dimensione in stress_summary.csv,
    grafici e stress_result.json. M* si sceglie guardando la curva (un punto ben prima del gomito).
    """
    os.makedirs(out_dir, exist_ok=True)
    runs_file = os.path.join(out_dir, "stress_runs.csv")
    if threads is not None:
        import gurobipy as gp
        gp.setParam("Threads", threads)

    runs = pd.read_csv(runs_file) if os.path.isfile(runs_file) else pd.DataFrame()
    summary_rows = []
    consecutive_tl = 0
    stop_reason = f"raggiunto n_max = {n_max}"

    n = n_start
    while n <= n_max:
        for replica in range(n_seeds):
            done = runs[runs["n_pairs"] == n] if len(runs) else runs
            if len(done) and replica in set(done["replica"]):
                continue
            if len(done) and done["time_limit_reached"].astype(bool).sum() >= tl_stop_runs:
                break   # dimensione già in time limit: gli altri run non cambierebbero la curva

            st = MCPA_Instance_Settings(grid_side=grid_side, n_quadrants=1, n_pairs_per_quadrant=n, resources_capacity=resources_capacity,
                                        offset=0, seed=_stress_seed(n, replica, resources_capacity), max_cluster_size=n * demand, k=k,
                                        demand_min=demand, demand_max=demand)
            row = {"n_pairs": n, "n_agents": n * demand, "replica": replica, "seed": st.seed, "grid_side": grid_side,
                   "resources_capacity": resources_capacity, "k": k, "time_limit": time_limit, "threads": threads,
                   "status": None, "time": None, "model_time": None, "resolution_time": None, "n_resolutions": None,
                   "time_limit_reached": None, "UB": None, "LB": None, "mip_gap": None, "n_paths": None, "error": None}
            try:
                env = st.make_environment()
                solver = MCPA_Heuristic_Solver(env.G, env.elements, time_limit=time_limit)
                solver.solve()
                total = sum(solver.model_times) + sum(solver.resolution_times)
                has_sol = solver.m.SolCount > 0
                row.update({"status": solver.status, "time": total, "model_time": sum(solver.model_times),
                            "resolution_time": sum(solver.resolution_times), "n_resolutions": len(solver.resolution_times),
                            "time_limit_reached": solver.status == "TIME_LIMIT" or total >= time_limit,
                            "UB": solver.m.getAttr("ObjVal") if has_sol else None, "LB": solver.m.getAttr("ObjBound"),
                            "mip_gap": solver.m.getAttr("MIPGap") if has_sol else None,
                            "n_paths": sum(len(od.all_paths) for od in env.elements)})
            except Exception:
                row.update({"status": "ERROR", "time": float(time_limit), "time_limit_reached": True,
                            "error": traceback.format_exc(limit=3).strip().splitlines()[-1]})
            pd.DataFrame([row]).to_csv(runs_file, mode="a", index=False, header=not os.path.isfile(runs_file))
            runs = pd.DataFrame(runs.to_dict("records") + [row])   # evita l'avviso di pandas sulla concatenazione con colonne vuote
            if verbose:
                print(f"{datetime.now():%d-%m-%Y %H:%M:%S}  n={n} replica={replica}  {row['status']}  {row['time']:.1f}s"
                      f"{'  ERROR: ' + row['error'] if row['error'] else ''}", flush=True)

        size_runs = runs[runs["n_pairs"] == n]
        times = np.minimum(size_runs["time"].to_numpy(dtype=float), time_limit)
        tl_count = int(size_runs["time_limit_reached"].astype(bool).sum())
        in_tl = tl_count >= tl_stop_runs
        summary_rows.append({"n_pairs": n, "n_agents": n * demand, "runs": len(size_runs), "time_limit_count": tl_count,
                             "median_time": float(np.median(times)), "p90_time": float(np.quantile(times, 0.9)),
                             "max_time": float(times.max()), "in_time_limit": in_tl})
        if verbose:
            s = summary_rows[-1]
            print(f"--- n={n} ({n * demand} agenti): mediana {s['median_time']:.1f}s  p90 {s['p90_time']:.1f}s  max {s['max_time']:.1f}s  "
                  f"time limit {tl_count}/{len(size_runs)}", flush=True)

        consecutive_tl = consecutive_tl + 1 if in_tl else 0
        if consecutive_tl >= consecutive_tl_stop:
            stop_reason = f"{consecutive_tl_stop} dimensioni consecutive in time limit (ultima: {n} coppie OD)"
            break
        n += step

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(out_dir, "stress_summary.csv"), index=False)
    if len(summary):
        _save_mcpa_stress_plot(summary, out_dir)
    result = {"stop_reason": stop_reason, "resources_capacity": resources_capacity, "grid_side": grid_side, "demand": demand, "k": k,
              "n_start": n_start, "step": step, "n_seeds": n_seeds, "tl_stop_runs": tl_stop_runs,
              "consecutive_tl_stop": consecutive_tl_stop, "time_limit": time_limit, "threads": threads}
    with open(os.path.join(out_dir, "stress_result.json"), "w") as f:
        json.dump(result, f, indent=2)
    if verbose:
        print(f"\nStress test MCPA (c = {resources_capacity}): {stop_reason}. Scegliere M* dai grafici in {out_dir}.")
    return {"summary": summary, **result}
