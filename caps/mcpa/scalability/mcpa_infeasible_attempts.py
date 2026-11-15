"""
Misura il tempo speso dal modello completo nei tentativi infeasible che precedono la risoluzione vera.

Quando il modello completo è infeasible con le opzioni iniziali, l'orizzonte di tutte le coppie viene esteso e il modello
viene costruito e risolto di nuovo; il global_time salvato nella campagna somma tutti i tentativi. Qui, per ogni run con il
globale in cui c'è stata almeno un'estensione (global_n_resolutions > 1), si rigenera l'istanza con le opzioni iniziali e
si ripetono SOLO i primi n - 1 tentativi (costruzione del modello, verifica che sia infeasible, estensione), senza
eseguire l'ultima risoluzione. Il tempo misurato va sottratto a global_time per avere il tempo sull'istanza definitiva.

I risultati della campagna non vengono modificati: si scrive infeasible_attempts.csv nella stessa cartella, da cui si
riprende se l'esecuzione viene interrotta. Va eseguito sulla stessa macchina e con gli stessi thread della campagna.
"""
import csv
import json
import os
import time
import traceback
from dataclasses import fields

from caps.mcpa.scalability.mcpa_runner import MCPA_Instance_Settings
from caps.mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver

OUT_NAME = "infeasible_attempts.csv"
COLUMNS = ["id", "instance_id", "n_pairs", "k", "global_status", "n_resolutions", "attempts_done", "all_infeasible",
           "attempt_statuses", "model_times", "resolution_times", "infeasible_model_time", "infeasible_resolution_time",
           "infeasible_time", "instance_time", "global_time_recorded", "global_time_corrected", "error"]


def measure_infeasible_attempts(results_dir: str, threads: int | None = None, attempt_time_limit: int = 120,
                                limit: int | None = None, verbose: bool = True) -> str:
    """
    attempt_time_limit: limite per ogni tentativo; se un modello non risulta infeasible entro questo tempo il run viene
    segnalato (all_infeasible = False) e si passa al successivo.
    limit: numero massimo di run da misurare (per le prove). Restituisce il percorso del CSV.
    """
    results_file = os.path.join(results_dir, "results.jsonl")
    out = os.path.join(results_dir, OUT_NAME)
    if not os.path.isfile(results_file):
        if verbose:
            print(f"{results_file} non esiste, salto")
        return out
    if threads is not None:
        import gurobipy as gp
        gp.setParam("Threads", threads)

    with open(results_file, encoding="utf8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    todo = [r for r in rows if r.get("global_reference") == r.get("id") and (r.get("global_n_resolutions") or 0) > 1]

    done = set()
    if os.path.isfile(out):
        with open(out, newline="", encoding="utf8") as fh:
            done = {r["id"] for r in csv.DictReader(fh)}
    todo = [r for r in todo if r["id"] not in done]
    if limit:
        todo = todo[:limit]
    if verbose:
        print(f"{results_dir}: {len(rows)} run, {len(todo)} da misurare, {len(done)} già presenti in {OUT_NAME}", flush=True)

    setting_names = [f.name for f in fields(MCPA_Instance_Settings)]
    write_header = not os.path.isfile(out)
    with open(out, "a", newline="", encoding="utf8") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        if write_header:
            writer.writeheader()
        for i, r in enumerate(todo, 1):
            n = int(r["global_n_resolutions"])
            rec = {"id": r["id"], "instance_id": r["instance_id"], "n_pairs": r.get("n_pairs"), "k": r.get("k"),
                   "global_status": r.get("global_status"), "n_resolutions": n, "global_time_recorded": r.get("global_time"),
                   "error": None}
            try:
                settings = MCPA_Instance_Settings(**{name: r[name] for name in setting_names if name in r})
                if settings.instance_id != r["instance_id"]:
                    raise RuntimeError("le impostazioni ricostruite non danno lo stesso instance_id")
                start = time.perf_counter()
                env = settings.make_environment()
                rec["instance_time"] = round(time.perf_counter() - start, 3)

                solver = MCPA_Heuristic_Solver(env.G, env.elements, time_limit=attempt_time_limit)
                statuses = []
                for _ in range(n - 1):
                    solver._set_model()
                    solver._optimize_model()
                    statuses.append(solver.status)
                    if solver.status != "INFEASIBLE" or not solver._handle_infeasibility():
                        break
                model_t, res_t = sum(solver.model_times), sum(solver.resolution_times)
                rec.update({
                    "attempts_done": len(statuses),
                    "all_infeasible": len(statuses) == n - 1 and all(s == "INFEASIBLE" for s in statuses),
                    "attempt_statuses": json.dumps(statuses),
                    "model_times": json.dumps([round(t, 3) for t in solver.model_times]),
                    "resolution_times": json.dumps([round(t, 3) for t in solver.resolution_times]),
                    "infeasible_model_time": round(model_t, 3),
                    "infeasible_resolution_time": round(res_t, 3),
                    "infeasible_time": round(model_t + res_t, 3),
                    "global_time_corrected": (round(r["global_time"] - model_t - res_t, 3)
                                              if r.get("global_time") is not None else None),
                })
            except Exception:
                rec["error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
            writer.writerow(rec)
            fh.flush()
            if verbose:
                print(f"[{i}/{len(todo)}] {r['id']}  coppie={r.get('n_pairs')}  tentativi={rec.get('attempts_done')}/{n - 1}  "
                      f"infeasible={rec.get('all_infeasible')}  modello={rec.get('infeasible_model_time')}s  "
                      f"solver={rec.get('infeasible_resolution_time')}s  globale registrato={r.get('global_time')}"
                      f"{'  ERRORE: ' + rec['error'] if rec['error'] else ''}", flush=True)
    return out
