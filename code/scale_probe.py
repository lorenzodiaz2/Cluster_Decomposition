"""
Sonda dei costi fissi per la prova di scala (Task 1 di workflow/specs/scale-test.md), per SS-CFL, CPA e MS-CFL.

Per ogni problema e taglia (numero di blocchi q, quadrato perfetto) genera un'istanza del blocco base della campagna A
del problema e misura, in due processi separati:
  - "heur": generazione dell'istanza, calcolo della similarità, costruzione dell'albero NJ, taglio dell'albero,
    metriche per l'ISA, con tempo e picco di memoria (RAM e GPU); non risolve sottoproblemi né repair;
  - "global": generazione dell'istanza, costruzione del modello completo e risoluzione con un time limit breve, con il
    tempo di costruzione, gli incumbent nel tempo e il picco di memoria.
Ogni misura gira in un processo a sé: se una taglia esaurisce la memoria il processo viene ucciso (su Linux non è un
errore che si può catturare), la sonda la registra come fallita e salta le taglie più grandi per quel problema e quella
parte. Il processo "heur" scrive righe parziali dopo la similarità e dopo NJ, così i tempi non si perdono se muore dopo.

Istanze: parametri del blocco base della campagna A (main_cfl.py per SS-CFL e MS-CFL, main.py per il CPA), griglia
m x m con m = ceil(sqrt(q)). D8 (specs): con offset negativo la griglia si allarga di |offset| righe tra blocchi
adiacenti, così ogni blocco resta esattamente n0 x n0 come nello stress test. A q = 9 e offset 0 l'istanza coincide
con quella della campagna A (replica 0).

Non scrive in results/: l'output va in una cartella fuori dal repository (default: ~/scale_probe).

Uso, dalla cartella code/:
    python scale_probe.py                                  # SS-CFL, poi CPA, poi MS-CFL, con le taglie di default
    python scale_probe.py --problems ss cpa --sizes 9 25   # solo alcuni problemi e taglie
    python scale_probe.py --parts heur                     # solo similarità, NJ e metriche
"""
import argparse
import json
import math
import os
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import replace
from datetime import datetime

try:
    import resource   # solo Linux/macOS: picco di memoria del processo
except ImportError:   # su Windows la memoria non viene misurata
    resource = None


CODE_DIR = os.path.dirname(os.path.abspath(__file__))
PROBLEMS = ("ss", "cpa", "sd")                      # ordine di esecuzione: SS-CFL, CPA, MS-CFL
PROBLEM_NAMES = {"ss": "SS-CFL", "cpa": "CPA", "sd": "MS-CFL"}
# taglie di default: per l'MS-CFL un blocco ha 400 clienti (M*), quindi a q = 81 sarebbero 32.400 clienti, oltre la
# memoria di matrici dense n x n e dell'istanza; si parte comunque da q = 9 come gli altri
DEFAULT_SIZES = {"ss": [9, 25, 49, 81, 121], "cpa": [9, 25, 49, 81, 121], "sd": [9, 25, 49]}
DEFAULT_OFFSET = 0
DEFAULT_REPLICA = 0
DEFAULT_GLOBAL_TIME_LIMIT = 300
DEFAULT_WORKER_TIMEOUT_H = 6.0
DEFAULT_OUT_DIR = os.path.join(os.path.expanduser("~"), "scale_probe")


# ------------------------------------------------------------------ misure e ambiente

def _peak_mb(who: str = "self") -> float | None:
    """
    Picco di memoria residente in MB (ru_maxrss è in KB su Linux). È un massimo cumulativo dall'avvio del processo: un
    valore letto dopo una fase include i picchi delle fasi precedenti.
    who="children": picco del più grande tra i processi figli già terminati (pool della generazione dell'istanza e della
    similarità). Con fork conta anche le pagine condivise con il padre, quindi NON va sommato al picco del padre.
    """
    if resource is None:
        return None
    usage = resource.getrusage(resource.RUSAGE_SELF if who == "self" else resource.RUSAGE_CHILDREN)
    return usage.ru_maxrss / 1024.0


def _gpu_peak_mb() -> dict:
    """Picco di memoria della GPU usata da torch in questo processo (NJ tiene lì le matrici n x n); vuoto senza CUDA."""
    try:
        import torch
        if not torch.cuda.is_available():
            return {}
        return {"gpu_peak_mb_allocated": torch.cuda.max_memory_allocated() / 2 ** 20,
                "gpu_peak_mb_reserved": torch.cuda.max_memory_reserved() / 2 ** 20}
    except Exception as exc:
        return {"gpu_peak_error": repr(exc)}


def _environment_info() -> dict:
    """Macchina e versioni, per sapere cosa si è misurato (in particolare se NJ ha usato la GPU)."""
    info = {"python": platform.python_version(), "cpu_count": os.cpu_count(), "host": platform.node()}
    try:
        from caps.cfl.scalability.cfl_runner import _git_version
        info["code_version"] = _git_version()   # come nei runner: commit corrente, con "-dirty" se ci sono modifiche
    except Exception as exc:
        info["code_version_error"] = repr(exc)
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["nj_device"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    except Exception as exc:   # torch assente o rotto: lo si registra senza fermare la misura
        info["torch_error"] = repr(exc)
    try:
        import gurobipy as gp
        info["gurobi"] = ".".join(str(v) for v in gp.gurobi.version())
    except Exception as exc:
        info["gurobi_error"] = repr(exc)
    return info


def _append_row(out_file: str, row: dict):
    with open(out_file, "a") as f:
        f.write(json.dumps(row) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _final_row_status(out_file: str, problem: str, part: str, q: int) -> str | None:
    """Stato ("ok" o "failed") della riga finale di (problem, part, q), oppure None se c'è solo la parziale o nessuna."""
    if not os.path.isfile(out_file):
        return None
    status = None
    with open(out_file) as f:
        for line in f:
            r = json.loads(line)
            if (r.get("problem"), r.get("part"), r.get("q")) == (problem, part, q) and r.get("status") in ("ok", "failed"):
                status = r["status"]
    return status


def _cut_time(partition_time: float | None, nj_build_time: float | None) -> float | None:
    """Tempo del taglio dell'albero e della costruzione dei cluster: partition_time meno la costruzione di NJ."""
    if partition_time is None or nj_build_time is None:
        return None
    return partition_time - nj_build_time


# ------------------------------------------------------------------ istanze

def _check_size(q: int):
    if math.isqrt(q) ** 2 != q:
        raise ValueError(f"q = {q} non è un quadrato perfetto: la griglia avrebbe blocchi vuoti e una densità diversa")


def _grid_side(q: int, n0: int, offset: int) -> int:
    """Lato della griglia m x m; con offset negativo allargato secondo D8, così ogni blocco resta n0 x n0."""
    from utils.grid_utils import blocks_per_side
    m = blocks_per_side(q)
    return m * n0 + (m - 1) * abs(offset) if offset < 0 else m * n0


def make_settings(problem: str, q: int, offset: int, replica: int):
    """Impostazioni del blocco base della campagna A del problema, con q blocchi su una griglia m x m."""
    _check_size(q)
    if problem == "cpa":
        import main
        settings = main._quadrant_settings(q, offset, replica)
        settings = replace(settings, grid_side=_grid_side(q, main.BASE_GRID_SIDE, offset))
        return main._independent_seeds([settings])[0]
    import main_cfl
    settings = main_cfl._quadrant_settings(problem, q, offset, replica)
    settings = replace(settings, grid_side=_grid_side(q, main_cfl.quadrant_side(problem), offset))
    return main_cfl._independent_seeds([settings])[0]


def _threads(problem: str) -> int:
    if problem == "cpa":
        import main
        return main.THREADS
    import main_cfl
    return main_cfl.THREADS


def _instance_columns(problem: str, env) -> dict:
    cols = {"n_entities": len(env.elements), "n_nodes": env.G.number_of_nodes()}
    if problem == "cpa":
        cols["n_agents"] = sum(len(od.agents) for od in env.elements)
        cols["n_options"] = sum(len(od.all_paths) for od in env.elements)
    else:
        cols["n_facilities"] = len(env.facilities)
        cols["n_options"] = sum(len(c.k_facilities) for c in env.elements)
    return cols


# ------------------------------------------------------------------ worker

def worker_heur(problem: str, q: int, offset: int, replica: int, out_file: str) -> dict:
    import caps.general.general_environment as general_environment

    if problem == "cpa":
        from caps.mcpa.elements.pair import OD_Pair
        sim_fn = OD_Pair.compute_similarity
    else:
        from caps.cfl.elements.client import Client
        sim_fn = Client.compute_similarity

    out = {"problem": problem, "part": "heur", "q": q, "offset": offset, **_environment_info()}
    settings = make_settings(problem, q, offset, replica)
    out["instance_id"] = settings.instance_id

    start = time.perf_counter()
    env = settings.make_environment()
    out["instance_time"] = time.perf_counter() - start
    out.update(_instance_columns(problem, env))
    out["peak_mb_after_instance"] = _peak_mb()

    # punti di misura senza toccare il codice del metodo: si avvolgono, solo in questo processo, la creazione dell'albero
    # (subito dopo la similarità) e il primo calcolo delle metriche (subito dopo NJ e il taglio)
    marks = {}
    original_tree = general_environment.TreePartition
    original_metrics = general_environment.General_Environment._compute_congestion_indexes

    def tree_with_mark(*args, **kwargs):
        marks["peak_mb_after_similarity"] = _peak_mb()
        # prima riga parziale: tempo della similarità salvato prima di NJ (se NJ esaurisce la memoria della GPU va in
        # eccezione e senza questa riga il tempo della similarità andrebbe perso)
        _append_row(out_file, {**out, "status": "partial", "stage": "similarity", "similarity_time": env.matrix_time,
                               "peak_mb_after_similarity": marks["peak_mb_after_similarity"],
                               "finished_at": datetime.now().isoformat(timespec="seconds")})
        # la costruzione dell'albero NJ avviene nel costruttore (NjTree.__init__ -> nj_torch); il taglio viene dopo, in
        # compute_clusters: separandoli si sa se il tempo va nella parte su GPU (NJ) o in quella in Python (taglio)
        t0 = time.perf_counter()
        tree = original_tree(*args, **kwargs)
        marks["nj_build_time"] = time.perf_counter() - t0
        return tree

    def metrics_with_mark(self, *args, **kwargs):
        marks["t_before_metrics"] = time.perf_counter()
        marks["peak_mb_after_nj"] = _peak_mb()
        marks["gpu_after_nj"] = _gpu_peak_mb()
        # seconda riga parziale: tempi di similarità e NJ salvati prima delle metriche, che alle taglie grandi pesano molto
        _append_row(out_file, {**out, "status": "partial", "stage": "nj", "similarity_time": self.matrix_time,
                               "nj_time": self.partition_time, "nj_build_time": marks.get("nj_build_time"),
                               "cut_time": _cut_time(self.partition_time, marks.get("nj_build_time")),
                               "n_clusters": len(self.clusters),
                               "peak_mb_after_similarity": marks.get("peak_mb_after_similarity"),
                               "peak_mb_after_nj": marks["peak_mb_after_nj"], **marks["gpu_after_nj"],
                               "finished_at": datetime.now().isoformat(timespec="seconds")})
        return original_metrics(self, *args, **kwargs)

    general_environment.TreePartition = tree_with_mark
    general_environment.General_Environment._compute_congestion_indexes = metrics_with_mark
    try:
        # come in solve_clusters dei due problemi: pesi delle risorse, poi similarità, partizione e metriche
        resource_weights = env._compute_resource_weights() if env.resource_weights != "uniform" else {}
        start = time.perf_counter()
        env.compute_clusters(sim_fn, resource_weights)
        end = time.perf_counter()
    finally:
        general_environment.TreePartition = original_tree
        general_environment.General_Environment._compute_congestion_indexes = original_metrics

    out["similarity_time"] = env.matrix_time
    out["nj_time"] = env.partition_time   # NJ + taglio dell'albero, come partition_time nei risultati delle campagne
    out["nj_build_time"] = marks.get("nj_build_time")   # solo costruzione dell'albero NJ
    out["cut_time"] = _cut_time(env.partition_time, marks.get("nj_build_time"))   # taglio dell'albero e cluster
    out["metrics_time"] = end - marks["t_before_metrics"] if "t_before_metrics" in marks else None
    out["clusters_total_time"] = end - start
    out["n_clusters"] = len(env.clusters)
    out["peak_mb_after_similarity"] = marks.get("peak_mb_after_similarity")
    out["peak_mb_after_nj"] = marks.get("peak_mb_after_nj")
    out["peak_mb_end"] = _peak_mb()
    out["peak_mb_children_max"] = _peak_mb("children")   # vedi _peak_mb: non va sommato al picco del padre
    out.update(_gpu_peak_mb())   # picco della GPU (NJ); le metriche girano su CPU, quindi coincide con quello dopo NJ
    return out


def worker_global(problem: str, q: int, offset: int, replica: int, time_limit: int, threads: int) -> dict:
    import gurobipy as gp

    gp.setParam("Threads", threads)
    out = {"problem": problem, "part": "global", "q": q, "offset": offset, "time_limit": time_limit,
           "threads": threads, **_environment_info()}
    settings = make_settings(problem, q, offset, replica)
    out["instance_id"] = settings.instance_id

    start = time.perf_counter()
    env = settings.make_environment()
    out["instance_time"] = time.perf_counter() - start
    out.update(_instance_columns(problem, env))
    out["peak_mb_after_instance"] = _peak_mb()

    if problem == "cpa":
        from caps.mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver
        solver = MCPA_Heuristic_Solver(env.G, env.elements, time_limit=time_limit)
    else:
        from caps.cfl.scalability.cfl_runner import VARIANTS
        solver = VARIANTS[problem][0](env.G, env.elements, env.facilities, time_limit=time_limit)
    # come nei runner: un errore del solver (es. GurobiError per memoria esaurita) non deve far perdere tempi e
    # dimensioni del modello già costruito
    out["solver_error"] = None
    try:
        solver.solve()
    except Exception:
        out["solver_error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]

    out["n_attempts"] = len(solver.model_times)
    out["model_times"] = list(solver.model_times)
    out["resolution_times"] = list(solver.resolution_times)
    out["model_time_last"] = solver.model_times[-1] if solver.model_times else None
    # nel CPA i tentativi infeasible (orizzonte allungato) fanno parte del costo prima del modello ammissibile
    out["model_time_total"] = sum(solver.model_times) if solver.model_times else None
    out["infeasible_attempts_time"] = (sum(solver.model_times[:-1]) + sum(solver.resolution_times[:len(solver.model_times) - 1])
                                       if solver.model_times else None)
    out["solver_status"] = solver.status
    m = solver.m

    def attr(name):
        # attributo del modello Gurobi, None se il modello manca o l'attributo non è disponibile (es. dopo un errore)
        try:
            return getattr(m, name) if m is not None else None
        except (AttributeError, gp.GurobiError):
            return None

    out["n_vars"] = attr("NumVars")
    out["n_constrs"] = attr("NumConstrs")
    out["UB"] = attr("ObjVal") if attr("SolCount") else None
    out["LB"] = attr("ObjBound")
    out["gurobi_max_mem_gb"] = attr("MaxMemUsed")   # picco di memoria secondo Gurobi
    # tempo degli incumbent come nei runner: nel CFL dall'inizio dell'ultimo tentativo (cfl_runner._incumbent_rows),
    # nel CPA dall'inizio del primo, tentativi infeasible compresi (mcpa_runner._incumbent_rows)
    if problem == "cpa":
        offset_t = sum(solver.model_times) + sum(solver.resolution_times[:-1])
    else:
        offset_t = out["model_time_last"] or 0.0
    out["incumbents"] = [[offset_t + t, obj] for t, obj in zip(solver.incumbent.times, solver.incumbent.solutions)]
    out["peak_mb_end"] = _peak_mb()
    return out


# ------------------------------------------------------------------ orchestrazione

def run_worker_subprocess(problem: str, part: str, q: int, args, out_file: str) -> dict:
    cmd = [sys.executable, os.path.join(CODE_DIR, "scale_probe.py"), "--worker", part, "--problem", problem,
           "--q", str(q), "--offset", str(args.offset), "--replica", str(args.replica),
           "--global-time-limit", str(args.global_time_limit), "--out-file", out_file]
    start = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=CODE_DIR, timeout=args.worker_timeout_h * 3600)
        code = proc.returncode
    except subprocess.TimeoutExpired:
        code = "timeout"
    wall = time.perf_counter() - start

    if code == 0:
        return {"status": "ok", "wall_time": wall}
    written = _final_row_status(out_file, problem, part, q)
    if written == "ok":
        # la misura è riuscita e il figlio ha scritto la sua riga, ma è uscito con un errore (es. in chiusura di CUDA)
        return {"status": "ok", "wall_time": wall}
    if written == "failed":
        # eccezione Python nel figlio: la riga con il traceback l'ha già scritta il figlio
        return {"status": "failed", "reason": "exception", "wall_time": wall}
    # nessuna riga finale: processo ucciso (codice negativo su Linux, -9 di solito per memoria esaurita), timeout, oppure
    # errore fuori dal blocco protetto del figlio
    if code == "timeout":
        reason = "timeout"
    elif isinstance(code, int) and code < 0:
        reason = "killed (probabile memoria esaurita)"
    else:
        reason = "error (vedi l'output del processo)"
    row = {"problem": problem, "part": part, "q": q, "status": "failed", "reason": reason, "returncode": code,
           "wall_time": wall, "finished_at": datetime.now().isoformat(timespec="seconds")}
    _append_row(out_file, row)
    return row


def _nan(v):
    return float("nan") if v is None else v


def print_summary(out_file: str):
    if not os.path.isfile(out_file):
        return
    rows = [json.loads(line) for line in open(out_file)]
    key = lambda r: (r.get("problem"), r.get("part"), r.get("q"))
    ok_keys = {key(r) for r in rows if r.get("status") == "ok"}
    final_keys = {key(r) for r in rows if r.get("status") in ("ok", "failed")}
    print("\n=== Riepilogo ===")
    for problem in PROBLEMS:
        prows = [r for r in rows if r.get("problem") == problem]
        if not prows:
            continue
        print(f"--- {PROBLEM_NAMES[problem]}")
        for r in sorted(prows, key=lambda x: (x["part"], x["q"])):
            status, head = r.get("status"), f"{r['part']:6s} q={r['q']:4d}"
            if status == "partial":
                # le righe parziali si mostrano solo se la misura non ha una riga finale riuscita
                if key(r) in ok_keys:
                    continue
                nj = f"NJ={r['nj_time']:.1f}s" if r.get("nj_time") is not None else "NJ non concluso"
                print(f"{head}  (parziale, dopo {r.get('stage')}) entità={r['n_entities']:6d}  "
                      f"similarità={r['similarity_time']:.1f}s  {nj}  GPU={_nan(r.get('gpu_peak_mb_allocated')):.0f} MB")
            elif status == "failed":
                print(f"{head}  FALLITO: {r.get('reason')}")
            elif r["part"] == "heur":
                print(f"{head}  entità={r['n_entities']:6d}  istanza={r['instance_time']:.1f}s  "
                      f"similarità={r['similarity_time']:.1f}s  NJ={r['nj_time']:.1f}s ({r.get('nj_device')}; albero "
                      f"{_nan(r.get('nj_build_time')):.1f}s, taglio {_nan(r.get('cut_time')):.1f}s)  "
                      f"metriche={_nan(r.get('metrics_time')):.1f}s  RAM dopo NJ={_nan(r.get('peak_mb_after_nj')):.0f} MB  "
                      f"RAM finale={_nan(r.get('peak_mb_end')):.0f} MB  GPU={_nan(r.get('gpu_peak_mb_allocated')):.0f} MB")
            else:
                inc = r.get("incumbents") or []
                first = f"{inc[0][0]:.1f}s" if inc else "nessuno"
                error = f"  ERRORE: {r['solver_error']}" if r.get("solver_error") else ""
                print(f"{head}  entità={r['n_entities']:6d}  costruzione={_nan(r.get('model_time_last')):.1f}s "
                      f"(totale {_nan(r.get('model_time_total')):.1f}s, tentativi infeasible "
                      f"{_nan(r.get('infeasible_attempts_time')):.1f}s)  tentativi={r.get('n_attempts')}  "
                      f"stato={r.get('solver_status')}  primo incumbent={first}  n incumbent={len(inc)}  "
                      f"RAM={_nan(r.get('peak_mb_end')):.0f} MB{error}")
        for p, part, q in sorted({key(r) for r in prows} - final_keys, key=lambda k: (k[1], k[2])):
            print(f"{part:6s} q={q:4d}  solo riga parziale (misura non conclusa)")


def main():
    parser = argparse.ArgumentParser(description="Sonda dei costi fissi per la prova di scala (SS-CFL, CPA, MS-CFL)")
    parser.add_argument("--problems", nargs="+", default=list(PROBLEMS), choices=list(PROBLEMS),
                        help="problemi da misurare, nell'ordine dato (default: ss cpa sd)")
    parser.add_argument("--sizes", type=int, nargs="+", default=None,
                        help="numeri di blocchi q (quadrati perfetti), uguali per tutti i problemi (default: per problema)")
    parser.add_argument("--parts", nargs="+", default=["heur", "global"], choices=["heur", "global"])
    parser.add_argument("--offset", type=int, default=DEFAULT_OFFSET)
    parser.add_argument("--replica", type=int, default=DEFAULT_REPLICA)
    parser.add_argument("--global-time-limit", type=int, default=DEFAULT_GLOBAL_TIME_LIMIT)
    parser.add_argument("--worker-timeout-h", type=float, default=DEFAULT_WORKER_TIMEOUT_H, help="ore massime per ogni misura")
    parser.add_argument("--out-dir", default=DEFAULT_OUT_DIR, help="cartella di output, fuori da code/results/")
    # argomenti interni, usati dai processi figli
    parser.add_argument("--worker", choices=["heur", "global"], help=argparse.SUPPRESS)
    parser.add_argument("--problem", choices=list(PROBLEMS), help=argparse.SUPPRESS)
    parser.add_argument("--q", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--out-file", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if CODE_DIR not in sys.path:   # i moduli del progetto (caps, utils, main, main_cfl) stanno in code/
        sys.path.insert(0, CODE_DIR)

    if args.worker:
        try:
            if args.worker == "heur":
                row = worker_heur(args.problem, args.q, args.offset, args.replica, args.out_file)
            else:
                row = worker_global(args.problem, args.q, args.offset, args.replica, args.global_time_limit,
                                    _threads(args.problem))
            row["status"] = "ok"
        except Exception:
            row = {"problem": args.problem, "part": args.worker, "q": args.q, "status": "failed", "reason": "exception",
                   "traceback": traceback.format_exc(limit=5)}
            traceback.print_exc()
        row["finished_at"] = datetime.now().isoformat(timespec="seconds")
        _append_row(args.out_file, row)
        sys.exit(0 if row["status"] == "ok" else 1)

    sizes = {p: sorted(args.sizes) if args.sizes else DEFAULT_SIZES[p] for p in args.problems}
    for p in args.problems:   # controllo prima di lanciare qualunque misura
        for q in sizes[p]:
            _check_size(q)
    if "results" in os.path.normpath(os.path.abspath(args.out_dir)).split(os.sep):
        raise ValueError("la sonda non scrive in una cartella results/: scegliere un'altra --out-dir")

    os.makedirs(args.out_dir, exist_ok=True)
    out_file = os.path.join(args.out_dir, f"probe_{datetime.now():%Y%m%d_%H%M%S}.jsonl")
    print(f"Output: {out_file}")

    for problem in args.problems:
        for part in args.parts:
            for q in sizes[problem]:
                print(f"{datetime.now():%d-%m-%Y %H:%M:%S}  {PROBLEM_NAMES[problem]} {part} q={q} ...", flush=True)
                result = run_worker_subprocess(problem, part, q, args, out_file)
                print(f"    {result['status']} in {result['wall_time']:.0f} s", flush=True)
                if result["status"] != "ok":
                    print(f"    {PROBLEM_NAMES[problem]} {part}: salto le taglie più grandi di q={q}", flush=True)
                    break

    print_summary(out_file)


if __name__ == "__main__":
    main()
