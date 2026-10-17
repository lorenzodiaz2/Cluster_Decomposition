import gzip
import hashlib
import json
import os
import subprocess
import time
import traceback
from collections import defaultdict
from dataclasses import dataclass, asdict, field
from datetime import datetime

import pandas as pd

from caps.mcpa.elements.mcpa_environment import MCPA_Environment
from caps.mcpa.solver.mcpa_critical_resources import MCPA_Critical_Resources
from caps.mcpa.solver.mcpa_heuristic_solver import MCPA_Heuristic_Solver


# indicatori scalari calcolati dall'environment durante il clustering (quelli per cluster vanno in clusters.csv)
INDICATORS = [
    "similarity_index", "max_mean_intercluster_similarity", "silhouette_score",
    "global_congestion_index_absolute", "global_congestion_ratio_max",
    "cross_congestion_index_absolute", "cross_congestion_rate", "cross_congestion_share",
    "normalized_macro_components", "max_eigengap", "clustering_distortion_ratio",
    "interaction_density", "similarity_gini", "similarity_skewness", "mass_cv", "boundary_entities_ratio",
    "partition_entropy", "critical_resources_ratio", "congestion_gini", "cross_cluster_footprint_density",
    "average_footprint_breadth", "average_guaranteed_slack", "space_to_mass_ratio",
]

# impostazioni del metodo: non cambiano l'istanza né il modello globale, quindi più run possono condividere lo stesso globale
METHOD_FIELDS = ("max_cluster_size", "partitioner", "sim_weighting", "sim_beta", "resource_weights")

GLOBAL_COLUMNS = ("global_status", "global_n_resolutions", "global_model_time", "global_resolution_time", "global_time",
                  "global_time_limit", "global_UB", "global_LB", "global_mip_gap", "global_n_incumbents", "global_error")


@dataclass
class MCPA_Instance_Settings:
    grid_side: int
    n_quadrants: int
    n_pairs_per_quadrant: int
    resources_capacity: int
    offset: int
    seed: int
    max_cluster_size: int
    k: int = 10
    remove_rnd_nodes_flag: bool = False
    remove_percentage: float = 0.0
    demand_min: int = 5
    demand_max: int = 5
    capacity_delta: int = 0
    partitioner: str = "nj"
    sim_weighting: str = "exponential"
    sim_beta: float | None = 0.95
    resource_weights: str = "linear"
    family: str = "quadrants"
    family_params: dict = field(default_factory=dict)   # parametri specifici della famiglia (L_min, L_max, n_hotspots, sigma, noise, obstacles, ...)

    def instance_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key not in METHOD_FIELDS}

    @property
    def instance_id(self) -> str:
        # stessa istanza (generazione) -> stesso id, qualunque sia il metodo
        key = json.dumps(self.instance_dict(), sort_keys=True)
        return f"{self.family}_{hashlib.sha1(key.encode()).hexdigest()[:12]}"

    @property
    def run_id(self) -> str:
        # istanza + metodo: un id per ogni esecuzione dell'euristica (serve anche per riprendere una campagna interrotta)
        key = json.dumps(asdict(self), sort_keys=True)
        return f"{self.instance_id}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"

    def make_environment(self) -> MCPA_Environment:
        return MCPA_Environment(
            grid_side=self.grid_side,
            max_cluster_size=self.max_cluster_size,
            n_quadrants=self.n_quadrants,
            n_pairs_per_quadrant=self.n_pairs_per_quadrant,
            resources_capacity=self.resources_capacity,
            offset=self.offset,
            k=self.k,
            reproducibility_flag=True,
            remove_rnd_nodes_flag=self.remove_rnd_nodes_flag,
            remove_percentage=self.remove_percentage,
            seed=self.seed,
            sim_method=(self.sim_weighting, self.sim_beta),
            resource_weights=self.resource_weights,
            demand_min=self.demand_min,
            demand_max=self.demand_max,
            capacity_delta=self.capacity_delta,
            partitioner=self.partitioner,
            family=self.family,
            family_params=self.family_params,
        )


def _attr(model, name):
    # attributo Gurobi o None se non disponibile (es. nessuna soluzione trovata)
    try:
        return model.getAttr(name)
    except Exception:
        return None


def _json_default(o):
    # tipi numpy -> tipi Python
    return o.item() if hasattr(o, "item") else str(o)


def _read_jsonl(file_path: str) -> list[dict]:
    if not os.path.isfile(file_path):
        return []
    with open(file_path) as f:
        return [json.loads(line) for line in f if line.strip()]


def load_results(out_dir: str) -> pd.DataFrame:
    return pd.DataFrame(_read_jsonl(os.path.join(out_dir, "results.jsonl")))


def _instance_fingerprint(env: MCPA_Environment) -> str:
    ods = sorted((od.id, od.src, od.dst, len(od.agents)) for od in env.elements)
    caps = sorted((v, env.G.nodes[v]["capacity"]) for v in env.G.nodes)
    return hashlib.sha1(json.dumps([ods, caps]).encode()).hexdigest()


def _changed_horizons(env: MCPA_Environment, reference_T: dict[int, int]) -> dict[int, int]:
    # solo le coppie OD il cui orizzonte T è cambiato: da k path + T si ricostruiscono tutti i path
    return {od.id: od.T for od in env.elements if od.T != reference_T[od.id]}


def _apply_horizons(env: MCPA_Environment, horizons: dict) -> None:
    od_by_id = {od.id: od for od in env.elements}
    for od_id, T in horizons.items():
        od = od_by_id[int(od_id)]
        od.delay_shortest_paths(T)
        od.T = T


def _check_feasibility(env: MCPA_Environment) -> tuple[int, int]:
    # ritorna (agenti senza path, risorse spazio-temporali violate) della soluzione attualmente assegnata agli agenti
    occupancy = defaultdict(int)
    missing = 0
    for a in env.agents:
        if a.path is None:
            missing += 1
            continue
        for t, v in enumerate(a.path.visits):
            occupancy[v, t] += 1
    violated = sum(1 for (v, t), occ in occupancy.items() if occ > env.G.nodes[v]["capacity"])
    return missing, violated


def _git_version() -> str | None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL).strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return None


def load_instance(instance_path: str, settings: MCPA_Instance_Settings | None = None, stage: str = "global") -> tuple[MCPA_Environment, dict]:
    """
    Rigenera l'istanza e verifica che sia identica a quella salvata.
    settings: impostazioni con cui costruire l'environment (stessa istanza, metodo anche diverso); se None si usano quelle salvate.
    stage = "initial": path come generati; "global": dopo le estensioni del solver globale (quelli visti dall'euristica).
    """
    with gzip.open(instance_path, "rt") as f:
        data = json.load(f)

    saved = MCPA_Instance_Settings(**data["settings"])
    if settings is None:
        settings = saved
    elif settings.instance_dict() != saved.instance_dict():
        raise ValueError(f"Le impostazioni non corrispondono all'istanza salvata ({instance_path})")

    env = settings.make_environment()
    if _instance_fingerprint(env) != data["fingerprint"]:
        raise ValueError(f"L'istanza rigenerata non coincide con quella salvata ({instance_path}): il generatore è cambiato?")

    if stage == "global":
        _apply_horizons(env, data["T_after_global"])
    elif stage != "initial":
        raise ValueError(f"stage '{stage}' non supportato")

    return env, data


def load_run(out_dir: str, run_id: str) -> MCPA_Environment:
    """Istanza con i path visti dalla repair di un run: estensioni del globale + estensioni fatte dai cluster di quel run."""
    row = next(r for r in _read_jsonl(os.path.join(out_dir, "results.jsonl")) if r["id"] == run_id)
    settings = MCPA_Instance_Settings(**{key: row[key] for key in MCPA_Instance_Settings.__dataclass_fields__})
    env, _ = load_instance(os.path.join(out_dir, "instances", f"{row['instance_id']}.json.gz"), settings, stage="global")
    _apply_horizons(env, row["T_extended_by_clusters"] or {})
    return env


class MCPA_Runner:
    def __init__(self, out_dir: str, threads: int | None = None, verbose: bool = True):
        self.out_dir = out_dir
        self.instances_dir = os.path.join(out_dir, "instances")
        self.results_file = os.path.join(out_dir, "results.jsonl")   # una riga JSON per run: le colonne possono variare tra famiglie
        self.clusters_file = os.path.join(out_dir, "clusters.csv")
        self.incumbents_file = os.path.join(out_dir, "incumbents.csv")
        os.makedirs(self.instances_dir, exist_ok=True)

        self.verbose = verbose
        self.threads = threads
        if threads is not None:
            import gurobipy as gp
            gp.setParam("Threads", threads)   # vale per tutti i modelli creati dopo: globale, cluster e repair

        self.code_version = _git_version()

        rows = _read_jsonl(self.results_file)
        self.done_ids = {r["id"] for r in rows}
        # per ogni istanza, il primo run in cui è stato eseguito il globale (riferimento per i run "solo euristica")
        self.global_references: dict[str, dict] = {}
        for r in rows:
            if r.get("global_reference") == r["id"]:
                self.global_references.setdefault(r["instance_id"], r)
        self._incumbents = None   # caricati solo quando servono


    def run_all(self, settings_list: list[MCPA_Instance_Settings], run_global: bool = True):
        for settings in settings_list:
            self.run(settings, run_global)


    def run(self, settings: MCPA_Instance_Settings, run_global: bool = True) -> str:
        """
        run_global=True: risolve anche il modello globale (e salva l'istanza se è la prima volta).
        run_global=False: solo euristica; globale, incumbent e path estesi vengono presi dal run di riferimento della stessa istanza.
        """
        run_id, instance_id = settings.run_id, settings.instance_id
        if run_id in self.done_ids:
            if self.verbose:
                print(f"{run_id} già presente, salto")
            return run_id

        row = {"id": run_id, "instance_id": instance_id, **self._settings_columns(settings), "code_version": self.code_version,
               "threads": self.threads, "started_at": datetime.now().isoformat(timespec="seconds"),
               "global_reference": run_id if run_global else None, "global_error": None, "error": None,
               "T_extended_by_clusters": None}
        cluster_rows, incumbent_rows = [], []
        env, data_to_save = None, None

        try:
            start = time.perf_counter()
            if run_global:
                env = settings.make_environment()
                initial_T = {od.id: od.T for od in env.elements}
                row["instance_time"] = time.perf_counter() - start
                row.update(self._instance_columns(env))

                # un errore del globale (es. nessuna soluzione entro il limite) non ferma l'euristica
                global_solver = MCPA_Heuristic_Solver(env.G, env.elements)
                try:
                    global_solver.solve()
                except Exception:
                    row["global_error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
                row.update(self._global_columns(global_solver))
                incumbent_rows = self._incumbent_rows(run_id, global_solver)
                data_to_save = self._instance_data(settings, env, _changed_horizons(env, initial_T))
            else:
                reference = self.global_references.get(instance_id)
                if reference is None:
                    raise RuntimeError(f"nessun run con il globale per l'istanza {instance_id}: eseguirlo prima con run_global=True")
                env, _ = load_instance(os.path.join(self.instances_dir, f"{instance_id}.json.gz"), settings, stage="global")
                row["instance_time"] = time.perf_counter() - start
                row.update(self._instance_columns(env))
                row["n_paths_initial"] = reference.get("n_paths_initial")
                row.update({key: reference.get(key) for key in GLOBAL_COLUMNS})
                row["global_reference"] = reference["id"]
                incumbent_rows = self._reference_incumbents(reference["id"])

            T_before_clusters = {od.id: od.T for od in env.elements}
            env.solve_clusters()
            row.update(self._clustering_columns(env))
            cluster_rows = self._cluster_rows(run_id, env)

            critical_resources, repair_solver = None, None
            if all(hs.status == "OPTIMAL" for hs in env.clusters_solvers):
                critical_resources = MCPA_Critical_Resources(env.G, env.elements)
                if not critical_resources.is_initially_feasible:
                    repair_solver = MCPA_Heuristic_Solver(env.G, env.elements, critical_resources)
                    repair_solver.solve()

            row["T_extended_by_clusters"] = _changed_horizons(env, T_before_clusters)
            row.update(self._heuristic_columns(env, critical_resources, repair_solver))
            row.update(self._comparison_columns(row, incumbent_rows))

        except Exception:
            row["error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
            if self.verbose:
                traceback.print_exc()

        # il file d'istanza si scrive una sola volta (dal primo run con il globale): è uguale per tutti i metodi
        instance_path = os.path.join(self.instances_dir, f"{instance_id}.json.gz")
        if data_to_save is not None and not os.path.isfile(instance_path):
            with gzip.open(instance_path, "wt") as f:
                json.dump(data_to_save, f)

        self._append(self.clusters_file, cluster_rows)
        if run_global:
            self._append(self.incumbents_file, incumbent_rows)
            if self._incumbents is not None:
                self._incumbents[run_id] = incumbent_rows
            self.global_references.setdefault(instance_id, row)
        with open(self.results_file, "a") as f:   # per ultimo: se la riga è presente, il run è completo
            f.write(json.dumps(row, default=_json_default) + "\n")
        self.done_ids.add(run_id)

        if self.verbose:
            print(f"{datetime.now():%d-%m-%Y %H:%M:%S}  {run_id}  {'global+heur' if run_global else 'heur only'}  "
                  f"q={settings.n_quadrants} off={settings.offset} cap={settings.resources_capacity} seed={settings.seed} "
                  f"M={settings.max_cluster_size} part={settings.partitioner}  gap={row.get('gap')}  speedup={row.get('speedup')}"
                  f"{'  ERROR: ' + row['error'] if row['error'] else ''}")
        return run_id


    def _reference_incumbents(self, reference_id: str) -> list[dict]:
        if self._incumbents is None:
            self._incumbents = defaultdict(list)
            if os.path.isfile(self.incumbents_file):
                for rec in pd.read_csv(self.incumbents_file).to_dict("records"):
                    self._incumbents[rec["id"]].append(rec)
        return self._incumbents.get(reference_id, [])


    # ------------------------------------------------------------------ colonne

    @staticmethod
    def _settings_columns(settings: MCPA_Instance_Settings) -> dict:
        cols = asdict(settings)   # family_params resta anche come dizionario, per poter ricostruire le impostazioni
        cols.update({f"param_{key}": value for key, value in settings.family_params.items()})
        return cols


    @staticmethod
    def _instance_columns(env: MCPA_Environment) -> dict:
        return {
            "n_nodes": env.G.number_of_nodes(),
            "n_pairs": len(env.elements),
            "n_agents": len(env.agents),
            "mean_node_capacity": sum(env.G.nodes[v]["capacity"] for v in env.G.nodes) / env.G.number_of_nodes(),
            "n_paths_initial": sum(len(od.all_paths) for od in env.elements),
        }


    @staticmethod
    def _global_columns(solver: MCPA_Heuristic_Solver) -> dict:
        return {
            "global_status": solver.status,
            "global_n_resolutions": len(solver.resolution_times),
            "global_model_time": sum(solver.model_times),
            "global_resolution_time": sum(solver.resolution_times),
            "global_time": sum(solver.model_times) + sum(solver.resolution_times),
            "global_time_limit": solver.time_limit,
            "global_UB": _attr(solver.m, "ObjVal") if solver.m is not None else None,
            "global_LB": _attr(solver.m, "ObjBound") if solver.m is not None else None,
            "global_mip_gap": _attr(solver.m, "MIPGap") if solver.m is not None else None,
            "global_n_incumbents": len(solver.incumbent.solutions),
        }


    @staticmethod
    def _incumbent_rows(run_id: str, solver: MCPA_Heuristic_Solver) -> list[dict]:
        # i risolutori precedenti all'ultimo sono infeasible, quindi tutti gli incumbent vengono dall'ultima optimize():
        # il tempo viene riportato dall'inizio del solver globale (costruzione dei modelli inclusa), come il tempo dell'euristica
        if not solver.incumbent.solutions:
            return []
        offset = sum(solver.model_times) + sum(solver.resolution_times[:-1])
        return [{"id": run_id, "time": offset + t, "obj": obj} for t, obj in zip(solver.incumbent.times, solver.incumbent.solutions)]


    @staticmethod
    def _clustering_columns(env: MCPA_Environment) -> dict:
        cols = {
            "similarity_matrix_time": env.matrix_time,
            "partition_time": env.nj_time,
            "n_clusters": len(env.clusters),
            "n_paths_heuristic": sum(len(od.all_paths) for od in env.elements),
            "min_cluster_similarity_index": min(env.cluster_similarity_indexes) if env.cluster_similarity_indexes else None,
        }
        cols.update({name: getattr(env, name) for name in INDICATORS})
        return cols


    @staticmethod
    def _cluster_rows(run_id: str, env: MCPA_Environment) -> list[dict]:
        rows = []
        for idx, (cluster, solver) in enumerate(zip(env.clusters, env.clusters_solvers)):
            rows.append({
                "id": run_id,
                "cluster": idx,
                "n_pairs": len(cluster.elements),
                "n_agents": cluster.n_agents,
                "similarity_index": env.cluster_similarity_indexes[idx],
                "congestion_index": env.cluster_congestion_indexes[idx],
                "congestion_index_absolute": env.cluster_congestion_indexes_absolute[idx],
                "congestion_ratio_max": env.cluster_congestion_ratio_max[idx],
                "status": solver.status,
                "n_resolutions": len(solver.resolution_times),
                "model_time": sum(solver.model_times),
                "resolution_time": sum(solver.resolution_times),
                "obj": _attr(solver.m, "ObjVal"),
            })
        return rows


    @staticmethod
    def _heuristic_columns(env: MCPA_Environment, critical_resources: MCPA_Critical_Resources | None,
                           repair_solver: MCPA_Heuristic_Solver | None) -> dict:
        cluster_time = sum(sum(hs.model_times) + sum(hs.resolution_times) for hs in env.clusters_solvers)
        cols = {
            "clusters_time": cluster_time,
            "n_clusters_not_optimal": sum(1 for hs in env.clusters_solvers if hs.status != "OPTIMAL"),
            "merge_feasible": None,
            "repair_status": None,
            "repair_n_resolutions": None,
            "repair_time": None,
            "critical_resources_time": None,
            "initial_critical_resources": None,
            "final_tolerance": None,
            "released_agents": None,
            "heuristic_UB": None,
            "heuristic_LB": None,
            "heuristic_missing_paths": None,
            "heuristic_violated_resources": None,
            "heuristic_feasible": False,
        }
        time_heuristic = env.matrix_time + env.nj_time + cluster_time

        if critical_resources is not None:
            cols["merge_feasible"] = critical_resources.is_initially_feasible
            cols["initial_critical_resources"] = critical_resources.critical_resources_per_tol[0]
            cols["critical_resources_time"] = sum(critical_resources.creation_times) + sum(critical_resources.unassigning_times)
            time_heuristic += cols["critical_resources_time"]

            if repair_solver is None:
                delay = sum(a.delay for a in env.agents)
                cols["heuristic_UB"] = cols["heuristic_LB"] = delay
            else:
                cols["repair_status"] = repair_solver.status
                cols["repair_n_resolutions"] = len(repair_solver.resolution_times)
                cols["repair_time"] = sum(repair_solver.model_times) + sum(repair_solver.resolution_times)
                cols["final_tolerance"] = critical_resources.current_tol
                cols["released_agents"] = len(critical_resources.removed_items)
                time_heuristic += cols["repair_time"]

                repair_ub, repair_lb = _attr(repair_solver.m, "ObjVal"), _attr(repair_solver.m, "ObjBound")
                if repair_ub is not None:
                    removed = set(critical_resources.removed_items)
                    fixed_delay = sum(a.delay for a in env.agents if a not in removed)
                    cols["heuristic_UB"] = fixed_delay + repair_ub
                    cols["heuristic_LB"] = fixed_delay + repair_lb

            # controllo sulla soluzione effettivamente assegnata agli agenti
            missing, violated = _check_feasibility(env)
            cols["heuristic_missing_paths"] = missing
            cols["heuristic_violated_resources"] = violated
            cols["heuristic_feasible"] = (missing == 0 and violated == 0 and cols["heuristic_UB"] is not None
                                          and (repair_solver is None or repair_solver.status == "OPTIMAL"))

        cols["heuristic_time"] = time_heuristic
        return cols


    @staticmethod
    def _comparison_columns(row: dict, incumbent_rows: list[dict]) -> dict:
        ub_h, ub_g, lb_g, t_h = row.get("heuristic_UB"), row.get("global_UB"), row.get("global_LB"), row.get("heuristic_time")
        cols = {"gap": None, "gap_vs_global_LB": None, "speedup": None, "global_UB_at_heuristic_time": None, "global_time_to_heuristic_UB": None}

        # con global_UB = 0 e heuristic_UB > 0 il gap relativo non è definito: resta None (i due UB sono comunque salvati)
        if ub_h is not None and ub_g is not None and (ub_g > 0 or ub_h == 0):
            cols["gap"] = 100 * (ub_h - ub_g) / ub_g if ub_g > 0 else 0.0
        if ub_h is not None and lb_g is not None and lb_g > 0:
            cols["gap_vs_global_LB"] = 100 * (ub_h - lb_g) / lb_g
        if t_h and row.get("global_time") is not None:
            cols["speedup"] = row["global_time"] / t_h

        if t_h is not None:
            before = [inc["obj"] for inc in incumbent_rows if inc["time"] <= t_h]
            cols["global_UB_at_heuristic_time"] = before[-1] if before else None
        if ub_h is not None:
            reached = [inc["time"] for inc in incumbent_rows if inc["obj"] <= ub_h + 1e-9]
            cols["global_time_to_heuristic_UB"] = reached[0] if reached else None
        return cols


    # ------------------------------------------------------------------ scrittura

    @staticmethod
    def _instance_data(settings: MCPA_Instance_Settings, env: MCPA_Environment, T_after_global: dict[int, int]) -> dict:
        return {
            "settings": asdict(settings),
            "fingerprint": _instance_fingerprint(env),
            "pairs": [[od.id, list(od.src), list(od.dst), len(od.agents)] for od in env.elements],
            "removed_nodes": [list(v) for v in sorted(set((i, j) for i in range(env.grid_side) for j in range(env.grid_side)) - set(env.G.nodes))],
            "T_after_global": T_after_global,
        }


    @staticmethod
    def _append(file_path: str, rows: list[dict]):
        if not rows:
            return
        pd.DataFrame(rows).to_csv(file_path, mode="a", index=False, header=not os.path.isfile(file_path))
