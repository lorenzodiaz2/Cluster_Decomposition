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

from caps.cfl.elements.cfl_environment import CFL_Environment
from caps.cfl.solver.cfl_heuristic_solver import total_solution_cost
from caps.cfl.solver.single_source.sscfl_heuristic_solver import SSCFL_Heuristic_Solver
from caps.cfl.solver.single_source.sscfl_critical_resources import SSCFL_Critical_Resources
from caps.cfl.solver.multi_source.mscfl_heuristic_solver import MSCFL_Heuristic_solver
from caps.cfl.solver.multi_source.mscfl_critical_resources import MSCFL_Critical_Resources


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
                  "global_time_limit", "global_UB", "global_LB", "global_mip_gap", "global_n_incumbents", "global_error",
                  "global_n_open_facilities", "global_opening_cost_share")

# variante -> (solver, risorse critiche)
VARIANTS = {"ss": (SSCFL_Heuristic_Solver, SSCFL_Critical_Resources), "sd": (MSCFL_Heuristic_solver, MSCFL_Critical_Resources)}

EPS = 1e-6


@dataclass
class CFL_Instance_Settings:
    variant: str                    # "ss" (single source) o "sd" (domanda divisibile): fa parte dell'istanza, cambia il modello globale
    grid_side: int
    n_quadrants: int
    n_clients_per_quadrant: int
    n_facilities_per_quadrant: int
    offset: int
    seed: int
    max_cluster_size: int
    k: int = 10
    generator: str = "cornuejols"
    capacity_ratio: float = 3.0
    opening_cost_scale: float = 1.0
    demand_range: tuple = (5, 35)
    capacity_range: tuple = (10, 160)
    resources_capacity: int | None = None     # solo per generator = "legacy"
    remove_rnd_nodes_flag: bool = False
    remove_percentage: float = 0.0
    partitioner: str = "nj"
    sim_weighting: str = "exponential"
    sim_beta: float | None = 1.0
    resource_weights: str = "linear"
    family: str = "quadrants"
    family_params: dict = field(default_factory=dict)   # parametri della famiglia (n_hotspots, sigma, noise, obstacles, ...)

    def __post_init__(self):
        if self.variant not in VARIANTS:
            raise ValueError(f"variant '{self.variant}' non supportata (usare 'ss' o 'sd')")
        # le tuple diventano liste quando passano da JSON: le normalizzo per avere id stabili
        self.demand_range = tuple(self.demand_range)
        self.capacity_range = tuple(self.capacity_range)

    def instance_dict(self) -> dict:
        return {key: value for key, value in asdict(self).items() if key not in METHOD_FIELDS}

    @property
    def instance_id(self) -> str:
        # stessa istanza (generazione + variante) -> stesso id, qualunque sia il metodo
        key = json.dumps(self.instance_dict(), sort_keys=True)
        return f"{self.variant}_{self.family}_{hashlib.sha1(key.encode()).hexdigest()[:12]}"

    @property
    def run_id(self) -> str:
        # istanza + metodo: un id per ogni esecuzione dell'euristica (serve anche per riprendere una campagna interrotta)
        key = json.dumps(asdict(self), sort_keys=True)
        return f"{self.instance_id}_{hashlib.sha1(key.encode()).hexdigest()[:8]}"

    def make_environment(self) -> CFL_Environment:
        return CFL_Environment(
            grid_side=self.grid_side,
            max_cluster_size=self.max_cluster_size,
            n_quadrants=self.n_quadrants,
            n_clients_per_quadrant=self.n_clients_per_quadrant,
            n_facilities_per_quadrant=self.n_facilities_per_quadrant,
            resources_capacities=self.resources_capacity,
            offset=self.offset,
            k=self.k,
            reproducibility_flag=True,
            remove_rnd_nodes_flag=self.remove_rnd_nodes_flag,
            remove_percentage=self.remove_percentage,
            seed=self.seed,
            sim_method=(self.sim_weighting, self.sim_beta),
            resource_weights=self.resource_weights,
            partitioner=self.partitioner,
            family=self.family,
            family_params=self.family_params,
            generator=self.generator,
            capacity_ratio=self.capacity_ratio,
            opening_cost_scale=self.opening_cost_scale,
            demand_range=self.demand_range,
            capacity_range=self.capacity_range,
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


def _git_version() -> str | None:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL).strip()
        return commit + ("-dirty" if dirty else "")
    except Exception:
        return None


def _instance_fingerprint(env: CFL_Environment) -> str:
    clients = sorted((c.id, list(c.position), c.demand) for c in env.elements)
    facilities = sorted((f.id, list(f.position), f.capacity, f.activation_cost) for f in env.facilities)
    nodes = sorted(list(v) for v in env.G.nodes)
    return hashlib.sha1(json.dumps([clients, facilities, nodes]).encode()).hexdigest()


def _changed_k(env: CFL_Environment, reference_k: dict[int, int]) -> dict[int, int]:
    # solo i clienti la cui lista di facility candidate è stata allungata: dalle facility ordinate per distanza + k si ricostruisce tutto
    return {c.id: len(c.k_facilities) for c in env.elements if len(c.k_facilities) != reference_k[c.id]}


def _apply_k(env: CFL_Environment, k_by_client: dict, refresh_option_weights: bool = False) -> None:
    client_by_id = {c.id: c for c in env.elements}
    for cid, k in k_by_client.items():
        c = client_by_id[int(cid)]
        while len(c.k_facilities) < k:
            c.add_facility()
    if refresh_option_weights:
        _refresh_option_weights(env, k_by_client.keys())


def _refresh_option_weights(env: CFL_Environment, client_ids) -> None:
    # dopo un'estensione delle facility candidate i pesi delle opzioni vanno ricalcolati prima del clustering:
    # da loro dipendono similarità, pesi delle risorse e indicatori di congestione
    ids = {int(i) for i in client_ids}
    weighting, beta = env.sim_method
    for c in env.elements:
        if c.id in ids:
            c.build_option_weights(weighting, beta)


def _check_feasibility(env: CFL_Environment, variant: str) -> dict:
    # controllo sulla soluzione effettivamente assegnata a clienti e facility
    fac_by_id = {f.id: f for f in env.facilities}
    over = sum(1 for f in env.facilities if sum(f.shipment_by_client.values()) > f.capacity + EPS)
    unserved = sum(1 for c in env.elements if abs(sum(c.shipment_by_facility.values()) - c.demand) > EPS)
    from_closed = sum(1 for c in env.elements for fid, q in c.shipment_by_facility.items() if q > EPS and not fac_by_id[fid].is_open)
    negative = sum(1 for c in env.elements for q in c.shipment_by_facility.values() if q < -EPS)
    mismatch = sum(1 for c in env.elements for fid, q in c.shipment_by_facility.items()
                   if abs(fac_by_id[fid].shipment_by_client.get(c.id, 0) - q) > EPS)
    multi = sum(1 for c in env.elements if sum(1 for q in c.shipment_by_facility.values() if q > EPS) > 1) if variant == "ss" else 0
    feasible = over == 0 and unserved == 0 and from_closed == 0 and negative == 0 and mismatch == 0 and multi == 0
    return {"heuristic_overloaded_facilities": over, "heuristic_unserved_clients": unserved, "heuristic_shipments_from_closed": from_closed,
            "heuristic_negative_shipments": negative, "heuristic_inconsistent_shipments": mismatch, "heuristic_multi_source_clients": multi,
            "heuristic_solution_ok": feasible}


def load_instance(instance_path: str, settings: CFL_Instance_Settings | None = None, stage: str = "global") -> tuple[CFL_Environment, dict]:
    """
    Rigenera l'istanza e verifica che sia identica a quella salvata.
    settings: impostazioni con cui costruire l'environment (stessa istanza, metodo anche diverso); se None si usano quelle salvate.
    stage = "initial": facility candidate come generate; "global": dopo le estensioni del solver globale (quelle viste dall'euristica).
    """
    with gzip.open(instance_path, "rt") as f:
        data = json.load(f)

    saved = CFL_Instance_Settings(**data["settings"])
    if settings is None:
        settings = saved
    elif settings.instance_dict() != saved.instance_dict():
        raise ValueError(f"Le impostazioni non corrispondono all'istanza salvata ({instance_path})")

    env = settings.make_environment()
    if _instance_fingerprint(env) != data["fingerprint"]:
        raise ValueError(f"L'istanza rigenerata non coincide con quella salvata ({instance_path}): il generatore è cambiato?")

    if stage == "global":
        _apply_k(env, data["k_after_global"], refresh_option_weights=True)
    elif stage != "initial":
        raise ValueError(f"stage '{stage}' non supportato")

    return env, data


def load_run(out_dir: str, run_id: str) -> CFL_Environment:
    """Istanza con le facility candidate viste dalla repair di un run: estensioni del globale + estensioni fatte dai cluster di quel run."""
    row = next(r for r in _read_jsonl(os.path.join(out_dir, "results.jsonl")) if r["id"] == run_id)
    settings = CFL_Instance_Settings(**{key: row[key] for key in CFL_Instance_Settings.__dataclass_fields__})
    env, _ = load_instance(os.path.join(out_dir, "instances", f"{row['instance_id']}.json.gz"), settings, stage="global")
    _apply_k(env, row["k_extended_by_clusters"] or {})
    return env


class CFL_Runner:
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


    def run_all(self, settings_list: list[CFL_Instance_Settings], run_global: bool = True):
        for settings in settings_list:
            self.run(settings, run_global)


    def run(self, settings: CFL_Instance_Settings, run_global: bool = True) -> str:
        """
        run_global=True: risolve anche il modello globale (e salva l'istanza se è la prima volta).
        run_global=False: solo euristica; globale, incumbent e facility candidate estese vengono presi dal run di riferimento.
        """
        run_id, instance_id = settings.run_id, settings.instance_id
        if run_id in self.done_ids:
            if self.verbose:
                print(f"{run_id} già presente, salto")
            return run_id

        Solver, Critical = VARIANTS[settings.variant]
        row = {"id": run_id, "instance_id": instance_id, **self._settings_columns(settings), "code_version": self.code_version,
               "threads": self.threads, "started_at": datetime.now().isoformat(timespec="seconds"),
               "global_reference": run_id if run_global else None, "global_error": None, "error": None,
               "k_extended_by_clusters": None}
        cluster_rows, incumbent_rows = [], []
        env, data_to_save = None, None

        try:
            start = time.perf_counter()
            if run_global:
                env = settings.make_environment()
                initial_k = {c.id: len(c.k_facilities) for c in env.elements}
                row["instance_time"] = time.perf_counter() - start
                row.update(self._instance_columns(env))

                # un errore del globale (es. nessuna soluzione entro il limite) non ferma l'euristica
                global_solver = Solver(env.G, env.elements, env.facilities)
                try:
                    global_solver.solve()
                except Exception:
                    row["global_error"] = traceback.format_exc(limit=3).strip().splitlines()[-1]
                row.update(self._global_columns(env, global_solver))
                incumbent_rows = self._incumbent_rows(run_id, global_solver)
                k_after_global = _changed_k(env, initial_k)
                _refresh_option_weights(env, k_after_global)
                data_to_save = self._instance_data(settings, env, k_after_global)
            else:
                reference = self.global_references.get(instance_id)
                if reference is None:
                    raise RuntimeError(f"nessun run con il globale per l'istanza {instance_id}: eseguirlo prima con run_global=True")
                env, _ = load_instance(os.path.join(self.instances_dir, f"{instance_id}.json.gz"), settings, stage="global")
                row["instance_time"] = time.perf_counter() - start
                row.update(self._instance_columns(env))
                row["n_options_initial"] = reference.get("n_options_initial")
                row.update({key: reference.get(key) for key in GLOBAL_COLUMNS})
                row["global_reference"] = reference["id"]
                incumbent_rows = self._reference_incumbents(reference["id"])

            k_before_clusters = {c.id: len(c.k_facilities) for c in env.elements}
            env.solve_clusters(lambda e, cluster: Solver(e.G, cluster.elements, cluster.facilities))
            row.update(self._clustering_columns(env))
            cluster_rows = self._cluster_rows(run_id, env)

            critical_resources, repair_solver = None, None
            if all(hs.status == "OPTIMAL" for hs in env.clusters_solvers):
                critical_resources = Critical(env.elements, env.facilities)
                if not critical_resources.is_initially_feasible:
                    repair_solver = Solver(env.G, env.elements, env.facilities, critical_resources)
                    repair_solver.solve()

            row["k_extended_by_clusters"] = _changed_k(env, k_before_clusters)
            row.update(self._heuristic_columns(env, settings.variant, critical_resources, repair_solver))
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
                  f"{settings.variant} {settings.family} q={settings.n_quadrants} off={settings.offset} r={settings.capacity_ratio} "
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
    def _settings_columns(settings: CFL_Instance_Settings) -> dict:
        cols = asdict(settings)   # family_params resta anche come dizionario, per poter ricostruire le impostazioni
        cols.update({f"param_{key}": value for key, value in settings.family_params.items()})
        return cols


    @staticmethod
    def _instance_columns(env: CFL_Environment) -> dict:
        total_demand = sum(c.demand for c in env.elements)
        total_capacity = sum(f.capacity for f in env.facilities)
        return {
            "n_nodes": env.G.number_of_nodes(),
            "n_clients": len(env.elements),
            "n_facilities": len(env.facilities),
            "total_demand": total_demand,
            "total_capacity": total_capacity,
            "actual_capacity_ratio": total_capacity / total_demand if total_demand else None,
            "mean_opening_cost": sum(f.activation_cost for f in env.facilities) / len(env.facilities),
            "n_options_initial": sum(len(c.k_facilities) for c in env.elements),
        }


    @staticmethod
    def _global_columns(env: CFL_Environment, solver) -> dict:
        ub = _attr(solver.m, "ObjVal") if solver.m is not None else None
        open_ids = []
        if solver.m is not None and solver.x is not None and (_attr(solver.m, "SolCount") or 0) > 0:
            open_ids = [fid for fid, var in solver.x.items() if var.X > 0.5]
        opening = sum(f.activation_cost for f in env.facilities if f.id in set(open_ids))
        return {
            "global_status": solver.status,
            "global_n_resolutions": len(solver.resolution_times),
            "global_model_time": sum(solver.model_times),
            "global_resolution_time": sum(solver.resolution_times),
            "global_time": sum(solver.model_times) + sum(solver.resolution_times),
            "global_time_limit": solver.time_limit,
            "global_UB": ub,
            "global_LB": _attr(solver.m, "ObjBound") if solver.m is not None else None,
            "global_mip_gap": _attr(solver.m, "MIPGap") if solver.m is not None else None,
            "global_n_incumbents": len(solver.incumbent.solutions),
            "global_n_open_facilities": len(open_ids),
            "global_opening_cost_share": opening / ub if ub else None,
        }


    @staticmethod
    def _incumbent_rows(run_id: str, solver) -> list[dict]:
        # i risolutori precedenti all'ultimo sono infeasible, quindi tutti gli incumbent vengono dall'ultima optimize():
        # il tempo viene riportato dall'inizio del solver globale (costruzione dei modelli inclusa), come il tempo dell'euristica
        if not solver.incumbent.solutions:
            return []
        offset = sum(solver.model_times) + sum(solver.resolution_times[:-1])
        return [{"id": run_id, "time": offset + t, "obj": obj} for t, obj in zip(solver.incumbent.times, solver.incumbent.solutions)]


    @staticmethod
    def _clustering_columns(env: CFL_Environment) -> dict:
        cols = {
            "similarity_matrix_time": env.matrix_time,
            "partition_time": env.partition_time,
            "n_clusters": len(env.clusters),
            "n_options_heuristic": sum(len(c.k_facilities) for c in env.elements),
            "min_cluster_similarity_index": min(env.cluster_similarity_indexes) if env.cluster_similarity_indexes else None,
        }
        cols.update({name: getattr(env, name) for name in INDICATORS})
        return cols


    @staticmethod
    def _cluster_rows(run_id: str, env: CFL_Environment) -> list[dict]:
        rows = []
        for idx, (cluster, solver) in enumerate(zip(env.clusters, env.clusters_solvers)):
            rows.append({
                "id": run_id,
                "cluster": idx,
                "n_clients": len(cluster.elements),
                "n_facilities": len(cluster.facilities),
                "demand": sum(c.demand for c in cluster.elements),
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
    def _heuristic_columns(env: CFL_Environment, variant: str, critical_resources, repair_solver) -> dict:
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
            "released_clients": None,
            "released_quantity": None,
            "heuristic_UB": None,
            "heuristic_LB": None,
            "heuristic_n_open_facilities": None,
            "heuristic_feasible": False,
        }
        time_heuristic = env.matrix_time + env.partition_time + cluster_time

        if critical_resources is not None:
            cols["merge_feasible"] = critical_resources.is_initially_feasible
            cols["initial_critical_resources"] = critical_resources.critical_resources_per_tol[0]
            cols["critical_resources_time"] = sum(critical_resources.creation_times) + sum(critical_resources.release_times)
            time_heuristic += cols["critical_resources_time"]

            if repair_solver is not None:
                cols["repair_status"] = repair_solver.status
                cols["repair_n_resolutions"] = len(repair_solver.resolution_times)
                cols["repair_time"] = sum(repair_solver.model_times) + sum(repair_solver.resolution_times)
                cols["final_tolerance"] = critical_resources.current_tol
                cols["released_clients"] = len(critical_resources.released_units)
                cols["released_quantity"] = sum(critical_resources.removed_units_by_client.values())
                time_heuristic += cols["repair_time"]
                repair_lb = _attr(repair_solver.m, "ObjBound")
                if repair_lb is not None:
                    cols["heuristic_LB"] = repair_solver.fixed_cost_before + repair_lb

            # costo e controllo calcolati sulla soluzione effettivamente assegnata (aperture + spedizioni)
            check = _check_feasibility(env, variant)
            cols.update(check)
            repair_ok = repair_solver is None or repair_solver.status == "OPTIMAL"
            cols["heuristic_feasible"] = check["heuristic_solution_ok"] and repair_ok
            if cols["heuristic_feasible"]:
                cols["heuristic_UB"] = total_solution_cost(env.elements, env.facilities)
                if repair_solver is None:
                    cols["heuristic_LB"] = cols["heuristic_UB"]
            cols["heuristic_n_open_facilities"] = sum(1 for f in env.facilities if f.is_open)

        cols["heuristic_time"] = time_heuristic
        return cols


    @staticmethod
    def _comparison_columns(row: dict, incumbent_rows: list[dict]) -> dict:
        ub_h, ub_g, lb_g, t_h = row.get("heuristic_UB"), row.get("global_UB"), row.get("global_LB"), row.get("heuristic_time")
        cols = {"gap": None, "gap_vs_global_LB": None, "speedup": None, "global_UB_at_heuristic_time": None, "global_time_to_heuristic_UB": None}

        if ub_h is not None and ub_g is not None and ub_g > 0:
            cols["gap"] = 100 * (ub_h - ub_g) / ub_g
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
    def _instance_data(settings: CFL_Instance_Settings, env: CFL_Environment, k_after_global: dict[int, int]) -> dict:
        return {
            "settings": asdict(settings),
            "fingerprint": _instance_fingerprint(env),
            "clients": [[c.id, list(c.position), c.demand] for c in env.elements],
            "facilities": [[f.id, list(f.position), f.capacity, f.activation_cost] for f in env.facilities],
            "removed_nodes": [list(v) for v in sorted(set((i, j) for i in range(env.grid_side) for j in range(env.grid_side)) - set(env.G.nodes))],
            "k_after_global": k_after_global,
        }


    @staticmethod
    def _append(file_path: str, rows: list[dict]):
        if not rows:
            return
        pd.DataFrame(rows).to_csv(file_path, mode="a", index=False, header=not os.path.isfile(file_path))
