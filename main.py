import argparse
import hashlib
import json
from dataclasses import replace

from caps.mcpa.scalability.mcpa_runner import MCPA_Runner, MCPA_Instance_Settings
from caps.mcpa.scalability.mcpa_stress_test import run_mcpa_stress_test


# ----- parametri fissi (dallo stress test) -----
BASE_GRID_SIDE = 20         # lato di un quadrante n0
N_PAIRS_BASE = 140          # |E_q*|: coppie OD per quadrante a densità rho = 1
M_STAR = 700                # M* = 5 * |E_q*|
K = 10
SEEDS = range(5)            # indici di replica: il seed vero di ogni istanza è derivato da parametri + replica (_independent_seeds)
THREADS = 12

OFFSETS = [-1, 0, 1, 2, 3, 5]
Q_ALL = [2, 4, 6, 9]
Q_REDUCED = [4, 9]          # per gli sweep su M e k delle ablazioni

OUT_A = "results/mcpa/campaign_A"
OUT_B = "results/mcpa/campaign_B"
OUT_TUNING = "results/mcpa/tuning"
OUT_STRESS = "results/mcpa/stress_test_c3"

# ----- stress test (c = 3, la capacità più stretta della campagna) -----
# con STRESS_DONE = False il Run esegue solo lo stress test e si ferma; scelto M* dai grafici, aggiornare N_PAIRS_BASE / M_STAR
# e impostare STRESS_DONE = True
STRESS_DONE = True

# ----- similarità -----
# 1) con TUNING_DONE = False il Run esegue solo la taratura, stampa l'analisi e si ferma
# 2) scelta la variante, impostare qui SIMILARITY e TUNING_DONE = True: il Run lancia le campagne vere
TUNING_DONE = True
SIMILARITY = dict(sim_weighting="exponential", sim_beta=1.0, resource_weights="linear")

TUNING_VARIANTS = [dict(sim_weighting=w, sim_beta=b, resource_weights=r)
                   for w, b in (("uniform", None), ("exponential", 0.5), ("exponential", 1.0))
                   for r in ("uniform", "linear")]
TUNING_REPLICAS = range(100, 103)   # indici di replica diversi da quelli delle campagne: istanze indipendenti
TUNING_OFFSETS = [0, 1, 2, 3]
TUNING_Q = [4, 6]

# taratura della similarità su istanze piccole (copie in scala del blocco A1): stessa densità di coppie per nodo
# (N_PAIRS_BASE / BASE_GRID_SIDE^2), quadranti di lato 10 e M = massa di un quadrante; il globale chiude all'ottimo,
# quindi si possono usare molte istanze. Tutte le varianti girano sulle stesse istanze (il seed non dipende dalla similarità)
TUNING_SMALL_GRID_SIDE = 10
TUNING_SMALL_N_PAIRS = round(N_PAIRS_BASE * TUNING_SMALL_GRID_SIDE ** 2 / BASE_GRID_SIDE ** 2)   # 38 coppie per quadrante
TUNING_SMALL_M = 5 * TUNING_SMALL_N_PAIRS                                                      # 190 agenti
TUNING_SMALL_OFFSETS = [0, 1, 2]                    # 0, 0.1, 0.2 del lato del quadrante (campagna: da -0.05 a 0.25)
TUNING_SMALL_REPLICAS = range(1000, 1042)           # 2 q x 3 offset x 42 repliche = 252 istanze
OUT_TUNING_SIMILARITY = "results/mcpa/tuning_similarity"


def _quadrant_settings(q: int, offset: int, seed: int, **kw) -> MCPA_Instance_Settings:
    grid_side = 2 * BASE_GRID_SIDE if q <= 4 else 3 * BASE_GRID_SIDE
    params = dict(grid_side=grid_side, n_quadrants=q, n_pairs_per_quadrant=N_PAIRS_BASE, resources_capacity=5,
                  offset=offset, seed=seed, max_cluster_size=M_STAR, k=K, **SIMILARITY)
    params.update(kw)
    return MCPA_Instance_Settings(**params)


def _independent_seeds(settings_list: list[MCPA_Instance_Settings]) -> list[MCPA_Instance_Settings]:
    """
    Nelle liste il campo seed contiene l'indice di replica; qui viene sostituito con un seed derivato da tutti i parametri
    dell'istanza + replica, così istanze diverse non condividono la sequenza casuale (deterministico: rilanciando si ottengono
    gli stessi seed, quindi la ripresa di una campagna funziona).
    """
    out = []
    for st in settings_list:
        key = {k: v for k, v in st.instance_dict().items() if k != "seed"}
        key["replica"] = st.seed
        seed = int(hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest(), 16) % (2 ** 31)
        out.append(replace(st, seed=seed))
    return out


def campaign_A_base() -> list[MCPA_Instance_Settings]:
    # A1: c = 5, b = 5, rho = 1  ->  6 offset x 4 q x 5 seed = 120
    return _independent_seeds([_quadrant_settings(q, o, s) for q in Q_ALL for o in OFFSETS for s in SEEDS])


def campaign_A() -> list[MCPA_Instance_Settings]:
    settings = []
    for q in Q_ALL:
        for o in OFFSETS:
            for s in SEEDS:
                settings += [_quadrant_settings(q, o, s, resources_capacity=c) for c in (4, 6)]          # A2 capacità: 240
                settings.append(_quadrant_settings(q, o, s, demand_min=1, demand_max=9))                # A3 domanda: 120
                settings += [_quadrant_settings(q, o, s, n_pairs_per_quadrant=round(rho * N_PAIRS_BASE))
                             for rho in (0.95, 1.05)]                                                     # A4 densità: 240
    return campaign_A_base() + _independent_seeds(settings)   # 720 istanze


def campaign_B() -> list[MCPA_Instance_Settings]:
    # stesse masse totali di A con q = 4 e q = 9 a rho = 1, stessa densità per area
    sizes = [(2 * BASE_GRID_SIDE, 4 * N_PAIRS_BASE), (49, 6 * N_PAIRS_BASE)]
    settings = []
    for grid_side, n_pairs in sizes:
        for s in SEEDS:
            base = dict(grid_side=grid_side, n_quadrants=1, n_pairs_per_quadrant=n_pairs, resources_capacity=5, offset=0,
                        seed=s, max_cluster_size=M_STAR, k=K, **SIMILARITY)

            # B1 lunghezza degli spostamenti x capacità omogenea/eterogenea: 3 x 2 x 2 dimensioni x 5 seed = 60
            for L in ({"L_min": 2, "L_max": 6}, {"L_min": 6, "L_max": 14}, {}):
                for delta in (0, 2):
                    settings.append(MCPA_Instance_Settings(**base, family="uniform", family_params=L, capacity_delta=delta))

            # B2 hotspot: 2 x 2 x 2 x 2 dimensioni x 5 seed = 80
            for h in (4, 8):
                for sigma in (2, 4):
                    for noise in (0.0, 0.3):
                        settings.append(MCPA_Instance_Settings(**base, family="hotspots",
                                                               family_params={"n_hotspots": h, "sigma": sigma, "noise": noise}))

            # B3 colli di bottiglia (spostamenti medi): 6 varianti x 2 dimensioni x 5 seed = 60
            medium = {"L_min": 6, "L_max": 14}
            for p in (0.1, 0.2, 0.3):
                settings.append(MCPA_Instance_Settings(**base, family="uniform", family_params=medium,
                                                       remove_rnd_nodes_flag=True, remove_percentage=p))
            for gates in (1, 3):
                settings.append(MCPA_Instance_Settings(**base, family="uniform",
                                                       family_params={**medium, "obstacles": "walls", "n_gates": gates}))
            settings.append(MCPA_Instance_Settings(**base, family="uniform", family_params={**medium, "obstacles": "blocks"}))
    return _independent_seeds(settings)   # 200 istanze


def campaign_tuning() -> list[MCPA_Instance_Settings]:
    # istanze di taratura (tipo A1): 4 offset x 2 q x 3 repliche = 24, con la prima variante di similarità
    return _independent_seeds([_quadrant_settings(q, o, r, **TUNING_VARIANTS[0])
                               for q in TUNING_Q for o in TUNING_OFFSETS for r in TUNING_REPLICAS])


def run_tuning(runner: MCPA_Runner):
    instances = campaign_tuning()
    runner.run_all(instances)                                                                   # 24 run con il globale
    for variant in TUNING_VARIANTS[1:]:
        runner.run_all([replace(st, **variant) for st in instances], run_global=False)         # 5 x 24 run solo euristica


def campaign_tuning_similarity() -> list[MCPA_Instance_Settings]:
    # istanze piccole: 3 offset x 2 q x 42 repliche = 252, con la prima variante di similarità
    settings = []
    for q in TUNING_Q:
        grid_side = (2 if q <= 4 else 3) * TUNING_SMALL_GRID_SIDE
        for o in TUNING_SMALL_OFFSETS:
            for r in TUNING_SMALL_REPLICAS:
                settings.append(_quadrant_settings(q, o, r, grid_side=grid_side, n_pairs_per_quadrant=TUNING_SMALL_N_PAIRS,
                                                   max_cluster_size=TUNING_SMALL_M, **TUNING_VARIANTS[0]))
    return _independent_seeds(settings)


def run_tuning_similarity(runner: MCPA_Runner):
    instances = campaign_tuning_similarity()
    runner.run_all(instances)                                                                   # 252 run con il globale
    for variant in TUNING_VARIANTS[1:]:
        runner.run_all([replace(st, **variant) for st in instances], run_global=False)         # 5 x 252 run solo euristica


def run_ablations_A(runner: MCPA_Runner):
    # da lanciare dopo la campagna A, nella stessa cartella (servono i run di riferimento con il globale)
    base = campaign_A_base()
    reduced = [st for st in base if st.n_quadrants in Q_REDUCED]

    runner.run_all([replace(st, partitioner="random") for st in base], run_global=False)                              # 120
    runner.run_all([replace(st, max_cluster_size=round(f * M_STAR)) for st in reduced for f in (0.5, 0.75, 1.25, 1.5)],
                   run_global=False)                                                                                   # 240
    # TODO (analisi e ISA): k qui è il valore NOMINALE. Se il modello completo è infeasible le opzioni vengono estese, quindi
    # nelle analisi va usato il numero effettivo di opzioni per coppia, n_paths_heuristic / n_pairs (vedi isa/IsaInterface/test_isa.py)
    runner.run_all([replace(st, k=k) for st in reduced for k in (5, 15)])                                              # 120, k cambia il modello: serve il globale


def block_label_A(st: MCPA_Instance_Settings) -> str:
    # nome del blocco della campagna A, solo per la stampa a video
    if st.resources_capacity != 5:
        return f"A2 c={st.resources_capacity}"
    if (st.demand_min, st.demand_max) != (5, 5):
        return f"A3 demand {st.demand_min}-{st.demand_max}"
    if st.n_pairs_per_quadrant != N_PAIRS_BASE:
        return f"A4 rho={st.n_pairs_per_quadrant / N_PAIRS_BASE:g}"
    return "A1"


def block_label_B(st: MCPA_Instance_Settings) -> str:
    # nome della sottofamiglia della campagna B, solo per la stampa a video
    size = "mass q=4" if st.grid_side == 2 * BASE_GRID_SIDE else "mass q=6"
    if st.family == "hotspots":
        return f"B2 hotspots, {size}"
    if st.remove_rnd_nodes_flag or st.family_params.get("obstacles"):
        return f"B3 obstacles, {size}"
    return f"B1 uniform, {size}"


def remove_runs_outside_A1():
    # cancella da OUT_A tutti i run delle istanze che non appartengono al blocco A1 (copia di sicurezza in OUT_A + "_backup");
    # da eseguire a campagna ferma
    import csv
    import os
    import shutil

    shutil.copytree(OUT_A, OUT_A + "_backup")   # si ferma con un errore se la copia esiste già, così non la sovrascrive
    keep = {st.instance_id for st in campaign_A_base()}

    results_file = os.path.join(OUT_A, "results.jsonl")
    with open(results_file, encoding="utf8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    removed_runs = {r["id"] for r in rows if r["instance_id"] not in keep}
    removed_instances = {r["instance_id"] for r in rows if r["instance_id"] not in keep}
    with open(results_file, "w", encoding="utf8") as fh:
        for r in rows:
            if r["id"] not in removed_runs:
                fh.write(json.dumps(r) + "\n")

    for name in ("clusters.csv", "incumbents.csv"):
        path = os.path.join(OUT_A, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames
            kept = [row for row in reader if row["id"] not in removed_runs]
        with open(path, "w", encoding="utf8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept)

    for instance_id in removed_instances:
        path = os.path.join(OUT_A, "instances", f"{instance_id}.json.gz")
        if os.path.isfile(path):
            os.remove(path)

    print(f"rimossi {len(removed_runs)} run e {len(removed_instances)} istanze; copia di sicurezza in {OUT_A}_backup")

def remove_runs_B(first: int = 101, last: int = 117):
  # cancella da OUT_B i run delle istanze di campaign_B() dalla posizione first alla posizione last (incluse, contando da 1,
  # come nella stampa "i/n" del runner): righe di results.jsonl, clusters.csv, incumbents.csv e file d'istanza.
  # Copia di sicurezza in OUT_B + "_backup"; da eseguire a campagna ferma
  import csv
  import os
  import shutil

  instances = campaign_B()
  remove = {st.instance_id for st in instances[first - 1:last]}
  print(f"istanze da cancellare: {len(remove)}  (dalla {first}: {block_label_B(instances[first - 1])}, "
        f"alla {last}: {block_label_B(instances[last - 1])})")

  shutil.copytree(OUT_B, OUT_B + "_backup")   # si ferma con un errore se la copia esiste già, così non la sovrascrive

  results_file = os.path.join(OUT_B, "results.jsonl")
  with open(results_file, encoding="utf8") as fh:
      rows = [json.loads(line) for line in fh if line.strip()]
  removed_rows = [r for r in rows if r["instance_id"] in remove]
  removed_runs = {r["id"] for r in removed_rows}
  with open(results_file, "w", encoding="utf8") as fh:
      for r in rows:
          if r["id"] not in removed_runs:
              fh.write(json.dumps(r) + "\n")

  for name in ("clusters.csv", "incumbents.csv"):
      path = os.path.join(OUT_B, name)
      if not os.path.isfile(path):
          continue
      with open(path, encoding="utf8", newline="") as fh:
          reader = csv.DictReader(fh)
          fields = reader.fieldnames
          kept = [row for row in reader if row["id"] not in removed_runs]
      with open(path, "w", encoding="utf8", newline="") as fh:
          writer = csv.DictWriter(fh, fieldnames=fields)
          writer.writeheader()
          writer.writerows(kept)

  for instance_id in remove:
      path = os.path.join(OUT_B, "instances", f"{instance_id}.json.gz")
      if os.path.isfile(path):
          os.remove(path)

  n_err = sum(1 for r in removed_rows if r.get("global_error") or r.get("error"))
  print(f"rimossi {len(removed_runs)} run ({n_err} con errore) di {len(remove)} istanze; "
        f"righe rimaste in results.jsonl: {len(rows) - len(removed_rows)}; copia di sicurezza in {OUT_B}_backup")


def expected_runs_A() -> list[MCPA_Instance_Settings]:
    # tutti i run che il codice attuale prevede nella cartella della campagna A: campagna + ablazioni
    # (le tre liste delle ablazioni devono restare uguali a quelle di run_ablations_A)
    base = campaign_A_base()
    reduced = [st for st in base if st.n_quadrants in Q_REDUCED]
    return (campaign_A()
            + [replace(st, partitioner="random") for st in base]
            + [replace(st, max_cluster_size=round(f * M_STAR)) for st in reduced for f in (0.5, 0.75, 1.25, 1.5)]
            + [replace(st, k=k) for st in reduced for k in (5, 15)])


def remove_unexpected_runs_A(dry_run: bool = True, out_dir: str = OUT_A):
    """
    Toglie dalla cartella della campagna A i run che il codice attuale non prevede (vecchi valori del blocco A4, con
    112, 126, 154 o 168 coppie per blocco): righe di results.jsonl, clusters.csv, incumbents.csv, infeasible_attempts.csv
    e i file d'istanza che non servono più a nessun run rimasto. Con dry_run=True stampa solo cosa verrebbe tolto.
    Prima di modificare fa una copia di sicurezza in out_dir + "_backup_before_clean". Da eseguire a campagna ferma.

    Sicurezza: si toglie un run solo se il suo id non è tra quelli previsti; se tra i run da togliere ce n'è uno con un
    numero di coppie per blocco usato dalla campagna attuale, la funzione si ferma senza toccare nulla.
    """
    import csv
    import os
    import shutil
    from collections import Counter

    expected = expected_runs_A()
    expected_ids = {st.run_id for st in expected}
    valid_n_pairs = {st.n_pairs_per_quadrant for st in expected}

    results_file = os.path.join(out_dir, "results.jsonl")
    with open(results_file, encoding="utf8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    to_remove = [r for r in rows if r["id"] not in expected_ids]
    kept = [r for r in rows if r["id"] in expected_ids]
    removed_ids = {r["id"] for r in to_remove}
    kept_instances = {r["instance_id"] for r in kept}
    orphan_instances = {r["instance_id"] for r in to_remove} - kept_instances

    print(f"{out_dir}: {len(rows)} run, previsti {len(expected_ids)} (presenti {len(kept)}, mancanti {len(expected_ids) - len(kept)}), "
          f"da togliere {len(to_remove)}, file d'istanza da togliere {len(orphan_instances)}")
    summary = Counter((r["n_quadrants"], r["n_pairs_per_quadrant"], r["resources_capacity"], r["k"], r["max_cluster_size"],
                       r["partitioner"]) for r in to_remove)
    for (q, n_pairs, c, k, m, part), count in sorted(summary.items()):
        print(f"  da togliere: {count:3d} run  q={q} coppie per blocco={n_pairs} c={c} k={k} M={m} part={part}")
    if to_remove:
        print(f"  avviati tra {min(r['started_at'] for r in to_remove)} e {max(r['started_at'] for r in to_remove)}")

    suspicious = [r for r in to_remove if r["n_pairs_per_quadrant"] in valid_n_pairs]
    if suspicious:
        raise SystemExit(f"FERMO: {len(suspicious)} run da togliere hanno un numero di coppie per blocco usato dalla campagna attuale "
                         f"({sorted(valid_n_pairs)}): controllare a mano prima di cancellare. Nessun file è stato modificato.")
    if dry_run or not to_remove:
        print("nessun file modificato" + (" (prova: usare clean_A per cancellare davvero)" if to_remove else ""))
        return

    shutil.copytree(out_dir, out_dir + "_backup_before_clean")   # si ferma con un errore se la copia esiste già

    with open(results_file, "w", encoding="utf8") as fh:
        for r in kept:
            fh.write(json.dumps(r) + "\n")

    for name in ("clusters.csv", "incumbents.csv", "infeasible_attempts.csv"):
        path = os.path.join(out_dir, name)
        if not os.path.isfile(path):
            continue
        with open(path, encoding="utf8", newline="") as fh:
            reader = csv.DictReader(fh)
            fields = reader.fieldnames
            all_rows = list(reader)
        kept_rows = [row for row in all_rows if row["id"] not in removed_ids]
        with open(path, "w", encoding="utf8", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fields)
            writer.writeheader()
            writer.writerows(kept_rows)
        print(f"  {name}: tolte {len(all_rows) - len(kept_rows)} righe, rimaste {len(kept_rows)}")

    n_files = 0
    for instance_id in orphan_instances:
        path = os.path.join(out_dir, "instances", f"{instance_id}.json.gz")
        if os.path.isfile(path):
            os.remove(path)
            n_files += 1
    print(f"tolti {len(to_remove)} run e {n_files} file d'istanza; rimasti {len(kept)} run in results.jsonl; "
          f"copia di sicurezza in {out_dir}_backup_before_clean")



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Campagne MCPA")
    # senza argomenti (es. pulsante Run di PyCharm) lancia A e B; rilanciando riprende dai run non ancora completati
    parser.add_argument("campaign", nargs="*", default=["A", "B"],
                        help="campagne da lanciare: tuning, tuning_similarity, A, B, ablations_A, infeasible_attempts (default: A B)")
    args = parser.parse_args()
    valid = ("tuning", "tuning_similarity", "A", "B", "ablations_A", "infeasible_attempts", "check_A", "clean_A")
    invalid = [c for c in args.campaign if c not in valid]
    if invalid:
        parser.error(f"campagne non valide: {invalid} (scegliere tra {', '.join(valid)})")

    # run della cartella della campagna A che il codice attuale non prevede: check_A li elenca soltanto, clean_A li toglie
    # (con copia di sicurezza). Poi ci si ferma
    if "check_A" in args.campaign or "clean_A" in args.campaign:
        remove_unexpected_runs_A(dry_run="clean_A" not in args.campaign)
        raise SystemExit(0)

    # tempo dei tentativi infeasible del modello completo, per le istanze di A e B (ablazioni comprese) in cui le opzioni sono
    # state estese: non modifica i risultati, scrive infeasible_attempts.csv nelle cartelle delle campagne e si ferma
    if "infeasible_attempts" in args.campaign:
        from caps.mcpa.scalability.mcpa_infeasible_attempts import measure_infeasible_attempts
        for out_dir in (OUT_A, OUT_B):
            measure_infeasible_attempts(out_dir, threads=THREADS)
        raise SystemExit(0)

    # taratura della similarità su istanze piccole: si esegue, si analizza e ci si ferma
    if "tuning_similarity" in args.campaign:
        from caps.mcpa.scalability.mcpa_tuning_analysis import analyze_tuning
        run_tuning_similarity(MCPA_Runner(OUT_TUNING_SIMILARITY, threads=THREADS))
        analyze_tuning(OUT_TUNING_SIMILARITY)
        raise SystemExit(0)

    # prima di tutto lo stress test per M*: alla fine ci si ferma per scegliere M* dai grafici
    if not STRESS_DONE:
        run_mcpa_stress_test(OUT_STRESS, resources_capacity=5, grid_side=BASE_GRID_SIDE, k=K, threads=THREADS)
        print("\nStress test concluso: scegliere M* dai grafici, aggiornare N_PAIRS_BASE e M_STAR, impostare STRESS_DONE = True "
              "in main.py e rilanciare.")
        raise SystemExit(0)

    # finché la taratura non è conclusa si esegue solo quella, poi ci si ferma per analizzarla
    if not TUNING_DONE or "tuning" in args.campaign:
        from caps.mcpa.scalability.mcpa_tuning_analysis import analyze_tuning
        run_tuning(MCPA_Runner(OUT_TUNING, threads=THREADS))
        analyze_tuning(OUT_TUNING)
        print("\nTaratura conclusa: scegliere la variante, impostare SIMILARITY e TUNING_DONE = True in main.py "
              "e rilanciare per eseguire le campagne.")
        raise SystemExit(0)

    for campaign in args.campaign:
        if campaign == "A":
            MCPA_Runner(OUT_A, threads=THREADS).run_all(campaign_A(), label_fn=block_label_A)
        elif campaign == "B":
            MCPA_Runner(OUT_B, threads=THREADS).run_all(campaign_B(), label_fn=block_label_B)
        elif campaign == "ablations_A":
            run_ablations_A(MCPA_Runner(OUT_A, threads=THREADS))












# todo single source sono apposto
# todo TB sono apposto quando uso Multi Source, quando uso Single Source sembrerebbe apposto quando modifico SSCFL_Heuristic_solver -> _compute_shipping_cost -> vedere commento
# todo multi source (TEST_BED_C) sono da trovare le soluzioni. Esistono anche TEST_BED_A e TEST_BED_C a questo link: http://wpage.unina.it/sforza/test/
