import argparse
import hashlib
import json
import math
from dataclasses import replace

from caps.cfl.scalability.cfl_runner import CFL_Runner, CFL_Instance_Settings
from caps.cfl.scalability.cfl_stress_test import run_cfl_stress_test


THREADS = 12                               # core fisici della macchina (Gurobi rende meglio sui core fisici che su quelli logici)

# ----- parametri fissati -----
OPENING_COST_SCALE = 1.0                  # formula originale di Cornuéjols et al. (1991)
BASE_CAPACITY_RATIO = 3.0                 # r base: stress test e blocco base (come c = 5 per MCPA)
DENSITY = 0.5                             # clienti per nodo: un passo di griglia = 0.1 del quadrato unitario di Cornuéjols et al.
                                          # (costo unitario 10 x distanza), quindi la loro classe da 50 clienti sta su una griglia 10 x 10
CLIENTS_PER_FACILITY = 3                  # come le classi 25x8 e 50x16 di Cornuéjols et al. (1991)
K = 10
SEEDS = range(5)                          # indici di replica: il seed vero di ogni istanza è derivato da parametri + replica

# offset in righe/colonne della griglia, uguali per SS e MS (deciso il 16-11-2026). Si parte da -6 e non da -1 come in
# MCPA: le 10 facility candidate di un cliente arrivano a 7-9 righe di distanza, quindi con -1 un terzo dei clienti ha
# candidate in un altro blocco (blocchi del tutto separati solo da -9 in giu'). La griglia NON viene allargata, come in
# MCPA: con offset negativo i blocchi perdono righe e colonne (con -6 e n0 = 18 diventano 15 x 15).
OFFSETS = [-6, -4, -2, 0, 2, 5]
Q_ALL = [2, 4, 6, 9]
Q_REDUCED = [4, 9]                        # per gli sweep su M e k delle ablazioni

OUT_A = {"ss": "results/cfl/ss/campaign_A", "sd": "results/cfl/sd/campaign_A"}
OUT_B = {"ss": "results/cfl/ss/campaign_B", "sd": "results/cfl/sd/campaign_B"}

# ----- stress test per M* (curva completa fino al time limit; M* si sceglie poi dai grafici) -----
# a r base come per MCPA (c = 5): r = 2 e r = 5 (valori adiacenti a 3 in Cornuéjols et al.) sono nella campagna A2.
# SS e SD usano lo stesso identico protocollo (stesse dimensioni, stesso passo)
# Criterio di arresto uguale a quello di MCPA (run_mcpa_stress_test): una dimensione è in time limit quando 4 run su 10
# raggiungono il limite (i restanti si saltano), e ci si ferma dopo 3 dimensioni consecutive in time limit
STRESS_STOP = dict(tl_bad_share=0.4, p90_bad_frac=float("inf"), window=3, window_bad=3)
STRESS = {
    "ss": dict(out_dir="results/cfl/ss/stress_test_r3", n_start=50, step=25, n_max=10000),
    "sd": dict(out_dir="results/cfl/sd/stress_test_r3", n_start=50, step=25, n_max=10000),
}
STRESS_DONE = False

# ----- M* (numero di clienti per cluster), da scegliere dai grafici dello stress test -----
M_STAR = {"ss": 150, "sd": None}   # SS: scelto il 16-11-2026 dallo stress test (mediana circa 2 s a 150 clienti)

# ----- similarità -----
# scelta di progetto, uguale per tutti i problemi (stessa di MCPA): pesi esponenziali sulle opzioni con beta = 1 e pesi
# lineari sulle risorse. Nessuna taratura per i CFL: la similarità è un componente sostituibile del metodo
SIMILARITY = {"ss": dict(sim_weighting="exponential", sim_beta=1.0, resource_weights="linear"),
              "sd": dict(sim_weighting="exponential", sim_beta=1.0, resource_weights="linear")}


def quadrant_side(variant: str) -> int:
    # lato del quadrante con la stessa densità dello stress test: un quadrante contiene M* clienti (densità rho = 1)
    return math.ceil(math.sqrt(M_STAR[variant] / DENSITY))


def _n_facilities(n_clients: int) -> int:
    return max(1, round(n_clients / CLIENTS_PER_FACILITY))


def _independent_seeds(settings_list: list[CFL_Instance_Settings]) -> list[CFL_Instance_Settings]:
    """
    Nelle liste il campo seed contiene l'indice di replica; qui viene sostituito con un seed derivato da tutti i parametri
    dell'istanza + replica (deterministico: rilanciando si ottengono gli stessi seed, quindi la ripresa funziona).
    """
    out = []
    for st in settings_list:
        key = {k: v for k, v in st.instance_dict().items() if k != "seed"}
        key["replica"] = st.seed
        seed = int(hashlib.sha1(json.dumps(key, sort_keys=True).encode()).hexdigest(), 16) % (2 ** 31)
        out.append(replace(st, seed=seed))
    return out


def _quadrant_settings(variant: str, q: int, offset: int, replica: int, **kw) -> CFL_Instance_Settings:
    n0 = quadrant_side(variant)
    params = dict(variant=variant, grid_side=2 * n0 if q <= 4 else 3 * n0, n_quadrants=q,
                  n_clients_per_quadrant=M_STAR[variant], n_facilities_per_quadrant=_n_facilities(M_STAR[variant]),
                  offset=offset, seed=replica, max_cluster_size=M_STAR[variant], k=K,
                  capacity_ratio=BASE_CAPACITY_RATIO, opening_cost_scale=OPENING_COST_SCALE, **SIMILARITY[variant])
    params.update(kw)
    return CFL_Instance_Settings(**params)


def campaign_A_base(variant: str) -> list[CFL_Instance_Settings]:
    # A1: r = 3, scala dei costi di apertura 1, rho = 1  ->  6 offset x 4 q x 5 seed = 120
    return _independent_seeds([_quadrant_settings(variant, q, f, s) for q in Q_ALL for f in OFFSETS for s in SEEDS])


def campaign_A(variant: str) -> list[CFL_Instance_Settings]:
    settings = []
    for q in Q_ALL:
        for f in OFFSETS:
            for s in SEEDS:
                settings += [_quadrant_settings(variant, q, f, s, capacity_ratio=r) for r in (2.0, 5.0)]             # A2 tensione: 240
                settings += [_quadrant_settings(variant, q, f, s, opening_cost_scale=c) for c in (0.25, 0.1)]        # A3 costi di apertura: 240
                # (a scala 1 le facility aperte sono già sature e alzare i costi non cambia la soluzione: si varia solo verso il basso)
                for rho in (0.95, 1.05):                                                                            # A4 massa del blocco (come MCPA): 240
                    n_clients = round(rho * M_STAR[variant])
                    settings.append(_quadrant_settings(variant, q, f, s, n_clients_per_quadrant=n_clients,
                                                       n_facilities_per_quadrant=_n_facilities(n_clients)))
    return campaign_A_base(variant) + _independent_seeds(settings)   # 840 istanze


def campaign_B(variant: str) -> list[CFL_Instance_Settings]:
    # stesse masse totali di A con q = 4 e q = 6 a rho = 1 (come MCPA), stessa densità per area, senza quadranti
    n0 = quadrant_side(variant)
    sizes = [(2 * n0, 4 * M_STAR[variant]), (round(math.sqrt(6) * n0), 6 * M_STAR[variant])]
    settings = []
    for grid_side, n_clients in sizes:
        for s in SEEDS:
            base = dict(variant=variant, grid_side=grid_side, n_quadrants=1, n_clients_per_quadrant=n_clients,
                        n_facilities_per_quadrant=_n_facilities(n_clients), offset=0, seed=s, max_cluster_size=M_STAR[variant],
                        k=K, capacity_ratio=BASE_CAPACITY_RATIO, opening_cost_scale=OPENING_COST_SCALE, **SIMILARITY[variant])

            # B1 uniforme, r x scala dei costi di apertura: 3 x 2 x 2 dimensioni x 5 seed = 60
            for r in (2.0, 3.0, 5.0):
                for c in (1.0, 0.25):
                    settings.append(CFL_Instance_Settings(**{**base, "capacity_ratio": r, "opening_cost_scale": c}, family="uniform"))

            # B2 clienti a hotspot (facility uniformi): 2 x 2 x 2 x 2 dimensioni x 5 seed = 80
            # sigma come frazione del lato della griglia, come in MCPA (sigma = 2 e 4 su lato 40: 0.05 e 0.1 del lato);
            # più clienti possono stare sullo stesso nodo
            for h in (4, 8):
                for frac in (0.05, 0.1):
                    for noise in (0.0, 0.3):
                        sigma = round(frac * grid_side, 2)
                        settings.append(CFL_Instance_Settings(**base, family="hotspots",
                                                              family_params={"n_hotspots": h, "sigma": sigma, "noise": noise}))

            # B3 ostacoli: 6 varianti x 2 dimensioni x 5 seed = 60
            for p in (0.1, 0.2, 0.3):
                settings.append(CFL_Instance_Settings(**base, family="uniform", remove_rnd_nodes_flag=True, remove_percentage=p))
            for gates in (1, 3):
                settings.append(CFL_Instance_Settings(**base, family="uniform", family_params={"obstacles": "walls", "n_gates": gates}))
            settings.append(CFL_Instance_Settings(**base, family="uniform", family_params={"obstacles": "blocks"}))
    return _independent_seeds(settings)   # 200 istanze


def run_ablations_A(variant: str, runner: CFL_Runner):
    # da lanciare dopo la campagna A, nella stessa cartella (servono i run di riferimento con il globale)
    base = campaign_A_base(variant)
    reduced = [st for st in base if st.n_quadrants in Q_REDUCED]
    m_star = M_STAR[variant]

    runner.run_all([replace(st, partitioner="random") for st in base], run_global=False)                              # 120
    runner.run_all([replace(st, max_cluster_size=round(f * m_star)) for st in reduced for f in (0.5, 0.75, 1.25, 1.5)],
                   run_global=False)                                                                                   # 240
    # TODO (analisi e ISA): k qui è il valore NOMINALE. Se il modello completo è infeasible le opzioni vengono estese, quindi
    # nelle analisi va usato il numero effettivo di opzioni per cliente, n_options_heuristic / numero di clienti
    # (vedi isa/IsaInterface/test_isa.py)
    runner.run_all([replace(st, k=k) for st in reduced for k in (5, 15)])                                              # 120, k cambia il modello: serve il globale


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Campagne CFL")
    # senza argomenti (es. pulsante Run di PyCharm) lancia A e B per SS e SD; rilanciando riprende dai run non ancora completati
    parser.add_argument("campaign", nargs="*", default=["A", "B"], help="campagne da lanciare: A, B, ablations_A (default: A B)")
    parser.add_argument("--variant", nargs="+", default=["ss", "sd"], choices=["ss", "sd"], help="varianti (default: ss sd)")
    args = parser.parse_args()
    invalid = [c for c in args.campaign if c not in ("A", "B", "ablations_A")]
    if invalid:
        parser.error(f"campagne non valide: {invalid} (scegliere tra A, B, ablations_A)")

    # 1) stress test: con STRESS_DONE = False si eseguono solo gli stress test (prima SS, poi SD) e ci si ferma per scegliere M*
    #    dai grafici; rilanciando si riprende dai run non ancora completati
    if not STRESS_DONE:
        for variant, cfg in STRESS.items():
            run_cfl_stress_test(variant, cfg["out_dir"], opening_cost_scale=OPENING_COST_SCALE, capacity_ratio=BASE_CAPACITY_RATIO,
                                density=DENSITY, clients_per_facility=CLIENTS_PER_FACILITY, k=K,
                                n_start=cfg["n_start"], step=cfg["step"], n_max=cfg["n_max"], threads=THREADS, **STRESS_STOP)
        print("\nStress test conclusi: scegliere M*_SS e M*_SD dai grafici, impostarli in M_STAR e impostare STRESS_DONE = True.")
        raise SystemExit(0)

    missing = [v for v, m in M_STAR.items() if not m]
    if missing:
        raise SystemExit(f"Impostare M_STAR per {missing} (dai grafici dello stress test) prima di proseguire.")

    # 2) campagne
    for variant in args.variant:
        for campaign in args.campaign:
            if campaign == "A":
                CFL_Runner(OUT_A[variant], threads=THREADS).run_all(campaign_A(variant))
            elif campaign == "B":
                CFL_Runner(OUT_B[variant], threads=THREADS).run_all(campaign_B(variant))
            elif campaign == "ablations_A":
                run_ablations_A(variant, CFL_Runner(OUT_A[variant], threads=THREADS))
