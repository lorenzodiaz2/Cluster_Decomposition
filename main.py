import argparse
from dataclasses import replace

from caps.mcpa.scalability.mcpa_runner import MCPA_Runner, MCPA_Instance_Settings


# ----- parametri fissi (dallo stress test) -----
BASE_GRID_SIDE = 20         # lato di un quadrante n0
N_PAIRS_BASE = 150          # |E_q*|: coppie OD per quadrante a densità rho = 1
M_STAR = 750                # M* = 5 * |E_q*|
K = 10
SEEDS = range(5)            # stessi seed per tutte le configurazioni: confronti appaiati tra configurazioni
THREADS = 12

OFFSETS = [-1, 0, 1, 2, 3, 5]
Q_ALL = [2, 4, 6, 9]
Q_REDUCED = [4, 9]          # per gli sweep su M e k delle ablazioni

OUT_A = "results/mcpa/campaign_A"
OUT_B = "results/mcpa/campaign_B"


def _quadrant_settings(q: int, offset: int, seed: int, **kw) -> MCPA_Instance_Settings:
    grid_side = 2 * BASE_GRID_SIDE if q <= 4 else 3 * BASE_GRID_SIDE
    params = dict(grid_side=grid_side, n_quadrants=q, n_pairs_per_quadrant=N_PAIRS_BASE, resources_capacity=5,
                  offset=offset, seed=seed, max_cluster_size=M_STAR, k=K)
    params.update(kw)
    return MCPA_Instance_Settings(**params)


def campaign_A_base() -> list[MCPA_Instance_Settings]:
    # A1: c = 5, b = 5, rho = 1  ->  6 offset x 4 q x 5 seed = 120
    return [_quadrant_settings(q, o, s) for q in Q_ALL for o in OFFSETS for s in SEEDS]


def campaign_A() -> list[MCPA_Instance_Settings]:
    settings = campaign_A_base()
    for q in Q_ALL:
        for o in OFFSETS:
            for s in SEEDS:
                settings += [_quadrant_settings(q, o, s, resources_capacity=c) for c in (3, 7)]          # A2 capacità: 240
                settings.append(_quadrant_settings(q, o, s, demand_min=1, demand_max=9))                # A3 domanda: 120
                settings += [_quadrant_settings(q, o, s, n_pairs_per_quadrant=round(rho * N_PAIRS_BASE))
                             for rho in (0.5, 1.5)]                                                     # A4 densità: 240
    return settings   # 720 istanze


def campaign_B() -> list[MCPA_Instance_Settings]:
    # stesse masse totali di A con q = 4 e q = 9 a rho = 1, stessa densità per area
    sizes = [(2 * BASE_GRID_SIDE, 4 * N_PAIRS_BASE), (3 * BASE_GRID_SIDE, 9 * N_PAIRS_BASE)]
    settings = []
    for grid_side, n_pairs in sizes:
        for s in SEEDS:
            base = dict(grid_side=grid_side, n_quadrants=1, n_pairs_per_quadrant=n_pairs, resources_capacity=5, offset=0,
                        seed=s, max_cluster_size=M_STAR, k=K)

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
    return settings   # 200 istanze


def run_ablations_A(runner: MCPA_Runner):
    # da lanciare dopo la campagna A, nella stessa cartella (servono i run di riferimento con il globale)
    base = campaign_A_base()
    reduced = [st for st in base if st.n_quadrants in Q_REDUCED]

    runner.run_all([replace(st, partitioner="random") for st in base], run_global=False)                              # 120
    runner.run_all([replace(st, max_cluster_size=round(f * M_STAR)) for st in reduced for f in (0.5, 0.75, 1.25, 1.5)],
                   run_global=False)                                                                                   # 240
    runner.run_all([replace(st, sim_weighting="uniform", sim_beta=None, resource_weights=w) for st in base
                    for w in ("uniform", "linear")], run_global=False)                                                 # 240
    runner.run_all([replace(st, k=k) for st in reduced for k in (5, 15)])                                              # 120, k cambia il modello: serve il globale


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Campagne MCPA")
    parser.add_argument("campaign", choices=["A", "B", "ablations_A"], nargs="+")
    args = parser.parse_args()

    for campaign in args.campaign:
        if campaign == "A":
            MCPA_Runner(OUT_A, threads=THREADS).run_all(campaign_A())
        elif campaign == "B":
            MCPA_Runner(OUT_B, threads=THREADS).run_all(campaign_B())
        elif campaign == "ablations_A":
            run_ablations_A(MCPA_Runner(OUT_A, threads=THREADS))












# todo single source sono apposto
# todo TB sono apposto quando uso Multi Source, quando uso Single Source sembrerebbe apposto quando modifico SSCFL_Heuristic_solver -> _compute_shipping_cost -> vedere commento
# todo multi source (TEST_BED_C) sono da trovare le soluzioni. Esistono anche TEST_BED_A e TEST_BED_C a questo link: http://wpage.unina.it/sforza/test/
