from dataclasses import dataclass
from datetime import datetime
from typing import Callable, Optional

import numpy as np
from matplotlib import pyplot as plt

from general.general_environment import General_Environment
from general.general_solver import General_Solver


@dataclass
class ScalabilityPoint:
    n: int
    runs_done: int
    tl_count: int
    tl_rate: float
    median_capped: float
    p90_capped: float
    max_capped: float
    is_bad: bool


@dataclass
class ScalabilityResult:
    points: dict[int, ScalabilityPoint]
    onset_n: Optional[int]  # primo n dove “diventa probabilmente difficile”, secondo regola di stop
    stop_reason: str


def read():
    with open("../results/mcpa/mcpa_parameters_selection_results.txt", "r") as f:
        lines = f.readlines()
    points = {}
    j = 0
    for i in range(len(lines)):
        if lines[i].__contains__("n: runs_done, tl_count, tl_rate, median_capped, p90_capped, max_capped, is_bad"):
            j = i + 1
    c = 0
    for i in range(j, len(lines)):
        if lines[i].strip() == "":
            continue

        n = int(lines[i].split(":")[0])
        right_arr = lines[i].split(":")[1].split(",")
        points[n] = ScalabilityPoint(n, 5, int(right_arr[1]), float(right_arr[2]), float(right_arr[3]), float(right_arr[4]), float(right_arr[5]), bool(int(right_arr[6])))
        print(n, right_arr)
        c += 1

    _save_scalability_plot(points, "mcpa_parameters_selection_results", "Number of OD Pairs", False)
    _save_scalability_plot(points, "mcpa_parameters_selection_results", "Number of OD Pairs", True)




def _save_scalability_plot(
    points: dict[int, ScalabilityPoint],
    name_file: str,
    x_label: str = "n",
    logy: bool = False,
):
    xs = sorted(points.keys())
    med = [points[x].median_capped for x in xs]
    p90 = [points[x].p90_capped for x in xs]
    mx = [points[x].max_capped for x in xs]

    plt.figure()
    plt.plot(xs, med, marker="o", markersize=3, label="median")
    plt.plot(xs, p90, marker="o", markersize=3, label="p90")
    plt.plot(xs, mx, marker="o", markersize=3, label="max")

    plt.xlabel(x_label)
    plt.ylabel("Time (s)")
    if logy:
        plt.yscale("log")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    # plt.savefig(f"{name_file}_600dpi.pdf", dpi=600)
    plt.savefig(f"{name_file}{"_log" if logy else ""}.pdf", dpi=1200)

    plt.close()

read()

def run_time_scalability(
    grid_side: int,
    k: int,
    env_maker: Callable[..., General_Environment],
    solver_maker: Callable[..., General_Solver],
    file_results_name: str,
    *,
    time_limit: int = 1800,
    n_seeds: int = 5,
    n_start: int = 50,
    step: int = 5,
    n_max: Optional[int] = None,
    # seed handling
    seed_base: int = 0,
    seed_mode: str = "by_n",   # "by_n" (default) oppure "fixed"
    # definizione di "bad point"
    cap_times: bool = True,
    tl_status: str = "TIME_LIMIT",
    tl_bad_count: int = 3,             # es: “almeno 3 su 5 vanno in TL”
    tl_rate_threshold: Optional[float] = None,  # alternativa (es 0.6). Se None non usata.
    p90_bad_frac: float = 0.8,         # p90 >= 0.8 * time_limit => bad
    median_bad_sec: Optional[float] = None,  # es 60 o 120; se None non usata.
    # regole di stop (robustezza)
    consecutive_bad_stop: int = 5,     # “per 5 volte di fila…”
    window_size: Optional[int] = None,         # es 7
    window_bad_needed: Optional[int] = None,   # es 5 (su 7)
    # plotting
    x_label: str = "n",
    logy_plot: bool = False,
    verbose: bool = True,
) -> ScalabilityResult:
    """
    Esegue un esperimento di scalabilità su n crescente.
    Registra tempi cappati (<= time_limit) e ferma quando:
      - bad streak raggiunge consecutive_bad_stop
      - oppure (se attivi) nella finestra mobile window_size ci sono >= window_bad_needed punti bad.
    """

    def make_seed(n: int, j: int) -> int:
        if seed_mode == "fixed":
            # stessi seed per ogni n (0..n_seeds-1), utile se vuoi “controllare” la variabilità
            return seed_base + j
        if seed_mode == "by_n":
            # seed diversi per ogni n (riduce correlazioni fra taglie)
            return seed_base + (n * 100000) + j
        raise ValueError(f"Unknown seed_mode: {seed_mode!r} (use 'fixed' or 'by_n')")

    points: dict[int, ScalabilityPoint] = {}
    bad_flags: list[bool] = []
    bad_streak = 0

    onset_n: Optional[int] = None

    n = n_start
    while True:
        if n_max is not None and n > n_max:
            stop_reason = f"Reached n_max={n_max}"
            break

        times_capped: list[float] = []
        tl_count = 0
        runs_done = 0

        # Early break interno: se hai già raggiunto la soglia TL per definire “bad”, non ha senso sprecare run.
        # (Comunque ricordati: runs_done cambia, quindi tl_rate va calcolato su runs_done!)
        for j in range(n_seeds):
            seed = make_seed(n, j)

            env = env_maker(grid_side, k, n, seed)
            if verbose:
                print(datetime.now().strftime("%d-%m-%Y   %H:%M:%S    "), end="")
                print(f"{env}   seed={seed}")

            solver = solver_maker(env)
            solver.solve()

            total_time = float(sum(getattr(solver, "model_times", [])) + sum(getattr(solver, "resolution_times", [])))
            is_tl = (getattr(solver, "status", None) == tl_status)

            if is_tl:
                tl_count += 1

            t = min(total_time, float(time_limit)) if cap_times else total_time
            times_capped.append(t)
            runs_done += 1

            # early stop sulle repliche: se abbiamo già “certificato” che è bad per TL-count
            if tl_count >= tl_bad_count:
                break

            # oppure se vuoi early stop su tl_rate_threshold (solo se definito)
            if tl_rate_threshold is not None:
                # NB: la max tl_rate raggiungibile aumentando j è monotona in tl_count/runs_done
                if (tl_count / runs_done) >= float(tl_rate_threshold):
                    break

        # metriche robuste (con pochi sample, p90 ~ max, ma va bene comunque)
        arr = np.array(times_capped, dtype=float)
        median_capped = float(np.median(arr)) if arr.size else 0.0
        p90_capped = float(np.quantile(arr, 0.9)) if arr.size else 0.0
        max_capped = float(np.max(arr)) if arr.size else 0.0

        tl_rate = float(tl_count / runs_done) if runs_done > 0 else 0.0

        # definisci “bad” (OR di criteri)
        is_bad = False
        if tl_count >= tl_bad_count:
            is_bad = True
        if (tl_rate_threshold is not None) and (tl_rate >= float(tl_rate_threshold)):
            is_bad = True
        if p90_capped >= (p90_bad_frac * float(time_limit)):
            is_bad = True
        if (median_bad_sec is not None) and (median_capped >= float(median_bad_sec)):
            is_bad = True

        points[n] = ScalabilityPoint(
            n=n,
            runs_done=runs_done,
            tl_count=tl_count,
            tl_rate=tl_rate,
            median_capped=median_capped,
            p90_capped=p90_capped,
            max_capped=max_capped,
            is_bad=is_bad,
        )

        if verbose:
            print(
                f"n={n:>4}  "
                f"median_capped={median_capped:>8.3f}s  "
                f"p90_capped={p90_capped:>8.3f}s  "
                f"max_capped={max_capped:>8.3f}s  "
                f"TL={tl_count}/{runs_done} (rate={tl_rate:.2f})  "
                f"bad={is_bad}"
            )

        bad_flags.append(is_bad)

        # regola 1: 5 volte di fila
        if is_bad:
            bad_streak += 1
        else:
            bad_streak = 0

        if bad_streak >= consecutive_bad_stop:
            onset_n = n
            stop_reason = f"Consecutive bad points reached: {bad_streak} >= {consecutive_bad_stop}"
            break

        # regola 2 (opzionale): finestra mobile
        if window_size is not None and window_bad_needed is not None and len(bad_flags) >= window_size:
            bad_in_window = sum(bad_flags[-window_size:])
            if bad_in_window >= window_bad_needed:
                onset_n = n
                stop_reason = (
                    f"Window rule triggered: last {window_size} have {bad_in_window} bad >= {window_bad_needed}"
                )
                break

        n += step

    # salva risultati
    with open(f"{file_results_name}.txt", "w") as f:
        f.write(f"Grid side = {grid_side}  (cells={grid_side*grid_side})\n")
        f.write(f"k={k}\n")
        f.write(f"time_limit={time_limit}s, n_seeds={n_seeds}\n")
        f.write(f"seed_base={seed_base}, seed_mode={seed_mode}\n\n")

        f.write("BAD definition:\n")
        f.write(f"  tl_bad_count >= {tl_bad_count}\n")
        if tl_rate_threshold is not None:
            f.write(f"  OR tl_rate >= {tl_rate_threshold}\n")
        f.write(f"  OR p90_capped >= {p90_bad_frac} * time_limit\n")
        if median_bad_sec is not None:
            f.write(f"  OR median_capped >= {median_bad_sec} sec\n")
        f.write("\nSTOP rules:\n")
        f.write(f"  consecutive_bad_stop={consecutive_bad_stop}\n")
        if window_size is not None and window_bad_needed is not None:
            f.write(f"  window_size={window_size}, window_bad_needed={window_bad_needed}\n")
        f.write("\n")

        f.write(f"STOP reason: {stop_reason}\n")
        f.write(f"onset_n: {onset_n}\n\n")

        f.write("n: runs_done, tl_count, tl_rate, median_capped, p90_capped, max_capped, is_bad\n")
        for nn in sorted(points):
            p = points[nn]
            f.write(
                f"{nn}: {p.runs_done}, {p.tl_count}, {p.tl_rate:.3f}, "
                f"{p.median_capped:.6f}, {p.p90_capped:.6f}, {p.max_capped:.6f}, {int(p.is_bad)}\n"
            )

    _save_scalability_plot(points, file_results_name, x_label=x_label, logy=logy_plot)

    return ScalabilityResult(points=points, onset_n=onset_n, stop_reason=stop_reason)
