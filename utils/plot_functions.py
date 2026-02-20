import random
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import seaborn as sns
import matplotlib.colors as mcolors
import matplotlib as mpl

from mcpa.elements.pair import OD_Pair

from typing import Any, Iterable, Optional


def plot_graph_clients_facilities(
    G: nx.Graph,
    clients: Iterable[Any],
    facilities: Iterable[Any],
    pos: Optional[dict[Any, tuple[float, float]]] = None,
    node_size: int = 30,
    client_size: int = 120,
    facility_size: int = 140,
    with_labels: bool = False,
    title: Optional[str] = None,
):
    """
    Plotta G e sovrappone clienti (blu) e facilities (rosso).
    - client.position e facility.position devono essere nodi presenti in G.
    - Se pos non è fornito:
        * se i nodi sono tuple (x, y) usa quelle come coordinate
        * altrimenti usa spring_layout
    """

    # 1) Calcolo posizioni dei nodi
    if pos is None:
        # se i nodi sembrano coordinate (x,y)
        sample = next(iter(G.nodes), None)
        if isinstance(sample, tuple) and len(sample) == 2:
            pos = {n: (float(n[0]), float(n[1])) for n in G.nodes}
        else:
            pos = nx.spring_layout(G, seed=0)

    # 2) Nodi dei clienti/facilities
    client_nodes = [c.position for c in clients if c.position in G]
    facility_nodes = [f.position for f in facilities if f.position in G]

    # 3) Disegno grafo base
    plt.figure(figsize=(10, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.35, width=1.0)
    nx.draw_networkx_nodes(G, pos, node_size=node_size, alpha=0.6)

    # 4) Overlay: facilities (rosso) e clients (blu)
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=facility_nodes,
        node_size=facility_size,
        node_color="red",
        alpha=0.9,
        label="Facilities"
    )
    nx.draw_networkx_nodes(
        G, pos,
        nodelist=client_nodes,
        node_size=client_size,
        node_color="blue",
        alpha=0.9,
        label="Clients"
    )

    if with_labels:
        nx.draw_networkx_labels(G, pos, font_size=5)

    plt.legend()
    if title:
        plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()



# ---------------------------------
# Plot dei grafi
# ---------------------------------
def plot_graph(
    G: nx.Graph,
    od_pairs: list[OD_Pair] | None = None
) -> None:
    plt.figure(figsize=(8, 6), dpi=200)

    pos = grid_layout(G)

    nx.draw(
        G, pos,
        node_size=1, node_color="black",
        edge_color="gray", with_labels=False
    )

    if od_pairs:
        for pair in od_pairs:
            src, dst, pair_id = pair.src, pair.dst, pair.id
            nx.draw_networkx_labels(
                G, pos,
                labels={src: f"s_{pair_id}"},
                font_size=4,
                font_color="black",
                font_weight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="black", lw=0.6, alpha=0.9)
            )
            nx.draw_networkx_labels(
                G, pos,
                labels={dst: f"d_{pair_id}"},
                font_size=4,
                font_color="black",
                font_weight="bold",
                bbox=dict(boxstyle="round,pad=0.2",
                          fc="white", ec="black", lw=0.6, alpha=0.9)
            )

    ax = plt.gca()
    draw_half_separators(ax, G)
    plt.axis("equal")
    plt.show()


def draw_half_separators(ax: plt.Axes, G) -> None:
    nodes = list(G.nodes())
    n_rows = max(i for i, _ in nodes) + 1
    n_cols = max(j for _, j in nodes) + 1

    mid_col = n_cols / 3
    mid_row = n_rows / 3

    # separatore verticale
    ax.plot([mid_col - 0.5, mid_col - 0.5],
            [0.5, -n_rows + 0.5],
            lw=2, color="k", alpha=1.)

    # separatore orizzontale
    ax.plot([-0.5, n_cols - 0.5],
            [-mid_row + 0.5, -mid_row + 0.5],
            lw=2, color="k", alpha=1.)

    mid_col = 2 * n_cols / 3
    mid_row = 2 * n_rows / 3

    # separatore verticale
    ax.plot([mid_col - 0.5, mid_col - 0.5],
            [0.5, -n_rows + 0.5],
            lw=2, color="k", alpha=1.)

    # separatore orizzontale
    ax.plot([-0.5, n_cols - 0.5],
            [-mid_row + 0.5, -mid_row + 0.5],
            lw=2, color="k", alpha=1.)




def plot_paths(
    G: nx.Graph,
    od_pairs: list[OD_Pair],
    colors: dict | None = None,
) -> None:
    plt.figure(figsize=(8, 6), dpi=200)

    pos = grid_layout(G)

    nx.draw(
        G, pos,
        node_size=1, node_color="black",
        edge_color="gray", with_labels=False
    )

    src_labels = {}
    dst_labels = {}

    for od_pair in od_pairs:
        dijkstra_path = od_pair.k_shortest_paths[0].visits
        edges = list(zip(dijkstra_path, dijkstra_path[1:]))
        edge_color = colors[od_pair.id] if colors else to_hex((random.random(), random.random(), random.random()))

        nx.draw_networkx_edges(
            G, pos,
            edgelist=edges,
            width=2.5,
            edge_color=edge_color,
            alpha=0.95,
        )

        src_labels[dijkstra_path[0]] = f"s_{od_pair.id}"
        dst_labels[dijkstra_path[-1]] = f"d_{od_pair.id}"

    nx.draw_networkx_labels(
        G, pos, labels=src_labels,
        font_size=4, font_color="black", font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6, alpha=0.9)
    )

    nx.draw_networkx_labels(
        G, pos, labels=dst_labels,
        font_size=4, font_color="black", font_weight="bold",
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="black", lw=0.6, alpha=0.9)
    )

    ax = plt.gca()
    draw_half_separators(ax, G)
    plt.axis("equal")
    plt.show()


def grid_layout(G: nx.Graph):
    return {(i, j): (j, -i) for (i, j) in G.nodes()}


def plot_heatmap(df, columns):
    subset = df[columns]
    matrice_corr = subset.corr(method='spearman')

    plt.figure(figsize=(12, 12))
    sns.heatmap(matrice_corr,
                annot=True,
                cmap='coolwarm',
                fmt=".2f",
                vmin=-1, vmax=1)

    plt.title('Correlation Matrix')
    plt.show()


def plot_curves(curves, y_label, x_values, min_value, max_value, cbar_str, name_file):
    # 'viridis', 'plasma', 'coolwarm', 'spring', 'berlin'
    cmap = mpl.colormaps["cividis"]
    norm = mcolors.Normalize(vmin=min_value, vmax=max_value)
    y_limit = 0

    plt.figure(figsize=(14, 10))

    for curve, (n_el, q) in curves:
        if max(curve) > y_limit:
            y_limit = max(curve)
        if n_el <= 375:
            color = cmap(norm(n_el * q))
            plt.plot(x_values, curve, marker='o', linestyle='-', color=color, alpha=0.7)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=plt.gca())

    cbar.set_label(cbar_str, fontsize=24)
    cbar.ax.tick_params(labelsize=24)

    plt.ylim((0, y_limit))

    plt.xticks(x_values)
    plt.grid(True, linestyle='--', alpha=0.5)

    plt.xlabel("Offset", fontsize=30)
    plt.ylabel(y_label, fontsize=30)
    plt.tick_params(axis='both', which='major', labelsize=26)

    plt.savefig(name_file, dpi=1200)
    # plt.show()


def get_curves(dir_path, offset_values):
    gap_curves = []
    time_curves = []
    min_val = float("inf")
    max_val = float("-inf")

    for j in range(4, 34):
        gap_curve = []
        time_curve = []
        n_elements = ()

        for i in offset_values:
            with open(f"{dir_path}/tex/summary_{i}.tex", "r", encoding="utf-8") as f:
                lines = f.readlines()
            line = lines[j]
            arr = line[:-2].replace(" ", "").split("&")
            gap = float(arr[-2])
            time = float(arr[4])
            gap_curve.append(gap)
            time_curve.append(time)
            n_elements = (int(arr[0]), int(arr[1])) # int(arr[0]) * int(arr[1])
            tot = int(arr[0]) * int(arr[1])
            if tot < min_val:
                min_val = tot
            if tot > max_val:
                max_val = tot

        gap_curves.append((gap_curve, n_elements))
        time_curves.append((time_curve, n_elements))

    return gap_curves, time_curves, min_val, max_val