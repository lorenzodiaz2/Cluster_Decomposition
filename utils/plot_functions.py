import random
from typing import List
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.colors import to_hex

from elements.pair import OD_Pair


# ---------------------------------
# Plot dei grafi
# ---------------------------------
def plot_graph(
    G: nx.Graph,
    od_pairs: List[OD_Pair] | None = None
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

    mid_col = n_cols / 2
    mid_row = n_rows / 2

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
    od_pairs: List[OD_Pair],
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
