import math
import random
from typing import Tuple, List
import networkx as nx

from elements.agent import Agent
from elements.pair import OD_Pair


# -----------------------------
# Costruzione della griglia
# -----------------------------
def create_grid_graph(
    n: int,
    remove_rnd_nodes_flag: bool | None = False,
) -> nx.Graph:
    G = nx.Graph()
    G.add_nodes_from((i, j) for i in range(n) for j in range(n))
    G.add_edges_from([((i, j), (i + 1, j)) for i in range(n - 1) for j in range(n)] +
                     [((i, j), (i, j + 1)) for i in range(n) for j in range(n - 1)])

    return G if not remove_rnd_nodes_flag else remove_random_nodes(G)


def set_capacities(G, od_pairs, seed: int | None = 42):
    rng = random.Random(seed)
    od_nodes = [node for od_pair in od_pairs for node in (od_pair.src, od_pair.dst)]
    V = list(G.nodes)
    for v in V:
        G.nodes[v]['capacity'] = 0

    for od in od_pairs:
        k = len(od.agents)
        G.nodes[od.src]['capacity'] += k
        G.nodes[od.dst]['capacity'] += k

    for v in V:
        if v not in od_nodes:
            G.nodes[v]["capacity"] = rng.randrange(5, 6) #############



# -------------------------------------
# Scelta di sorgenti e destinazioni
# -------------------------------------

def get_grid_side(G: nx.Graph) -> int:
    return int(math.isqrt(G.number_of_nodes()))


def quadrant_ranges(n: int, x: int, y: int, offset:  int) -> Tuple[range, range]:
    mid = n // 2
    row_range = range(0, mid + offset) if x == 0 else range(mid - offset, n)
    col_range = range(0, mid + offset) if y == 0 else range(mid - offset, n)
    return row_range, col_range


def rand_coord_in_quadrant(
    n: int,
    x: int,
    y: int,
    offset: int,
    rng: random.Random
) -> Tuple[int, int]:
    row_range, col_range = quadrant_ranges(n, x, y, offset)
    return rng.choice(row_range), rng.choice(col_range)


def choose_pairs(
    G: nx.Graph,
    pairs_per_quadrant: int,
    offset: int | None = 0,
    seed: int | None = 42
) -> List[OD_Pair]:

    rng = random.Random(seed)
    n = get_grid_side(G)
    od_pairs: List[OD_Pair] = []
    seen_pairs: set[Tuple] = set()
    total_agents = 0

    for x in range(2):
        for y in range(2):
            generated = 0
            while generated < pairs_per_quadrant:
                src = rand_coord_in_quadrant(n, x, y, offset, rng)
                dst = rand_coord_in_quadrant(n, x, y, offset, rng)

                if src == dst or ((src, dst) in seen_pairs) or src not in G.nodes or dst not in G.nodes:
                    continue

                number_of_agents = rng.randrange(1, 2)
                agents = [Agent(i, src, dst) for i in range(total_agents, total_agents + number_of_agents)]

                pair_id = generated + pairs_per_quadrant * (2 * x + y)
                od_pair = OD_Pair(pair_id, src, dst, agents)

                od_pairs.append(od_pair)
                seen_pairs.add((src, dst))
                total_agents += number_of_agents
                generated += 1

    set_capacities(G, od_pairs)
    return sorted(od_pairs, key=lambda od: od.id)


def remove_random_nodes(G: nx.Graph) -> nx.Graph:
    rng = random.Random()
    reduced_graph = G.copy()
    nodes_to_remove = [node for node in G.nodes() if rng.random() < 0.1]
    reduced_graph.remove_nodes_from(nodes_to_remove)

    nodes_to_remove = [node for node in reduced_graph.nodes() if reduced_graph.degree(node) == 0]
    reduced_graph.remove_nodes_from(nodes_to_remove)

    return reduced_graph
