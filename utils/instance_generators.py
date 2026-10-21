import networkx as nx

Coord = tuple[int, int]


# ----- generatori di coppie (partenza, arrivo) per le famiglie senza quadranti -----


def sample_pair_uniform(G: nx.Graph, nodes: list[Coord], rng, L_min: int | None = None, L_max: int | None = None) -> tuple[Coord, Coord]:
    """
    Partenza uniforme sui nodi; arrivo uniforme tra i nodi a distanza (sul grafo) compresa in [L_min, L_max].
    Senza L_max l'arrivo è uniforme su tutta la griglia.
    """
    while True:
        src = rng.choice(nodes)

        if L_max is None:
            dst = rng.choice(nodes)
            if dst != src:
                return src, dst
            continue

        dist = nx.single_source_shortest_path_length(G, src, cutoff=L_max)
        candidates = sorted(v for v, d in dist.items() if d >= max(L_min or 1, 1))
        if candidates:
            return src, rng.choice(candidates)


def make_hotspots(nodes: list[Coord], rng, n_hotspots: int) -> tuple[list[Coord], list[float]]:
    """Centri degli hotspot e pesi (masse diverse: ogni hotspot riceve una quota di coppie proporzionale al peso)."""
    centers = rng.sample(nodes, n_hotspots)
    weights = [0.5 + rng.random() for _ in range(n_hotspots)]
    return centers, weights


def sample_pair_hotspot(nodes: list[Coord], node_set: set[Coord], rng, centers: list[Coord], weights: list[float],
                        sigma: float, noise: float) -> tuple[Coord, Coord]:
    """
    Con probabilità noise la coppia è uniforme; altrimenti si sceglie un hotspot (in base al peso) e
    partenza e arrivo sono estratti da una gaussiana discretizzata di deviazione sigma attorno al centro.
    """
    if rng.random() < noise:
        while True:
            src, dst = rng.choice(nodes), rng.choice(nodes)
            if src != dst:
                return src, dst

    center = rng.choices(centers, weights=weights)[0]
    while True:
        src = _gaussian_node(center, sigma, rng, node_set)
        dst = _gaussian_node(center, sigma, rng, node_set)
        if src != dst:
            return src, dst


# ----- generatori di singole posizioni (es. clienti CFL) -----


def sample_point_hotspot(nodes: list[Coord], node_set: set[Coord], rng, centers: list[Coord], weights: list[float],
                         sigma: float, noise: float) -> Coord:
    """
    Con probabilità noise un nodo uniforme; altrimenti si sceglie un hotspot (in base al peso) e si estrae un nodo
    da una gaussiana discretizzata di deviazione sigma attorno al centro.
    """
    if rng.random() < noise:
        return rng.choice(nodes)
    center = rng.choices(centers, weights=weights)[0]
    return _gaussian_node(center, sigma, rng, node_set)


def _gaussian_node(center: Coord, sigma: float, rng, node_set: set[Coord]) -> Coord:
    while True:
        v = (round(rng.gauss(center[0], sigma)), round(rng.gauss(center[1], sigma)))
        if v in node_set:
            return v
