from pathlib import Path
from typing import Callable, Any
import networkx as nx
import pandas as pd

from cfl.elements.client import Client
from cfl.elements.facility import Facility


def read_OR_instance(
    filepath: str | Path,
    k: int | None = None,
    *,
    node_id: Callable[[str, int], Any] | None = None,
) -> tuple[nx.DiGraph, list[Client], list[Facility]]:

    filepath = Path(filepath)
    tokens = filepath.read_text().split()
    if len(tokens) < 2:
        raise ValueError(f"File '{filepath}' non valido: contiene meno di 2 token.")

    it = iter(tokens)
    try:
        J = int(next(it))  # facilities
        I = int(next(it))  # clients
    except StopIteration:
        raise ValueError(f"File '{filepath}' troncato: mancano J e I.")

    if k is None:
        k = J

    if node_id is None:
        node_id = lambda kind, idx: (kind, idx)

    # facilities: capacity + opening cost
    capacities = [0.0] * J
    open_costs = [0.0] * J
    for j in range(J):
        try:
            capacities[j] = float(next(it))
            open_costs[j] = float(next(it))
        except StopIteration:
            raise ValueError(f"File '{filepath}' troncato: mancano (capacity, opening_cost) per qualche facility.")

    # demands
    demands = []
    for _ in range(I):
        try:
            demands.append(float(next(it)))
        except StopIteration:
            raise ValueError(f"File '{filepath}' troncato: mancano le domande dei clienti.")

    # build objects (position coerente col node_id)
    clients: list[Client] = []
    for i in range(I):
        cpos = node_id("c", i)
        clients.append(Client(i, cpos, demands[i]))

    facilities: list[Facility] = []
    for j in range(J):
        fpos = node_id("f", j)
        facilities.append(Facility(j, fpos, capacities[j], open_costs[j]))

    # graph
    G = nx.DiGraph()

    for i, c in enumerate(clients):
        cnode = node_id("c", i)
        G.add_node(cnode, kind="client", demand=c.demand, obj=c)

    for j, f in enumerate(facilities):
        fnode = node_id("f", j)
        G.add_node(fnode, kind="facility", capacity=f.capacity, opening_cost=f.activation_cost, obj=f)

    # read costs and add edges on the fly: c_{j,i}
    for j in range(J):
        fnode = node_id("f", j)
        for i in range(I):
            cnode = node_id("c", i)
            try:
                cost = float(next(it))
            except StopIteration:
                raise ValueError(f"File '{filepath}' troncato: mancano costi di assegnazione.")
            G.add_edge(cnode, fnode, cost=cost)

    set_client_facility_lists_from_cost_graph(G, clients, k)
    return G, clients, facilities



def read_TBED1_instance(
    filepath: str | Path,
    k: int,
    *,
    node_id: Callable[[str, int], Any] | None = None,
) -> tuple[nx.DiGraph, list[Client], list[Facility]]:
    """
    Parse TBED1 instances (TB1..TB5) from OR-Brescia.

    Format (from readme):
        |J| |I|
        d_1 ... d_|I|
        s_1 ... s_|J|
        f_1 ... f_|J|
        c_{11} ... c_{1|I|}
        ...
        c_{|J|1} ... c_{|J||I|}

    where c_{ji} is the cost of supplying ONE unit of demand of customer i from facility j.

    Returns:
        G: nx.DiGraph with edges client->facility storing "cost" attribute (unit cost)
        clients: list[Client]
        facilities: list[Facility]
    """
    filepath = Path(filepath)
    tokens = filepath.read_text().split()
    if len(tokens) < 2:
        raise ValueError(f"File '{filepath}' non valido: contiene meno di 2 token.")

    it = iter(tokens)
    try:
        J = int(next(it))  # facilities
        I = int(next(it))  # customers / clients
    except StopIteration:
        raise ValueError(f"File '{filepath}' troncato: mancano J e I.")

    if k is None:
        k = J

    if node_id is None:
        node_id = lambda kind, idx: (kind, idx)

    # demands d_i (i=1..I)
    demands: list[float] = []
    for _ in range(I):
        try:
            demands.append(float(next(it)))
        except StopIteration:
            raise ValueError(f"File '{filepath}' troncato: mancano le domande dei clienti (d_i).")

    # capacities s_j (j=1..J)
    capacities: list[float] = []
    for _ in range(J):
        try:
            capacities.append(float(next(it)))
        except StopIteration:
            raise ValueError(f"File '{filepath}' troncato: mancano le capacità delle facility (s_j).")

    # opening costs f_j (j=1..J)
    open_costs: list[float] = []
    for _ in range(J):
        try:
            open_costs.append(float(next(it)))
        except StopIteration:
            raise ValueError(f"File '{filepath}' troncato: mancano i costi di apertura (f_j).")

    # build objects (positions consistent with node_id)
    clients: list[Client] = []
    for i in range(I):
        cpos = node_id("c", i)
        clients.append(Client(i, cpos, demands[i]))

    facilities: list[Facility] = []
    for j in range(J):
        fpos = node_id("f", j)
        facilities.append(Facility(j, fpos, capacities[j], open_costs[j]))

    # graph
    G = nx.DiGraph()

    for i, c in enumerate(clients):
        cnode = node_id("c", i)
        G.add_node(cnode, kind="client", demand=c.demand, obj=c)

    for j, f in enumerate(facilities):
        fnode = node_id("f", j)
        G.add_node(
            fnode,
            kind="facility",
            capacity=f.capacity,
            opening_cost=f.activation_cost,
            obj=f,
        )

    # costs c_{j,i}: J rows, each with I costs (unit costs!)
    for j in range(J):
        fnode = node_id("f", j)
        for i in range(I):
            cnode = node_id("c", i)
            try:
                unit_cost = float(next(it))
            except StopIteration:
                raise ValueError(
                    f"File '{filepath}' troncato: mancano costi c_{{j,i}} (unit cost) alla riga facility {j}."
                )
            G.add_edge(cnode, fnode, cost=unit_cost)

    # If extra tokens exist, warn (optional). Here we treat it as an error to catch format issues.
    try:
        extra = next(it)
        raise ValueError(f"File '{filepath}' contiene token extra non attesi (es. '{extra}'): formato non conforme?")
    except StopIteration:
        pass

    set_client_facility_lists_from_cost_graph(G, clients, k, node_id=node_id)
    return G, clients, facilities




def set_client_facility_lists_from_cost_graph(
    G: nx.DiGraph,
    clients: list[Client],
    k: int,
    *,
    node_id=lambda kind, idx: (kind, idx),
) -> None:

    for c in clients:
        cnode = node_id("c", c.id)

        scored = []
        for fnode in G.successors(cnode):
            f: Facility = G.nodes[fnode]["obj"]
            cost = G[cnode][fnode]["cost"]
            scored.append((cost, -f.capacity, f.id, f))  # f.id per determinismo

        scored.sort(key=lambda t: (t[0], t[1], t[2]))
        ordered = [f for *_ , f in scored]
        c.all_facilities = ordered

        chosen = []
        cap_sum = 0
        for f in ordered:
            chosen.append(f)
            cap_sum += f.capacity
            if len(chosen) >= k and cap_sum >= c.demand:
                break

        c.k_facilities = chosen


def get_sum_from_array_string(arr):
    arr = arr.strip("[]")
    num_arr = arr.split(", ")
    s = 0
    for n in num_arr:
        s += float(n)
    return s

def get_last_from_array_string(arr):
    if not "[" in arr:
        return 0
    arr = arr.strip("[]")
    num_arr = arr.split(", ")
    return float(num_arr[-1])


def fmt_int(x):
    return f"{int(x)}"


def fmt_float(x):
    if pd.isna(x): return "-"
    return f"{x:.2f}"