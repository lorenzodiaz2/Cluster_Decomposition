"""
Numeri dell'esempio della Sezione 3.3 del paper (similarità tra entità, figura fig:sim-example).

Calcola con le formule del paper (eq:sim:optweight, eq:sim:usage, eq:sim:resweight, eq:sim:sigma):
- i pesi delle opzioni delle entità dei pannelli a (CPA) e b (CFL), dove l'opzione costosa ha costo extra 2 (due passi di
  tempo nel CPA, due unità di distanza nel CFL), con beta = 1 e con pesi uniformi (beta -> 0);
- le similarità sigma(e1, e2) e sigma(e2, e3), e nel CFL anche sigma(e2, e4) e sigma(e3, e4), con domande unitarie e
  pesi delle risorse pari a uno;
- il costo dell'esempio sui costi di attivazione, con i due clienti nello stesso cluster o in cluster diversi;
- i pesi delle tre risorse del pannello c.

Dal 25-11-2026 il costo extra non è più diviso per una scala kappa_e (decisione del leader).

Uso (da code/):  python analysis/similarity_example.py
"""
import math

BETA = 1.0


def option_weights(costs, beta=BETA):
    # pesi esponenziali normalizzati (eq:sim:optweight); sottrarre il costo minimo non cambia i pesi
    c_min = min(costs)
    raw = [math.exp(-beta * (c - c_min)) for c in costs]
    tot = sum(raw)
    return [w / tot for w in raw]


def uniform_weights(costs):
    return [1.0 / len(costs)] * len(costs)


def fmt(ws):
    return ", ".join(f"{w:.4f}" for w in ws)


def main():
    print("== Pannelli (a) e (b): opzione economica con costo extra 0, opzione costosa con costo extra 2")
    costs = [0, 2]
    psi = option_weights(costs)
    print(f"  beta = 1:  psi = {fmt(psi)}")
    print(f"  uniformi:  psi = {fmt(uniform_weights(costs))}")

    print("== Pannelli (a) e (b): ogni entità ha un'opzione economica e una costosa; domande unitarie, w_r = 1")
    # e1 ed e2 condividono una risorsa con le opzioni economiche (stesso cluster), e2 ed e3 una risorsa con le
    # opzioni costose (cluster diversi); nel CFL r3 è la candidata lontana di e2, e3 ed e4
    for name, p in [("beta = 1", psi), ("uniformi", uniform_weights(costs))]:
        s12 = 1.0 * p[0] * p[0]
        s23 = 1.0 * p[1] * p[1]
        s24_cfl = p[1] * p[1]                   # nel CFL e2 ed e4 condividono solo r3 (costosa per entrambi)
        s34_cfl = p[0] * p[0] + p[1] * p[1]     # nel CFL e3 ed e4 condividono r2 (economica) e r3 (costosa)
        print(f"  {name}:  sigma(e1,e2) = {s12:.4f}   sigma(e2,e3) = {s23:.4f}   CFL sigma(e2,e4) = {s24_cfl:.4f}"
              f"   CFL sigma(e3,e4) = {s34_cfl:.4f}")

    print("== Costi di attivazione: e, e' possono usare r (f = 10, capacità per entrambi); e anche r1, e' anche r2"
          " (f = 7); costi di assegnamento trascurabili")
    f_shared, f_own = 10, 7
    together = f_shared                                   # stesso cluster: si apre solo r
    # cluster diversi: ogni sottoproblema vede un solo cliente e sceglie l'apertura più economica per lui
    separate = 2 * min(f_shared, f_own)
    print(f"  stesso cluster: costo = {together}   cluster diversi: costo = {separate}")

    print("== Pannello (c): carichi stimati b_e a_e(r) per entità, capacità, peso w_r")
    resources = {
        "r_a": ([1.0, 0.8, 0.7, 0.5], 2.0),
        "r_b": ([0.25, 0.25], 2.0),     # due entità con poca domanda
        "r_c": ([1.0, 0.8, 0.7, 0.5], 6.0),
    }
    for r, (loads, cap) in resources.items():
        print(f"  {r}: carico stimato = {sum(loads):.2f}, capacità = {cap:.0f}, w = {sum(loads) / cap:.4f}")
    # a parità di usi, il contributo di una risorsa a sigma è proporzionale a w_r
    w = {r: sum(loads) / cap for r, (loads, cap) in resources.items()}
    print(f"  rapporto dei contributi a sigma, r_a contro r_c: {w['r_a'] / w['r_c']:.2f}")


if __name__ == "__main__":
    main()
