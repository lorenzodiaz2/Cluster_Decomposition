import pandas as pd

from cfl.scalability.cfl_scalability_resolution import run_sscfl_scalability_instance, run_mscfl_scalability_instance
from mcpa.scalability.mcpa_scalability_resolution import run_mcpa_scalability




if __name__ == '__main__':
    seed = 1000


    df = pd.read_csv(f"results/cfl/ss/sscfl_results.csv")
    for offset in [-15, -7, 0, 5]:
        run_sscfl_scalability_instance(50, 45, 175, offset, 10, df, seed, [2, 3, 4, 6, 7, 9], 5)
        seed += 30
        print()


    print("\n\n\n=====================================================\n\n\n")


    df = pd.read_csv(f"results/cfl/ms/mscfl_results.csv")
    for n_pairs in [40, 45, 50, 55, 60]:
        for offset in [-15, -7, 0, 5]:
            run_mscfl_scalability_instance(50, n_pairs, 250, offset, 10, df, seed, [2, 3, 4, 6, 7, 9], 5)
            seed += 30
            print()



    print("\n\n\n=====================================================\n\n\n")


    seed = 1218
    df = pd.read_csv(f"results/mcpa/mcpa_results.csv")

    run_mcpa_scalability(20, 140, 750, 1, df, seed, [7], 5)
    seed += 5
    print()

    for n_pairs_per_quadrant in [145, 150, 155, 160]:
        run_mcpa_scalability(20, n_pairs_per_quadrant, 750, 1, df, seed, [3, 7], 5)
        seed += 10
        print()
    print()


    for n_pairs_per_quadrant in [140, 145, 150, 160]:
        run_mcpa_scalability(20, n_pairs_per_quadrant, 750, 2, df, seed, [3], 5)
        seed += 10
        print()
    print()


    run_mcpa_scalability(20, 140, 750, 5, df, seed, [7], 5)
    seed += 5
    print()

    for n_pairs_per_quadrant in [145, 150, 155, 160]:
        run_mcpa_scalability(20, n_pairs_per_quadrant, 750, 5, df, seed, [3, 7], 5)
        seed += 10
        print()
    print()





# todo single source sono apposto
# todo TB sono apposto quando uso Multi Source, quando uso Single Source sembrerebbe apposto quando modifico SSCFL_Heuristic_solver -> _compute_shipping_cost -> vedere commento
# todo multi source (TEST_BED_C) sono da trovare le soluzioni. Esistono anche TEST_BED_A e TEST_BED_C a questo link: http://wpage.unina.it/sforza/test/