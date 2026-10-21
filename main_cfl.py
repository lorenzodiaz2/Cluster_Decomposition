from caps.cfl.scalability.cfl_calibration import calibrate_opening_cost_scale


THREADS = 4                               # core fisici della macchina (Gurobi rende meglio sui core fisici che su quelli logici)
OUT_CALIBRATION = "results/cfl/calibration"


if __name__ == '__main__':
    # step 1: taratura della scala dei costi di apertura (rilanciando riprende dai run non ancora completati)
    calibrate_opening_cost_scale(OUT_CALIBRATION, threads=THREADS)
