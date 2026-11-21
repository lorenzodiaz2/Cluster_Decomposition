import pandas as pd

from caps.cfl.scalability.cfl_runner import load_results
from utils.tuning_analysis import analyze_similarity_tuning


def analyze_tuning(out_dir: str) -> pd.DataFrame:
    """Analisi della taratura della similarità per una variante CFL (SS o SD, una cartella ciascuna); logica comune in utils/tuning_analysis.py."""
    return analyze_similarity_tuning(load_results(out_dir), {"clienti rilasciati mediani": "released_clients",
                                                             "quantità rilasciata mediana": "released_quantity"})
