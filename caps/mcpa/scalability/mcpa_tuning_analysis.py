import pandas as pd

from caps.mcpa.scalability.mcpa_runner import load_results
from utils.tuning_analysis import analyze_similarity_tuning


def analyze_tuning(out_dir: str) -> pd.DataFrame:
    """Analisi della taratura della similarità per MCPA (la logica è in utils/tuning_analysis.py, comune a tutti i problemi)."""
    return analyze_similarity_tuning(load_results(out_dir), {"agenti rilasciati mediani": "released_agents"})
