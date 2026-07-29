from typing import Union, List, Dict

import numpy as np
from instancespace import InstanceSpace, Metadata
from instancespace.data import options as isa_options
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.trace import TraceStage
from instancespace._serialisers import _write_array_to_csv, _make_bind_labels
from dataclasses import replace
import pandas as pd
import sys
import os
from pathlib import Path

from shapely.geometry import Polygon, MultiPolygon


class ISA:

    def __init__(self, df: pd.DataFrame, instance_col_name: Union[str, int],
                 feat_names: List[str], alg_names: List[str], custom_options: Dict = None):

        df_features = df[feat_names] # nuovo dataframe con solo le colonne feature
        df_algs = df[alg_names] # nuovo dataframe con solo le colonne algoritmi
        self.feat_names = feat_names
        self.alg_names = alg_names
        self.inst_names = df[instance_col_name].astype(str).to_list()

        # feat_names = ['feature_' + f if not f.startswith('feature_') else f for f in self.feat_names]
        # alg_names = ['algo_' + f if not f.startswith('algo_') else f for f in self.alg_names]
        df_features.columns = self.feat_names

        self.metadata = Metadata(
            feature_names=self.feat_names,
            algorithm_names=self.alg_names,
            features=df_features.to_numpy().astype(float),
            algorithms=df_algs.to_numpy().astype(float),
            instance_sources=None,
            instance_labels=self.inst_names
        )

        self.options = isa_options.from_json_file("options.json")
        if custom_options is not None:
            self.set_options(custom_options)

        self.isa = InstanceSpace(self.metadata, self.options,
                                 stages=[PreprocessingStage, PrelimStage, SiftedStage, PilotStage, PythiaStage,
                                         CloisterStage, TraceStage, ], )
        self.output = None

    def run(self, verbose=True):
        old_stdout = None
        if not verbose:
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, "w")

        # reset old stdout
        else:
            print(self.isa._runner._stage_order)
        self.isa.build()
        self.output = self.isa._final_output
        if not verbose:
            sys.stdout = old_stdout

    def set_options(self, custom_options: Dict):
        self.options = replace(self.options, **{
            key: replace(getattr(self.options, key), **updates)
            for key, updates in custom_options.items()
        })

    def save_plots(self, output):
        self.isa.model.save_graphs(Path(output))

    def save_csv(self, output):
        self.isa.model.save_to_csv(Path(output))

    def save_full_csv(self, df_original, output_dir):
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        model = self.isa.model
        data = model.data
        trace_out = model.trace
        pilot_out = model.pilot
        cloister_out = model.cloister
        pythia_out = model.pythia

        instance_ids = df_original['instance'].astype(str).values

        # --- Footprint (poligoni best/good per algoritmo) ---
        num_algorithms = data.y.shape[1]
        for i in range(num_algorithms):
            for kind in ["best", "good"]:
                region = getattr(trace_out, kind)[i]
                if region is not None and region.polygon is not None:
                    boundaries = np.empty((1, 2))
                    poly = region.polygon
                    polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
                    for p in polys:
                        x, y = p.exterior.xy
                        boundaries = np.concatenate((boundaries, np.array([x, y]).T))
                    boundaries = boundaries[1:-1, :]
                    algo_label = data.algo_labels[i]
                    _write_array_to_csv(
                        boundaries, pd.Series(["z_1", "z_2"]),
                        _make_bind_labels(boundaries),
                        out / f"footprint_{algo_label}_{kind}.csv",
                    )

        # --- Coordinate Z (spazio delle istanze) ---
        _write_array_to_csv(
            pilot_out.z, pd.Series(["z_1", "z_2"]), data.inst_labels,
            out / "coordinates.csv",
        )

        # --- Bordi teorici dello spazio (CLOISTER) ---
        if cloister_out is not None:
            _write_array_to_csv(
                cloister_out.z_edge, pd.Series(["z_1", "z_2"]),
                _make_bind_labels(cloister_out.z_edge), out / "bounds.csv",
            )
            _write_array_to_csv(
                cloister_out.z_ecorr, pd.Series(["z_1", "z_2"]),
                _make_bind_labels(cloister_out.z_ecorr), out / "bounds_prunned.csv",
            )

        # --- Feature RAW: prese dal df originale, bypassando data.x_raw/idx ---
        df_feat_raw = df_original[data.feat_labels].copy()
        df_feat_raw.insert(0, "instance", instance_ids)
        df_feat_raw.to_csv(out / "feature_raw.csv", index=False)

        # --- Feature processate (gia' verificato corretto: 7==7) ---
        _write_array_to_csv(
            data.x, pd.Series(data.feat_labels), data.inst_labels,
            out / "feature_process.csv",
        )

        # --- Algoritmi / performance ---
        _write_array_to_csv(data.y_raw, pd.Series(data.algo_labels), data.inst_labels, out / "algorithm_raw.csv")
        _write_array_to_csv(data.y, pd.Series(data.algo_labels), data.inst_labels, out / "algorithm_process.csv")
        _write_array_to_csv(data.y_bin, pd.Series(data.algo_labels), data.inst_labels, out / "algorithm_bin.csv")
        _write_array_to_csv(data.num_good_algos, pd.Series(["NumGoodAlgos"]), data.inst_labels, out / "good_algos.csv")
        _write_array_to_csv(data.beta, pd.Series(["IsBetaEasy"]), data.inst_labels, out / "beta_easy.csv")
        _write_array_to_csv(data.p, pd.Series(["Best_Algorithm"]), data.inst_labels, out / "portfolio.csv")
        _write_array_to_csv(pythia_out.y_hat, pd.Series(data.algo_labels), data.inst_labels, out / "algorithm_svm.csv")
        _write_array_to_csv(pythia_out.selection0, pd.Series(["Best_Algorithm"]), data.inst_labels, out / "portfolio_svm.csv")

        # --- Tabelle riassuntive ---
        trace_summary = trace_out.summary.iloc[:, [0, 2, 4, 5, 7, 9, 10]].copy()
        trace_summary.rename(columns={"Algorithm": "Row"}, inplace=True)
        trace_summary.to_csv(out / "footprint_performance.csv", index=False)

        if pilot_out.summary is not None:
            pilot_out.summary.to_csv(out / "projection_matrix.csv", index=False)

        pythia_out.summary.rename(columns={"Algorithms": "Row"}).to_csv(
            out / "svm_table.csv", index=False,
        )

        print(f"-> Tutti i CSV salvati (bug del pacchetto aggirati) in: {out}")



