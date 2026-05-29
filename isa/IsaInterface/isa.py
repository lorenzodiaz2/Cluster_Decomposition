from typing import Union, List, Dict

from instancespace import InstanceSpace, Metadata
from instancespace.data import options as isa_options
from instancespace.stages.cloister import CloisterStage
from instancespace.stages.pilot import PilotStage
from instancespace.stages.prelim import PrelimStage
from instancespace.stages.preprocessing import PreprocessingStage
from instancespace.stages.pythia import PythiaStage
from instancespace.stages.sifted import SiftedStage
from instancespace.stages.trace import TraceStage
from dataclasses import replace
import pandas as pd
import sys
import os
from pathlib import Path

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