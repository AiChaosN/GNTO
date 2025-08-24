"""High level interface tying together the GNTO pipeline components."""

from __future__ import annotations

from typing import Dict, Any

from DataPreprocessor import DataPreprocessor
from Encoder import Encoder
from TreeModel import TreeModel
from Predictioner import Predictioner


class GNTO:
    """A minimal end-to-end pipeline.

    The class wires the individual components so that ``run`` can be called with
    a raw plan dictionary and returns a scalar prediction.  Each component can be
    replaced by a custom implementation which mirrors how the full project is
    expected to behave.
    """

    def __init__(self,
                 preprocessor: DataPreprocessor | None = None,
                 encoder: Encoder | None = None,
                 tree_model: TreeModel | None = None,
                 predictioner: Predictioner | None = None) -> None:
        self.preprocessor = preprocessor or DataPreprocessor()
        self.encoder = encoder or Encoder()
        self.tree_model = tree_model or TreeModel()
        self.predictioner = predictioner or Predictioner()

    # ------------------------------------------------------------------ pipeline --
    def run(self, plan: Dict[str, Any]) -> float:
        """Execute the end-to-end pipeline on ``plan``."""

        structured = self.preprocessor.preprocess(plan)
        encoded = self.encoder.encode(structured)
        vector = self.tree_model.forward([encoded])
        return self.predictioner.predict(vector)
