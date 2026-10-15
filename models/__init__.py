"""GNTO model zoo.

* ``models.gnto`` - GNTO (ours): node encoder -> GATv2 graph encoder -> readout -> head.
* ``models.qf``   - QueryFormer (Zhao et al., VLDB 2022) tree-transformer, ported from
                    the official QueryFormer ``model/`` package so the baseline trains
                    from this repo with the data under ``data/`` and ``data/qf/``.

Import the sub-package you need (``from models.gnto import GNTO``,
``from models.qf import QueryFormer``); nothing heavy is imported here.
"""
