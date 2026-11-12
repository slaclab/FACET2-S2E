"""
Tests for the DAQ loader helpers (simulators/bmad/experiment.py).
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import FACET2_S2E as qs
from FACET2_S2E.simulators.bmad import experiment as exp


class _FakeTao:
    """Only what get_l0a_energy_from_quads needs: design L and K1 of the quads."""
    def __init__(self, design):
        self.design = design

    def ele_gen_attribs(self, ele, which="model"):
        assert which == "design"
        return self.design[ele]


class _FakeDataset:
    def __init__(self, values):
        self._data = {"scalars": {}}
        for k, value in values.items():
            lst, pv = exp.bmad_quad_to_pv_map[k]
            self._data["scalars"].setdefault(lst, {})[pv] = np.array([value, value])


def _bdes_kG(K1, L, p_eV):
    # inverse of setQuadkG for an electron: B1_GRADIENT = -BDES/L/10 [T/m], K1 = -B1_GRADIENT*c/p
    return K1*L*10*p_eV/299792458.0


DESIGN = {"QA10361": {"L": 0.1, "K1": -11.3}, "QA10371": {"L": 0.1, "K1": 11.4}}


def test_l0a_energy_from_quads_inverts_the_field():
    ds = _FakeDataset({k: _bdes_kG(d["K1"], d["L"], 67e6) for k, d in DESIGN.items()})
    assert qs.get_l0a_energy_from_quads(_FakeTao(DESIGN), ds) == pytest.approx(67.0, rel=1e-12)


def test_l0a_energy_from_quads_averages_and_skips_switched_off_quads():
    ds = _FakeDataset({"QA10361": _bdes_kG(-11.3, 0.1, 65e6), "QA10371": _bdes_kG(11.4, 0.1, 69e6)})
    assert qs.get_l0a_energy_from_quads(_FakeTao(DESIGN), ds) == pytest.approx(67.0, rel=1e-12)
    ds = _FakeDataset({"QA10361": 0.0, "QA10371": _bdes_kG(11.4, 0.1, 69e6)})
    assert qs.get_l0a_energy_from_quads(_FakeTao(DESIGN), ds) == pytest.approx(69.0, rel=1e-12)
    ds = _FakeDataset({"QA10361": 0.0, "QA10371": 0.0})
    assert qs.get_l0a_energy_from_quads(_FakeTao(DESIGN), ds) is None


def test_s11_magnets_use_readbacks():
    for pv_map in (exp.bmad_quad_to_pv_map, exp.bmad_corrector_to_pv_map, exp.bmad_bend_to_pv_map):
        for k, (lst, pv) in pv_map.items():
            if lst == "nonBSA_List_S11":
                assert pv.endswith("_BACT"), (k, pv)
