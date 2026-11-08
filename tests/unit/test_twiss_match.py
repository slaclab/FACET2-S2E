"""
Tests for twiss_match_beam (simulators/bmad/runs.py): linear Twiss matching of a beam file.
"""

import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs
from FACET2_S2E.simulators.bmad.runs import _beam_twiss


def write_mismatched_beam(path, n=20000, seed=1):
    """Beam with beta ~80 m, alpha ~-16 in both planes (like the 2024-12-11 IMPACT bunch at L0AFEND)."""
    rng = np.random.default_rng(seed)
    pz = 88e6*(1 + 1e-3*rng.standard_normal(n))
    data = {"t": 1e-12*rng.standard_normal(n), "z": np.zeros(n), "pz": pz, "weight": np.full(n, 1.6e-9/n),
            "status": np.ones(n, dtype=int), "species": "electron"}
    eps, beta, alpha = 2.5e-8, 80.0, -16.0
    for u, pu in (("x", "px"), ("y", "py")):
        a, b = rng.standard_normal(n), rng.standard_normal(n)
        data[u] = np.sqrt(eps*beta)*a
        data[pu] = np.sqrt(eps/beta)*(b - alpha*a)*pz
    ParticleGroup(data=data).write(path + ".h5")


def test_matches_explicit_targets_and_keeps_emittance(tmp_path):
    path = str(tmp_path / "beam")
    write_mismatched_beam(path)
    P0 = ParticleGroup(path + ".h5")
    targets = {"beta_x": 15.3, "alpha_x": 2.18, "beta_y": 18.1, "alpha_y": 1.46}

    assert qs.twiss_match_beam(None, path, "L0AFEND", targets=targets) == targets

    P = ParticleGroup(path + ".h5")
    for plane in ("x", "y"):
        beta, alpha = _beam_twiss(P, plane)
        np.testing.assert_allclose(beta, targets[f"beta_{plane}"], rtol=1e-6)
        np.testing.assert_allclose(alpha, targets[f"alpha_{plane}"], rtol=1e-6)
        np.testing.assert_allclose(P[f"norm_emit_{plane}"], P0[f"norm_emit_{plane}"], rtol=1e-9)
    np.testing.assert_array_equal(P.pz, P0.pz)
    np.testing.assert_array_equal(P.t, P0.t)


def test_matching_twice_changes_nothing(tmp_path):
    path = str(tmp_path / "beam")
    write_mismatched_beam(path)
    targets = {"beta_x": 15.3, "alpha_x": 2.18, "beta_y": 18.1, "alpha_y": 1.46}
    qs.twiss_match_beam(None, path, "L0AFEND", targets=targets)
    P1 = ParticleGroup(path + ".h5")
    qs.twiss_match_beam(None, path, "L0AFEND", targets=targets)
    P2 = ParticleGroup(path + ".h5")
    for u in ("x", "px", "y", "py"):
        np.testing.assert_allclose(getattr(P2, u), getattr(P1, u), rtol=1e-6)   # floating-point level


class _FakeTao:
    """Only what golden_twiss_targets needs: tao.matrix(a, b)["mat6"]."""
    def __init__(self, mat6):
        self.mat6 = mat6

    def matrix(self, start, end):
        return {"mat6": self.mat6}


def test_golden_targets_propagate_onto_golden():
    # accelerating (non-symplectic) transport, det = p0/p1 = 0.55 per plane
    m = np.eye(6)
    m[0:2, 0:2] = np.array([[1.3, 4.2], [-0.05, 0.26]])
    m[2:4, 2:4] = np.array([[0.8, 3.1], [0.12, 1.15]]) * np.sqrt(0.55/np.linalg.det([[0.8, 3.1], [0.12, 1.15]]))
    targets = qs.golden_twiss_targets(_FakeTao(m), "L0AFEND")
    golden = qs.GOLDEN_TWISS_PR10571
    for plane, i in (("x", 0), ("y", 2)):
        b0, a0 = targets[f"beta_{plane}"], targets[f"alpha_{plane}"]
        B0 = np.array([[b0, -a0], [-a0, (1 + a0**2)/b0]])
        M = m[i:i+2, i:i+2]
        B1 = M @ B0 @ M.T / abs(np.linalg.det(M))
        np.testing.assert_allclose(B1[0, 0], golden[f"beta_{plane}"], rtol=1e-10)
        np.testing.assert_allclose(-B1[0, 1], golden[f"alpha_{plane}"], rtol=1e-8, atol=1e-10)


class _FakeTaoLattice:
    """matrix(), cmd("set element beginning ...") and ele_twiss() for one linear map from BEGINNING to any location,
    with the lattice Twiss propagated from the BEGINNING values as M B M^T / |det M| (Bmad, with acceleration)."""
    def __init__(self, mat6):
        self.mat6 = mat6
        self.begin = {"beta_a": 0.139, "alpha_a": 0.95, "beta_b": 0.365, "alpha_b": 2.0}

    def matrix(self, start, end):
        return {"mat6": self.mat6}

    def cmd(self, command):
        if command.startswith("set element beginning"):
            self.begin[command.split()[3]] = float(command.split("=")[1])
        return []

    def ele_twiss(self, location):
        out = {}
        for mode, i in (("a", 0), ("b", 2)):
            b, a = self.begin[f"beta_{mode}"], self.begin[f"alpha_{mode}"]
            M = self.mat6[i:i+2, i:i+2]
            B = M @ np.array([[b, -a], [-a, (1 + a**2)/b]]) @ M.T / abs(np.linalg.det(M))
            out[f"beta_{mode}"], out[f"alpha_{mode}"] = B[0, 0], -B[0, 1]
        return out


def test_set_lattice_twiss_puts_the_targets_at_the_location():
    m = np.eye(6)
    m[0:2, 0:2] = np.array([[2.1, 1.4], [-0.3, 0.14]])        # det 0.714: accelerating
    m[2:4, 2:4] = np.array([[1.7, 2.2], [0.1, 0.55]])
    tao = _FakeTaoLattice(m)
    targets = {"beta_x": 4.3051, "alpha_x": -0.4473, "beta_y": 4.1179, "alpha_y": -0.8119}
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        qs.set_lattice_twiss(tao, "L0AFEND", targets)
    tw = tao.ele_twiss("L0AFEND")
    for plane, mode in (("x", "a"), ("y", "b")):
        np.testing.assert_allclose(tw[f"beta_{mode}"], targets[f"beta_{plane}"], rtol=1e-10)
        np.testing.assert_allclose(tw[f"alpha_{mode}"], targets[f"alpha_{plane}"], rtol=1e-9, atol=1e-12)


def test_set_lattice_twiss_refuses_coupled_optics():
    m = np.eye(6)
    m[0, 2] = 0.1
    with pytest.raises(ValueError):
        qs.set_lattice_twiss(_FakeTaoLattice(m), "L0AFEND", {"beta_x": 4, "alpha_x": 0, "beta_y": 4, "alpha_y": 0})
