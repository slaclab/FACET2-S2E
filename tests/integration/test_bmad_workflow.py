"""
Tests with a real Tao for the Bmad workflow functions: lattice queries, energy tuning and dipoles,
tracking runs, DAQ-database lattice edits (with a fake DATASET) and get_tao_from_experiment.

Every test works in a temporary copy of the repository layout (bmad/ and setLattice_configs/ are
symlinked, beams/ and temp_beam/ are real folders), so nothing is written into the repository.
pytao keeps one global Tao state, so each test initializes its own Tao (about 1 s).
Tests that take more than ~10 s (whole-linac tracking, full get_tao_from_experiment runs) are marked slow.
"""

import os
import sys
from collections import defaultdict
from pathlib import Path
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from pmd_beamphysics import ParticleGroup

import FACET2_S2E as qs
from FACET2_S2E.simulators.bmad import experiment as exp
from FACET2_S2E.simulators.bmad.energy import default_bend_fields, save_dipoles, treat_dipoles
from FACET2_S2E.simulators.bmad.lattice.linac import matchStringWrapper

pytestmark = pytest.mark.requires_tao

REPO = Path(__file__).resolve().parents[2]
CONFIG = "setLattice_configs/2024-12-09_oneBunch_CSR-on_optimized.yml"
# RF steps per structure for these tests, instead of the default 100 (the lattice's 1000/5000 made the suite take
# ~55 min instead of ~5); the functions under test do not depend on the step count.
N_RF_STEPS = 10
IMPACT_BEAM = "beams/2024-12-11_Impact_OneBunch/2024-12-11_oneBunch"


@pytest.fixture(scope="module")
def s2e_root(tmp_path_factory):
    root = tmp_path_factory.mktemp("s2e_root")
    (root / "bmad").symlink_to(REPO / "bmad")
    (root / "setLattice_configs").symlink_to(REPO / "setLattice_configs")
    (root / IMPACT_BEAM).parent.mkdir(parents=True)
    (root / f"{IMPACT_BEAM}.h5").symlink_to(REPO / f"{IMPACT_BEAM}.h5")
    (root / "temp_beam").mkdir()
    return str(root)


@pytest.fixture
def tao(s2e_root):
    return qs.initializeTao(filePath=s2e_root, loadCustomLatticeTF=True, latticeFile=CONFIG, verbose=False,
                            autoLoadActiveFile=False, csrTF=False, lscTF=False, sr_wakes_on=False, lr_wakes_on=False,
                            n_rf_steps=N_RF_STEPS)


def P0C(tao, ele):
    return tao.ele_gen_attribs(ele)["P0C"]


def gaussian_beam_file(tao, root, name="test", N=300, at="L0AFEND"):
    np.random.seed(0)
    path = f"{root}/temp_beam/{name}"
    qs.make_simple_bunch_standalone(N=N, meanPzMeV=P0C(tao, at)*1e-6, moments=[1e-4, 1e-5, 1e-4, 1e-5, 3e-4, 0],
                                    charge=1e-9, save_path=path)
    return path


def mean_pz_after_tracking(tao, beam_file, start, finish):
    qs.set_beam(tao, beam_file)
    qs.run_initialized_sim(tao, start, finish, locations=None)
    return np.mean(qs.getBeamAtElement(tao, finish, tToZ=False).pz)


class FakeDataset:
    """Stands in for facet2_daq.DATASET: dataset._data["scalars"][list][pv] -> array of shots."""

    def __init__(self):
        self._data = {"scalars": defaultdict(lambda: defaultdict(lambda: np.array([0.0, 0.0])))}

    def set(self, entry, value):
        self._data["scalars"][entry[0]][entry[1]] = np.array([value, value])

    def get(self, entry):
        return self._data["scalars"][entry[0]][entry[1]]


def make_fake_dataset(tao):
    """A DAQ dataset close to the present lattice, with a few distinct, recognizable values."""
    ds = FakeDataset()
    for k, entry in exp.bmad_quad_to_pv_map.items():
        ds.set(entry, qs.getQuadkG(tao, k) - (np.mean(ds.get(exp.bmad_quad_to_pv_map_boost[k])) if k in exp.bmad_quad_to_pv_map_boost else 0))
    ds.set(exp.bmad_quad_to_pv_map["QA10361"], qs.getQuadkG(tao, "QA10361") + 0.1)
    ds.set(exp.bmad_quad_to_pv_map_boost['Q1EL'], np.mean(ds.get(exp.bmad_quad_to_pv_map_boost['Q1EL'])) + 0.5)
    for i, (k, entry) in enumerate(exp.bmad_sextupoles_to_pv_map.items()):
        ds.set(entry, qs.getSextkG(tao, k) + 1.0)
    for i in range(4):
        ds.set(exp.sextupole_offsets_x_from_daq[i], 0.1*(1 + i))     # mm: s1l, s2l, s2r, s1r
        ds.set(exp.sextupole_offsets_y_from_daq[i], -0.1*(1 + i))
    ds.set(["nonBSA_List_S10RF", 'KLYS_LI10_31_SFB_PDES'], 20 + 360*tao.ele_gen_attribs('L0AF')['PHI0'])
    ds.set(["nonBSA_List_S10RF", 'KLYS_LI10_31_ADES'], tao.ele_gen_attribs('L0AF')['VOLTAGE']/2.864664e6)
    ds.set(["nonBSA_List_S10RF", 'KLYS_LI10_41_SFB_PDES'], -15.0)
    ds.set(["nonBSA_List_S10RF", 'KLYS_LI10_41_ADES'], tao.ele_gen_attribs('L0BF')['VOLTAGE']/2.864664e6)
    ds.set(["nonBSA_List_LINAC_KLYS", 'KLYS_LI11_11_SSSB_PDES'], -20.0)
    ds.set(["nonBSA_List_LINAC_KLYS", 'LI14_SBST_1_PHAS'], -38.0)
    ds.set(["nonBSA_List_LINAC_KLYS", 'LI19_SBST_1_PHAS'], 0.0)
    ds.set(exp.bmad_corrector_to_pv_map['XC10721'], 2e-3)                # kG m
    ds.set(exp.bmad_corrector_to_pv_map_before_dogleg['XC10121'], 3e-4)
    for k, entry in exp.bmad_bend_to_pv_map.items():          # GeV
        ds.set(entry, 0.125 if k in ("BX10661", "BX10751") else 0.335 if k.startswith("BCX11") else 10.0)
    return ds


class TestLatticeQueries:
    def test_get_element_array(self, tao):
        bends = qs.get_element_array(tao, "BX0FBEG", "BX0FEND", values_to_show=["SBend"])
        assert set(bends[:, 1]) >= {"BX10661", "BX10751"}
        assert set(bends[:, 2]) == {"SBend"}
        everything = qs.get_element_array(tao, "BX0FBEG", "BX0FEND")
        assert everything[0, 1] == "BX0FBEG" and everything[-1, 1] == "BX0FEND"
        wider = qs.get_element_array(tao, "BX0FBEG", "BX0FEND", marginl=1, marginr=1)
        assert len(wider) == len(everything) + 2

    def test_get_element_array_show_and_remove(self, tao):
        shown = qs.get_element_array(tao, "BX0FBEG", "BX0FEND", values_to_show=["SBend", "Quadrupole"],
                                     values_to_remove=["Quadrupole"])
        assert set(shown[:, 2]) == {"SBend"}

    def test_get_rij_matches_tao_matrix(self, tao):
        mat6 = tao.matrix("BC11CBEG", "BC11CEND")["mat6"]
        for i, j in [(1, 1), (1, 2), (5, 6), (6, 6)]:
            assert qs.get_rij(tao, "BC11CBEG", "BC11CEND", i, j) == pytest.approx(mat6[i-1][j-1], rel=1e-5, abs=1e-12)

    def test_get_tijk_is_the_coefficient_of_delta_squared(self, tao, s2e_root):
        """z_out = R56 delta + T566 delta^2 (Bmad Taylor maps hold monomial coefficients, no 1/2)."""
        a, b = "BC11CBEG", "BC11CEND"
        p0 = P0C(tao, a)
        d = np.linspace(-0.02, 0.02, 9)
        n = len(d)
        ParticleGroup(data=dict(x=np.zeros(n), px=np.zeros(n), y=np.zeros(n), py=np.zeros(n), z=np.zeros(n),
                                t=np.zeros(n), pz=p0*(1 + d), status=np.ones(n, dtype=int), weight=np.full(n, 1e-15),
                                species='electron')).write(f"{s2e_root}/temp_beam/delta.h5")
        qs.set_beam(tao, f"{s2e_root}/temp_beam/delta", timeCenterTF=False)
        qs.run_initialized_sim(tao, a, b, locations=None)
        out = qs.getBeamAtElement(tao, b, tToZ=False)
        t = out.t[np.argsort(out.pz)]
        z = -299792458*(t - t[n//2])
        t566, r56, _ = np.polyfit(d, z, 2)
        assert qs.get_rij(tao, a, b, 5, 6) == pytest.approx(r56, rel=2e-3)
        assert qs.get_tijk(tao, a, b, 5, 6, 6) == pytest.approx(t566, rel=2e-3)
        assert qs.get_tijk(tao, a, b, 5, 6, 1) == qs.get_tijk(tao, a, b, 5, 1, 6)

    @pytest.mark.slow
    def test_second_order_bunch_length(self, tao, s2e_root):
        """Fully compressed by R56: what is left is T566 delta^2, which the 2nd-order theory must reproduce."""
        a, b = "BC11CBEG", "BC11CEND"
        p0 = P0C(tao, a)
        r56 = qs.get_rij(tao, a, b, 5, 6)
        sd = 0.01
        sz = r56*sd
        sigma = np.zeros((6, 6))
        sigma[0, 0] = sigma[2, 2] = (50e-6)**2
        sigma[1, 1] = sigma[3, 3] = (5e-6)**2
        sigma[4, 4], sigma[5, 5], sigma[4, 5] = sz**2, (sd*p0)**2, -sz*sd*p0   # delta = -z/R56
        sigma[5, 4] = sigma[4, 5]
        np.random.seed(1)
        qs.make_bunch(3000, sigma, [0, 0, 0, 0, 0, p0], save_path=f"{s2e_root}/temp_beam/chirped")
        size_sim, size_2nd = qs.make_comparison_dz_2nd_order(tao, beam_file=f"{s2e_root}/temp_beam/chirped", start=a, finish=b)
        _, size_1st = qs.make_comparison_dz_2nd_order(tao, beam_file=f"{s2e_root}/temp_beam/chirped", start=a, finish=b,
                                                      second_order=False)
        t566 = qs.get_tijk(tao, a, b, 5, 6, 6)
        assert size_sim == pytest.approx(abs(t566)*np.sqrt(2)*sd**2, rel=0.15)
        assert size_2nd == pytest.approx(size_sim, rel=0.05)
        assert size_1st < 0.2*size_sim


class TestMagnets:
    def test_sextupole_offsets(self, tao):
        qs.setAllWChicaneSextupolesXOffsets(tao, 1e-3, 2e-3, 3e-3, 4e-3, 5e-3, 6e-3)
        qs.setAllWChicaneSextupolesYOffsets(tao, -1e-3, -2e-3, -3e-3, -4e-3, -5e-3, -6e-3)
        expected = {"S1EL": 1e-3, "S2EL": 2e-3, "S3EL_1": 3e-3, "S3EL_2": 3e-3, "S3ER_1": 4e-3, "S3ER_2": 4e-3,
                    "S2ER": 5e-3, "S1ER": 6e-3}
        for ele, dx in expected.items():
            assert tao.ele_gen_attribs(ele)["X_OFFSET"] == pytest.approx(dx), ele
            assert tao.ele_gen_attribs(ele)["Y_OFFSET"] == pytest.approx(-dx), ele

    def test_fix_magnet_fields(self, tao):
        """After fixMagnetFields an energy change keeps the fields (K follows P0C); by default Bmad keeps K instead."""
        def state():
            return (qs.getQuadkG(tao, "Q1FF"), qs.getSextkG(tao, "S1EL"), tao.ele_gen_attribs("Q1FF")["K1"], P0C(tao, "Q1FF"))
        qs.tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        b0, s0, k0, p0 = state()
        qs.tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 9800])
        b1, _, k1, p1 = state()
        assert k1 == pytest.approx(k0, rel=1e-9) and b1 == pytest.approx(b0*p1/p0, rel=1e-9)   # default: K kept
        qs.tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        qs.fixMagnetFields(tao)
        b0, s0, k0, p0 = state()
        qs.tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 9800])
        b2, s2, k2, p2 = state()
        assert p2/p0 == pytest.approx(0.98, rel=1e-3)
        assert b2 == pytest.approx(b0, rel=1e-9) and s2 == pytest.approx(s0, rel=1e-9)
        assert k2 == pytest.approx(k0*p0/p2, rel=1e-9)
        tao.cmd("set ele Q1FF K1 = -0.5")          # K can still be set: the field follows
        assert qs.getQuadkG(tao, "Q1FF") == pytest.approx(b2*(-0.5)/k2, rel=1e-9)

    def test_collective_effect_settings(self, tao):
        qs.applyBMADCollectiveEffectSettings(tao, csrTF=True, lscTF=False, bmad_grid_size=[16, 16, 32],
                                             sr_wakes_on=True, lr_wakes_on=False, verbose=False, n_bin=40)
        com = tao.bmad_com()
        assert com["csr_and_space_charge_on"] and com["sr_wakes_on"] and not com["lr_wakes_on"]
        sc = tao.space_charge_com()
        assert list(sc["space_charge_mesh_size"]) == [16, 16, 32]
        assert sc["n_bin"] == 40
        qs.applyBMADCollectiveEffectSettings(tao, verbose=False, sr_wakes_on=False, lr_wakes_on=False)
        assert not tao.bmad_com()["csr_and_space_charge_on"]


class TestEnergy:
    def test_tune_to_P0Cs(self, tao):
        qs.tune_to_P0Cs(tao, desired_P0Cs_MeV=[124, 340, None, 9900])
        # one-step linear rescaling of the cavities: a few ppm off
        assert P0C(tao, "BX0FBEG") == pytest.approx(124e6, rel=1e-4)
        assert P0C(tao, "BC11CBEG") == pytest.approx(340e6, rel=1e-4)
        assert P0C(tao, "ENDL2F") == pytest.approx(4500e6, rel=1e-4)
        assert P0C(tao, "ENDL3F_2") == pytest.approx(9900e6, rel=1e-4)

    def test_tune_to_P0Cs_only_L0B(self, tao):
        l0a = tao.ele_gen_attribs("L0AF")["VOLTAGE"]
        qs.tune_to_P0Cs(tao, desired_P0Cs_MeV=[124, 335, 4500, 10000], change_only_L0B=True)
        assert tao.ele_gen_attribs("L0AF")["VOLTAGE"] == l0a
        assert P0C(tao, "BX0FBEG") == pytest.approx(124e6, rel=1e-4)

    @pytest.mark.slow
    def test_save_and_treat_dipoles(self, tao):
        p0c_before = P0C(tao, "BX0FBEG")
        fields = save_dipoles(tao, [125, 335, 4500, 10000])
        for name, value in default_bend_fields.items():
            key = name if name in fields else name + "#1"   # split bends are saved per slice (B2LE#1, B2LE#2)
            assert fields[key] == pytest.approx(value, rel=1e-3), name   # the stored table is the nominal one
        assert P0C(tao, "BX0FBEG") == pytest.approx(p0c_before, rel=1e-4)  # energies restored
        qs.tune_to_P0Cs(tao, desired_P0Cs_MeV=[120, 320, 4400, 9800])
        field = lambda name: tao.ele_gen_attribs(name)["B_FIELD"] + tao.ele_gen_attribs(name)["DB_FIELD"]
        assert abs(field("BX10661")/fields["BX10661"] - 1) > 0.02       # bends followed the lower energy
        treat_dipoles(tao, fields)
        for name in ("BX10661", "BCX11314", "BCX14720", "B1LE"):
            assert field(name) == pytest.approx(fields[name], rel=1e-9), name


class TestRuns:
    def test_set_beam_and_run(self, tao, s2e_root):
        qs.set_beam(tao, f"{s2e_root}/{IMPACT_BEAM}", numMacroParticles=300)
        e = ParticleGroup(f"{s2e_root}/{IMPACT_BEAM}_e.h5")
        assert len(e) == 300 and np.all(e.z == 0) and np.mean(e.t) == pytest.approx(0, abs=1e-20)
        qs.run_initialized_sim(tao, "L0AFEND", "BX0FBEG")
        for loc in ("L0AFEND", "BX0FBEG"):
            assert os.path.exists(f"{s2e_root}/temp_beam/{loc}temp.h5")
        start = ParticleGroup(f"{s2e_root}/temp_beam/L0AFENDtemp.h5")
        out = ParticleGroup(f"{s2e_root}/temp_beam/BX0FBEGtemp.h5")
        assert len(out) == 300
        # the raw IMPACT beam is not at the lattice energy (get_tao_from_experiment rescales it); compare the gain
        gain = P0C(tao, "BX0FBEG") - P0C(tao, "L0AFEND")
        assert np.mean(out.pz) - np.mean(start.pz) == pytest.approx(gain, rel=0.02)

    def test_more_particles_than_in_file(self, tao, s2e_root):
        f = gaussian_beam_file(tao, s2e_root, N=100)
        qs.set_beam(tao, f, numMacroParticles=5e4)
        assert len(ParticleGroup(f + "_e.h5")) == 100

    def test_edit_energy_based_on_beam_inj(self, tao, s2e_root):
        f = gaussian_beam_file(tao, s2e_root)
        mean_pz_after_tracking(tao, f, "L0AFEND", "BX0FBEG")
        qs.edit_energy_based_on_beam_inj(tao, desiredPzMeV=124, change_only_L0B=True)
        assert mean_pz_after_tracking(tao, f, "L0AFEND", "BX0FBEG") == pytest.approx(124e6, rel=1e-4)

    def test_edit_energy_based_on_beam_L1(self, tao, s2e_root):
        f = gaussian_beam_file(tao, s2e_root)
        mean_pz_after_tracking(tao, f, "L0AFEND", "BC11CBEG")
        qs.edit_energy_based_on_beam_L1(tao, desiredPzMeV=330)
        assert mean_pz_after_tracking(tao, f, "L0AFEND", "BC11CBEG") == pytest.approx(330e6, rel=1e-4)


class TestExperimentDatabase:
    def test_daq_scan_path(self, monkeypatch):
        monkeypatch.delenv("FACET2_DAQ_ROOT", raising=False)
        with pytest.raises(ValueError, match="FACET2_DAQ_ROOT"):
            exp.daq_scan_path("BEAMPHYS", "/2026/20260121")
        assert exp.daq_scan_path("BEAMPHYS", "/2026/20260121", daq_root="/daq") == "/daq/BEAMPHYS/2026/20260121"
        monkeypatch.setenv("FACET2_DAQ_ROOT", "/env/daq")
        assert exp.daq_scan_path("BEAMPHYS", "/2026/20260121") == "/env/daq/BEAMPHYS/2026/20260121"
        assert exp.daq_scan_path("BEAMPHYS", "/2026/20260121", daq_root="/daq") == "/daq/BEAMPHYS/2026/20260121"

    def test_cavity_getters(self):
        ds = FakeDataset()
        ds.set(["nonBSA_List_S10RF", 'KLYS_LI10_31_SFB_PDES'], 5.0)
        ds.set(["nonBSA_List_S10RF", 'KLYS_LI10_31_ADES'], 2.0)
        ds.set(["nonBSA_List_S10RF", 'KLYS_LI10_41_SFB_PDES'], -15.0)
        ds.set(["nonBSA_List_S10RF", 'KLYS_LI10_41_ADES'], 3.0)
        assert qs.get_l0a_phase(ds) == pytest.approx(-15.0)          # FACET phase offset of -20 deg
        assert qs.get_l0a_ampl(ds) == pytest.approx(2*2.864664e6)
        assert qs.get_l0b_phase(ds) == pytest.approx(-15.0)
        assert qs.get_l0b_ampl(ds) == pytest.approx(3*2.864664e6)

    def test_dipole_energies(self, tao):
        assert exp.save_dipole_energies_from_the_DAQ_database(make_fake_dataset(tao)) == \
            pytest.approx([125, 335, 4500, 10000])

    def test_edit_energy(self, tao):
        qs.edit_energy_tao_based_on_experiment_database(tao, make_fake_dataset(tao))
        assert 360*tao.ele_gen_attribs("L0BF")["PHI0"] == pytest.approx(-15.0)
        for ele, e in (("BX10661", 125e6), ("BC11CBEG", 335e6), ("ENDL2F", 4500e6), ("ENDL3F_2", 10000e6)):
            assert P0C(tao, ele) == pytest.approx(e, rel=1e-4), ele

    def test_edit_lattice(self, tao):
        ds = make_fake_dataset(tao)
        quads = {k: qs.getQuadkG(tao, k) for k in exp.bmad_quad_to_pv_map}
        xc_before = tao.ele_gen_attribs("XC10121")["BL_KICK"]
        qs.edit_tao_based_on_experiment_database(tao, ds, correctors_coef=-0.1)
        assert qs.getQuadkG(tao, "QA10361") == pytest.approx(quads["QA10361"] + 0.1, abs=1e-6)
        assert qs.getQuadkG(tao, "Q1EL") == pytest.approx(quads["Q1EL"] + 0.5, abs=1e-6)   # base + boost PV
        assert qs.getQuadkG(tao, "QE10425") == pytest.approx(quads["QE10425"], abs=1e-6)
        for k, entry in exp.bmad_sextupoles_to_pv_map.items():
            assert qs.getSextkG(tao, k) == pytest.approx(np.mean(ds.get(entry)), abs=1e-6), k
        # movers in mm -> m; S3 has no mover
        for ele, dx in (("S1EL", 1e-4), ("S2EL", 2e-4), ("S2ER", 3e-4), ("S1ER", 4e-4), ("S3EL_1", 0), ("S3ER_2", 0)):
            assert tao.ele_gen_attribs(ele)["X_OFFSET"] == pytest.approx(dx), ele
            assert tao.ele_gen_attribs(ele)["Y_OFFSET"] == pytest.approx(-dx), ele
        assert tao.ele_gen_attribs("XC10721")["BL_KICK"] == pytest.approx(-2e-4)       # kG m -> T m
        assert tao.ele_gen_attribs("XC10121")["BL_KICK"] == xc_before        # before the dogleg: off by default

    def test_correctors_from_beginning(self, tao):
        qs.edit_tao_based_on_experiment_database(tao, make_fake_dataset(tao), correctors_coef=-0.1, correctors_from_beg=True)
        assert tao.ele_gen_attribs("XC10121")["BL_KICK"] == pytest.approx(-3e-5)


class TestPerKlystronRF:
    def per_klystron_dataset(self, tao):
        """Fake DAQ with the June-2026 lists: SBST PDES per sector, klystron PDES/PHAS/AMPL."""
        ds = make_fake_dataset(tao)
        for sec in range(11, 20):
            ds.set(["nonBSA_List_LINAC_SBST", f"LI{sec}_SBST_1_PDES"], -50.0 if 12 <= sec <= 14 else 0.0)
            lst = "nonBSA_List_LINAC_L2KLYS" if sec < 15 else "nonBSA_List_LINAC_L3KLYS"
            for k in range(1, 9):
                ds.set([lst, f"LI{sec}_KLYS_{k}1_PDES"], 0.0)
                ds.set([lst, f"LI{sec}_KLYS_{k}1_PHAS"], 0.0)
                ds.set([lst, f"LI{sec}_KLYS_{k}1_AMPL"], 60.0)
        ds.set(["nonBSA_List_LINAC_SBST", "KLYS_LI11_11_SSSB_PDES"], -18.0)
        ds.set(["nonBSA_List_LINAC_L2KLYS", "LI12_KLYS_31_AMPL"], 0.0)          # off
        ds.set(["nonBSA_List_LINAC_L2KLYS", "LI13_KLYS_21_AMPL"], 120.0)        # twice the voltage
        ds.set(["nonBSA_List_LINAC_L3KLYS", "LI19_KLYS_41_PDES"], -70.0)        # phase-split pair
        ds.set(["nonBSA_List_LINAC_L3KLYS", "LI19_KLYS_51_PDES"], 70.0)
        return ds

    def test_phases_amplitudes_and_energy(self, tao):
        ds = self.per_klystron_dataset(tao)
        assert exp.has_per_klystron_rf(ds)
        qs.edit_energy_tao_based_on_experiment_database(tao, ds)
        ph = lambda e: 360*tao.ele_gen_attribs(e)["PHI0"]
        V = lambda k: sum(tao.ele_gen_attribs(c)["VOLTAGE"] for c in map(str, matchStringWrapper(tao, "L2")) if c.startswith(k))
        assert ph("K11_4A") == pytest.approx(0.0) and ph("K12_1A") == pytest.approx(-50.0)
        assert ph("K19_4A") == pytest.approx(-70.0) and ph("K19_5A") == pytest.approx(70.0)
        assert tao.ele_gen_attribs("K12_3A")["GRADIENT"] == 0
        assert V("K13_2") == pytest.approx(2*V("K13_1"), rel=1e-9)
        assert "12-3" in tao.daq_rf["L2"]["off"]
        assert P0C(tao, "ENDL2F") == pytest.approx(4.5e9, rel=1e-3)
        assert P0C(tao, "ENDL3F_2") == pytest.approx(10e9, rel=1e-3)
        assert 360*tao.ele_gen_attribs(matchStringWrapper(tao, "L1")[0])["PHI0"] == pytest.approx(-18.0)

    def test_scale_keeps_ratios(self, tao):
        names = list(map(str, matchStringWrapper(tao, "L2")))
        tao.cmd(f"set ele {names[0]} GRADIENT = {2*tao.ele_gen_attribs(names[1])['GRADIENT']}")
        r0 = tao.ele_gen_attribs(names[0])["GRADIENT"]/tao.ele_gen_attribs(names[1])["GRADIENT"]
        qs.scaleLinacGradientAuto(tao, "L2", 4.0e9)
        assert tao.ele_gen_attribs(names[0])["GRADIENT"]/tao.ele_gen_attribs(names[1])["GRADIENT"] == pytest.approx(r0)
        assert P0C(tao, "ENDL2F") - P0C(tao, "BEGL2F") == pytest.approx(4.0e9, rel=1e-4)


class TestL2VirtualMatch:
    def test_restores_the_match_with_the_non_daq_quads(self, tao):
        k_daq = tao.ele_gen_attribs("QM11393")["K1"]
        tao.cmd(f"set ele QM11393 K1 = {1.2*k_daq}")         # a mismatch upstream of the matching quads
        before = {q: tao.ele_gen_attribs(q)["K1"] for q in qs.L2_MATCH_QUADS}
        result = qs.l2_virtual_match(tao)
        assert max(result["bmag_before"]) > 1.5
        assert max(result["bmag_after"]) < 1.1     # the regularization leaves ~1.07 with the 2026 design targets
        assert tao.ele_gen_attribs("QM11393")["K1"] == pytest.approx(1.2*k_daq)   # only the listed quads change
        for q, f in result["factors"].items():
            assert tao.ele_gen_attribs(q)["K1"] == pytest.approx(f*before[q], rel=1e-9)
            assert 0.5 <= f <= 2.0


class TestRFSteps:
    def test_override_default_and_lattice_values(self, tao, s2e_root):
        cavities = ("L0BF", "K11_4A", "K15_1A")
        assert {tao.ele_gen_attribs(e)["N_RF_STEPS"] for e in cavities} == {N_RF_STEPS}
        tao = qs.initializeTao(filePath=s2e_root, verbose=False, autoLoadActiveFile=False)
        assert {tao.ele_gen_attribs(e)["N_RF_STEPS"] for e in cavities} == {100}
        tao = qs.initializeTao(filePath=s2e_root, verbose=False, autoLoadActiveFile=False, n_rf_steps=None)
        assert tao.ele_gen_attribs("L0BF")["N_RF_STEPS"] == 5000       # facet2-lattice: 5000 in L0 and LI11
        assert tao.ele_gen_attribs("K15_1A")["N_RF_STEPS"] == 1000      # 1000 elsewhere


class TestLatticeTwiss:
    def test_design_twiss_at_PR10571_is_golden(self, tao):
        # facet2-lattice (2026 optics) pins the design Twiss at MRK0F (fixer_mrk0f): PR10571 is then the golden match
        tw = tao.ele_twiss("PR10571", which="design")
        g = qs.GOLDEN_TWISS_PR10571
        for plane, mode in (("x", "a"), ("y", "b")):
            assert tw[f"beta_{mode}"] == pytest.approx(g[f"beta_{plane}"], rel=1e-3)
            assert tw[f"alpha_{mode}"] == pytest.approx(g[f"alpha_{plane}"], abs=2e-3)

    def test_set_lattice_twiss_overrides_the_fixer(self, tao):
        # targets that are not the golden match: the fixer at MRK0F would otherwise keep the golden Twiss everywhere
        targets = qs.golden_twiss_targets(tao, "L0AFEND")
        targets = {**targets, "beta_x": 2*targets["beta_x"], "alpha_y": targets["alpha_y"] + 1.0}
        qs.set_lattice_twiss(tao, "L0AFEND", targets)
        from FACET2_S2E.simulators.bmad.runs import _propagate_twiss
        for loc in ("PR10571", "BEGL1F"):
            want = _propagate_twiss(tao, "L0AFEND", loc, targets)
            tw = tao.ele_twiss(loc)
            assert [tw["beta_a"], tw["alpha_a"], tw["beta_b"], tw["alpha_b"]] == pytest.approx(want, rel=1e-6, abs=1e-6)
        assert tao.ele_twiss("PR10571", which="design")["beta_a"] == pytest.approx(qs.GOLDEN_TWISS_PR10571["beta_x"], rel=1e-3)


class TestBC20SextupoleTune:
    @pytest.mark.slow
    def test_tune_reduces_emittance_and_restores_state(self, tao, s2e_root):
        """A beam 1% below P0C with a 1% energy spread leaves dispersion after the W chicane; the sextupole x offsets
        (within +-2 mm) cancel it. Tao's beam settings are restored afterwards."""
        np.random.seed(1)
        path = f"{s2e_root}/temp_beam/bc20_tune"
        p = 0.99*P0C(tao, "BEGBC20")
        qs.make_simple_bunch_standalone(N=400, meanPzMeV=p*1e-6, moments=[30e-6, 3e-6, 30e-6, 3e-6, 50e-6, 0.01*p],
                                        charge=1.6e-9, save_path=path)
        beam_file, track_type = tao.beam_init(0)["position_file"], tao.tao_global()["track_type"]
        result = qs.tune_bc20_sextupole_offsets(tao, beam=ParticleGroup(path + ".h5"), n_particles=400)
        assert result["eps_n_after"][0] < 0.7*result["eps_n_before"][0]    # 28.9 -> 17.3 um for this beam
        for name, dx in result["x_offsets_m"].items():
            assert abs(dx) <= 2e-3 + 1e-12
            assert tao.ele_gen_attribs(name)["X_OFFSET"] == pytest.approx(dx)
        assert tao.beam_init(0)["position_file"] == beam_file and tao.tao_global()["track_type"] == track_type


class TestGetTaoFromExperiment:
    @pytest.mark.slow
    def test_generated_beam(self, s2e_root):
        np.random.seed(2)
        tao = qs.get_tao_from_experiment(start="L0AFEND", finish="BX0FBEG", filepath=s2e_root, N_in_simple_bunch=300,
                                         moments=[1e-4, 1e-5, 1e-4, None, 3e-4, None], n_rf_steps=N_RF_STEPS)
        start = ParticleGroup(f"{s2e_root}/temp_beam/L0AFENDtemp.h5")
        assert np.std(start.x) == pytest.approx(1e-4, rel=0.15)
        assert np.std(start.yp) == 0
        out = ParticleGroup(f"{s2e_root}/temp_beam/BX0FBEGtemp.h5")
        assert np.mean(out.pz) == pytest.approx(P0C(tao, "BX0FBEG"), rel=5e-3)

    @pytest.mark.slow
    def test_beam_from_file(self, s2e_root):
        tao = qs.get_tao_from_experiment(start="L0AFEND", finish="BX0FBEG", filepath=s2e_root,
                                         file_ext=f"{s2e_root}/{IMPACT_BEAM}", N_to_use_from_file=300, n_rf_steps=N_RF_STEPS)
        out = ParticleGroup(f"{s2e_root}/temp_beam/BX0FBEGtemp.h5")
        assert len(out) == 300
        assert np.mean(out.pz) == pytest.approx(P0C(tao, "BX0FBEG"), rel=5e-3)

    @pytest.mark.slow
    def test_daq_branch(self, tao, s2e_root):
        ds = make_fake_dataset(tao)
        q_before = qs.getQuadkG(tao, "QA10361")
        k1_before = tao.ele_gen_attribs("QA10361")["K1"]
        with patch.object(exp, "DATASET", lambda *args, **kwargs: ds):
            tao = qs.get_tao_from_experiment(experiment="TEST", scan_number="1", filepath=s2e_root, run=False, daq_root="/daq",
                                             N_in_simple_bunch=100, moments=[1e-4]*6, n_rf_steps=N_RF_STEPS)
            assert qs.getQuadkG(tao, "QA10361") == pytest.approx(q_before + 0.1, abs=1e-6)
            assert 360*tao.ele_gen_attribs("L0BF")["PHI0"] == pytest.approx(-15.0)
            assert hasattr(tao, "l2_match_result")          # l2_match="auto": DAQ magnets loaded
            assert not hasattr(tao, "bc20_tune_result")     # bc20_sextupole_tune="auto": no feedback, no tracked beam
            tao = qs.get_tao_from_experiment(experiment="TEST", scan_number="1", filepath=s2e_root, run=False, daq_root="/daq",
                                             N_in_simple_bunch=100, moments=[1e-4]*6, edit_only_energy_from_exp=True,
                                             n_rf_steps=N_RF_STEPS)
            # quads not loaded: K1 is kept (the kG readback follows the edited energy)
            assert tao.ele_gen_attribs("QA10361")["K1"] == pytest.approx(k1_before, rel=1e-9)
            assert 360*tao.ele_gen_attribs("L0BF")["PHI0"] == pytest.approx(-15.0)
            assert not hasattr(tao, "l2_match_result")      # magnets not loaded

    @pytest.mark.slow
    def test_feedback_keeps_beam_energy_and_l0a(self, s2e_root):
        """energy=-1 + energy_edit_on_beam: the beam keeps its file energy at L0AFEND and only L0B is scaled
        (scaling L0A and rescaling the beam to the new L0AFEND energy mismatched IMPACT beams to the injector quads)."""
        path = f"{s2e_root}/temp_beam/off_energy"
        qs.make_simple_bunch_standalone(N=200, meanPzMeV=66.0, moments=[1e-4, 1e-5, 1e-4, 1e-5, 3e-4, 0], charge=1e-9, save_path=path)
        kw = dict(start="L0AFEND", filepath=s2e_root, file_ext=path, energy=-1, run=False, sr_wakes_on=False, lr_wakes_on=False,
                  n_rf_steps=N_RF_STEPS)
        tao = qs.get_tao_from_experiment(**kw)
        p0c_l0a = P0C(tao, "L0AFEND")
        fields = {q: qs.getQuadkG(tao, q) for q in ("QM11358", "Q14701", "Q1FF")}
        desired = [124e6, 330e6, 4400e6, 9900e6]
        tao = qs.get_tao_from_experiment(**kw, energy_edit_on_beam=True, desired_beam_energies_for_the_feedback=desired)
        assert P0C(tao, "L0AFEND") == pytest.approx(p0c_l0a, rel=1e-12)
        for q, b in fields.items():     # keep_magnet_fields: the feedback changes P0C, not the magnet fields
            assert qs.getQuadkG(tao, q) == pytest.approx(b, rel=1e-9), q
        assert np.mean(ParticleGroup(f"{s2e_root}/temp_beam/temp.h5").pz) == pytest.approx(66e6, rel=1e-3)
        qs.run_initialized_sim(tao, "L0AFEND", "ENDL3F_2", locations=None)
        for loc, e in zip(("BX0FBEG", "BC11CBEG", "ENDL2F", "ENDL3F_2"), desired):
            assert np.mean(qs.getBeamAtElement(tao, loc, tToZ=False).pz) == pytest.approx(e, rel=1e-3), loc

    def test_missing_repository(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No FACET2-S2E lattice"):
            qs.get_tao_from_experiment(filepath=str(tmp_path))


class TestScans:
    def test_make_1d_scan(self):
        calls = []

        def change(state, value, **kwargs):
            calls.append(value)
            return state + [value]

        def result(state, **kwargs):
            return state[-1]**2

        values, output = qs.make_1d_scan([], mean=1.0, nscan=5, scan_span=2.0, function_to_change_tao_in_scan=change,
                                         function_to_get_results_in_scan=result, plot=True)
        assert np.allclose(values, [0, 0.5, 1, 1.5, 2])
        assert np.allclose(output, values**2)
        assert calls == list(values)


@pytest.mark.slow
class TestFullLinac:
    def test_edit_energy_based_on_beam_all(self, tao, s2e_root):
        f = gaussian_beam_file(tao, s2e_root, N=200)
        desired = [124e6, 330e6, 4400e6, 9900e6]
        tao = qs.edit_energy_based_on_beam_all(tao, "L0AFEND", f, desired_beam_energies=desired)
        qs.set_beam(tao, f)
        qs.run_initialized_sim(tao, "L0AFEND", "ENDL3F_2", locations=None)
        for loc, e in zip(("BX0FBEG", "BC11CBEG", "ENDL2F", "ENDL3F_2"), desired):
            assert np.mean(qs.getBeamAtElement(tao, loc, tToZ=False).pz) == pytest.approx(e, rel=1e-3), loc

    def test_run_initialized_sim_edit_bunch_energy(self, tao, s2e_root):
        f = gaussian_beam_file(tao, s2e_root, N=200)
        qs.set_beam(tao, f)
        qs.run_initialized_sim_edit_bunch_energy(tao, "L0AFEND", "BC11CEND", edited_bunch_energy_at_checkpoints_MeV=[None, 330, None, None])
        assert np.mean(qs.getBeamAtElement(tao, "BC11CEND", tToZ=False).pz) == pytest.approx(330e6, rel=1e-4)

    def test_run_initialized_sim_edit_lattice_energy_for_dipoles(self, tao, s2e_root):
        f = gaussian_beam_file(tao, s2e_root, N=200)
        qs.set_beam(tao, f)
        qs.run_initialized_sim_edit_lattice_energy_for_dipoles(tao, "L0AFEND", "BC11CBEG", desired_P0Cs_MeV=[124, None, None, None])
        assert os.path.exists(f"{s2e_root}/temp_beam/BC11CBEGtemp.h5")
        assert np.mean(ParticleGroup(f"{s2e_root}/temp_beam/BC11CBEGtemp.h5").pz) == pytest.approx(335e6, rel=5e-3)
