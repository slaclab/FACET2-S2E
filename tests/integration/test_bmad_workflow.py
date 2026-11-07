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

pytestmark = pytest.mark.requires_tao

REPO = Path(__file__).resolve().parents[2]
CONFIG = "setLattice_configs/2024-12-09_oneBunch_CSR-on_optimized.yml"
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
                            autoLoadActiveFile=False, csrTF=False, lscTF=False, sr_wakes_on=False, lr_wakes_on=False)


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


class TestGetTaoFromExperiment:
    @pytest.mark.slow
    def test_generated_beam(self, s2e_root):
        np.random.seed(2)
        tao = qs.get_tao_from_experiment(start="L0AFEND", finish="BX0FBEG", filepath=s2e_root, N_in_simple_bunch=300,
                                         moments=[1e-4, 1e-5, 1e-4, None, 3e-4, None])
        start = ParticleGroup(f"{s2e_root}/temp_beam/L0AFENDtemp.h5")
        assert np.std(start.x) == pytest.approx(1e-4, rel=0.15)
        assert np.std(start.yp) == 0
        out = ParticleGroup(f"{s2e_root}/temp_beam/BX0FBEGtemp.h5")
        assert np.mean(out.pz) == pytest.approx(P0C(tao, "BX0FBEG"), rel=5e-3)

    @pytest.mark.slow
    def test_beam_from_file(self, s2e_root):
        tao = qs.get_tao_from_experiment(start="L0AFEND", finish="BX0FBEG", filepath=s2e_root,
                                         file_ext=f"{s2e_root}/{IMPACT_BEAM}", N_to_use_from_file=300)
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
                                             N_in_simple_bunch=100, moments=[1e-4]*6)
            assert qs.getQuadkG(tao, "QA10361") == pytest.approx(q_before + 0.1, abs=1e-6)
            assert 360*tao.ele_gen_attribs("L0BF")["PHI0"] == pytest.approx(-15.0)
            tao = qs.get_tao_from_experiment(experiment="TEST", scan_number="1", filepath=s2e_root, run=False, daq_root="/daq",
                                             N_in_simple_bunch=100, moments=[1e-4]*6, edit_only_energy_from_exp=True)
            # quads not loaded: K1 is kept (the kG readback follows the edited energy)
            assert tao.ele_gen_attribs("QA10361")["K1"] == pytest.approx(k1_before, rel=1e-9)
            assert 360*tao.ele_gen_attribs("L0BF")["PHI0"] == pytest.approx(-15.0)

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
