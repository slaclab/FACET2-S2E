"""Run a tracked simulation on an initialized Tao object, saving beams at checkpoints.

Also holds edit_energy_based_on_beam_all, which iterates simulation runs and
energy retuning (simulation.energy).

Moved from simulationFunctions.py.
"""

import os
import warnings

import numpy as np
from pmd_beamphysics import ParticleGroup
from pmd_beamphysics.statistics import twiss_calc
from scipy.optimize import least_squares, minimize

from ...beam.analysis import calcBMAG
from ...beam.manipulation import modifyInputBeamSimple, edit_bunch_parameters_from_PG, edit_bunch_parameters
from .core import trackBeam, getBeamAtElement
from .energy import treat_dipoles, tune_to_P0Cs, default_bend_fields, edit_energy_based_on_beam_inj, edit_energy_based_on_beam_L1, edit_energy_based_on_beam_L2, edit_energy_based_on_beam_L3


def set_beam(tao, file, numMacroParticles = None, timeCenterTF=True):
    '''
    Sets the beam in the tao object. The beam is edited: drift to z, z=0. t=0 if time_centering.
    '''
    #filePath = os.getcwd()
    #file_e = f'{filePath}/beams/activeBeamFile.h5'
    file_e = file + "_e"
    #Write as the active file
    #modifyInputBeamSimple(folder + file + ".h5", numMacroParticles).write(file_e + ".h5")
    modifyInputBeamSimple(file + ".h5", numMacroParticles, timeCenterTF=timeCenterTF).write(file_e + ".h5")
    #tao.cmd(f'set beam_init position_file={folder + file_e + ".h5"}')
    tao.cmd(f'set beam_init position_file={file_e + ".h5"}')
    tao.cmd('reinit beam')


# Golden injector match (FACET-II operations): Twiss at PR10571, after L0B (see launchTwissCorrection).
# The same Twiss as the design lattice at PR10571: facet2-lattice (2026 optics) pins the design Twiss at MRK0F with
# fixer_mrk0f, which propagates to these values. The 2026-06 emittance scans on PROF:IN10:571 measure BMAG 1.02-1.12
# against them.
GOLDEN_TWISS_PR10571 = {"location": "PR10571", "beta_x": 5.73666431, "alpha_x": -2.14411559,
                        "beta_y": 2.57530302, "alpha_y": 0.01016211}


def golden_twiss_targets(tao, location, golden=GOLDEN_TWISS_PR10571):
    '''
    Twiss parameters at `location` that the present linear optics (Tao mat6 from `location` to golden["location"])
    map onto the golden Twiss. Matching the beam at `location` to these targets reproduces the operational
    injector match at PR10571 with whatever magnet and cavity settings are loaded (e.g. from the DAQ).
    '''
    m = np.array(tao.matrix(location, golden["location"])["mat6"])
    targets = {}
    for plane, i in (("x", 0), ("y", 2)):
        b1, a1 = golden[f"beta_{plane}"], golden[f"alpha_{plane}"]
        B1 = np.array([[b1, -a1], [-a1, (1 + a1**2)/b1]])
        Mi = np.linalg.inv(m[i:i+2, i:i+2])
        B0 = Mi @ B1 @ Mi.T
        B0 /= np.sqrt(np.linalg.det(B0))    # the transport is not symplectic with acceleration: renormalize
        targets[f"beta_{plane}"], targets[f"alpha_{plane}"] = float(B0[0, 0]), float(-B0[0, 1])
    return targets


def set_lattice_twiss(tao, location, targets, verbose=False):
    '''
    Set the Twiss of the lattice's BEGINNING element so that the model Twiss at `location` (tao.ele_twiss, normal
    modes a and b) equal `targets` ({"beta_x", "alpha_x", "beta_y", "alpha_y"}, as for twiss_match_beam): the linear
    optics from BEGINNING to `location` (Tao mat6) are run backwards, as in golden_twiss_targets.
    Only the lattice's Twiss functions change (ele_twiss, lat_list ele.a.beta, Twiss plots, BMAG against the lattice);
    they then describe a beam matched to `targets` at `location` instead of the f2_elec design start. Tracking a beam
    from a file does not use them. Needs uncoupled optics from BEGINNING to `location`.
    Fixer elements that are on (facet2-lattice pins the design Twiss at MRK0F with fixer_mrk0f) override BEGINNING:
    Bmad propagates the Twiss from the fixer both ways, back to BEGINNING too, and ignores the BEGINNING values. They
    are switched off in the model lattice; the design lattice keeps them. fixer_mrk0f is the golden match, so for the
    golden targets this gives the same Twiss as the fixer. Fixers do not act on tracked particles.
    Returns the Twiss set at BEGINNING.
    '''
    m = np.array(tao.matrix("BEGINNING", location)["mat6"])
    if max(np.abs(m[0:2, 2:4]).max(), np.abs(m[2:4, 0:2]).max()) > 1e-9:
        raise ValueError(f"set_lattice_twiss: the optics from BEGINNING to {location} couple x and y")
    begin = golden_twiss_targets(tao, "BEGINNING", golden={"location": location, **targets})
    tao.cmd("set global lattice_calc_on = F")
    for fixer in tao.lat_list("fixer::*", "ele.name", flags="-no_slaves"):
        tao.cmd(f"set element {fixer} is_on = F")
    for mode, plane in (("a", "x"), ("b", "y")):
        tao.cmd(f"set element beginning beta_{mode} = {begin[f'beta_{plane}']!r}")
        tao.cmd(f"set element beginning alpha_{mode} = {begin[f'alpha_{plane}']!r}")
    tao.cmd("set global lattice_calc_on = T")
    tw = tao.ele_twiss(location)
    got = {"beta_x": tw["beta_a"], "alpha_x": tw["alpha_a"], "beta_y": tw["beta_b"], "alpha_y": tw["alpha_b"]}
    if verbose:
        print(f"Lattice Twiss at {location}: beta {got['beta_x']:.3f}, {got['beta_y']:.3f} m, alpha {got['alpha_x']:.3f}, "
              f"{got['alpha_y']:.3f} (BEGINNING: beta {begin['beta_x']:.4g}, {begin['beta_y']:.4g} m)")
    if any(abs(got[k] - targets[k]) > 1e-6*max(1.0, abs(targets[k])) for k in targets):
        warnings.warn(f"set_lattice_twiss: lattice Twiss at {location} {got} differ from the targets {targets}")
    return begin


def _propagate_twiss(tao, start, end, twiss):
    """(beta_x, alpha_x, beta_y, alpha_y) at `end` for Twiss `twiss` (same keys as GOLDEN_TWISS_PR10571) at `start`."""
    m = np.array(tao.matrix(start, end)["mat6"])
    out = []
    for plane, i in (("x", 0), ("y", 2)):
        b, a = twiss[f"beta_{plane}"], twiss[f"alpha_{plane}"]
        M = m[i:i+2, i:i+2]
        B = M @ np.array([[b, -a], [-a, (1 + a**2)/b]]) @ M.T
        B /= np.sqrt(np.linalg.det(B))
        out += [B[0, 0], -B[0, 1]]
    return out


# Quads that the DAQ does not record (SLC magnets), used by the lattice designers to match into L2 ("BC14E" in f2_elec.lat.bmad)
L2_MATCH_QUADS = ["Q11401", "Q11501", "Q11601", "Q11701", "Q11801", "Q11901", "Q12201", "Q12301", "Q12401"]


def l2_virtual_match(tao, quads=L2_MATCH_QUADS, target="LI13BEG", golden=GOLDEN_TWISS_PR10571, regularization=1.0,
                     bounds=(0.5, 2.0), verbose=False):
    '''
    Virtual L2 match: rescale the K1 of `quads` so that the golden injector Twiss at PR10571, propagated through the
    loaded optics (e.g. the DAQ settings), arrive at `target` with the design Twiss.
    Operations match the beam into L2, but the matching quads after BC11 are not all in the DAQ (Q11401 and the L2
    FODO quads keep their lattice values), and the recorded QM11358/62/93 do not match the golden beam: for BEAMPHYS
    14438 BMAG is ~4.6/5.4 in L2, and the emittance grows ~4x along L2 and L3 without collective effects.
    Only quads that the DAQ does not record are changed (by default), so the DAQ magnets are kept.

    regularization: weight of the relative K1 changes in the least-squares fit (residuals: log(beta/beta_design),
    alpha - alpha_design per plane), to keep the quads close to their lattice values. With the 2026 facet2-lattice
    targets, 1.0 leaves BMAG ~1.04-1.07 in y (E320_17418, E300_16612); 0.3 reaches 1.001 with larger K1 changes.
    bounds: allowed range of K1/K1_before.
    Returns a dict with the K1 factors and the BMAG (x, y) at `target` before and after.
    '''
    d = tao.ele_twiss(target, which="design")
    design = (d["beta_a"], d["alpha_a"], d["beta_b"], d["alpha_b"])
    k0 = np.array([tao.ele_gen_attribs(q)["K1"] for q in quads])

    def set_k1(factors):
        for q, k in zip(quads, k0*factors):
            tao.cmd(f"set ele {q} K1 = {k}")

    def mismatch():
        bx, ax, by, ay = _propagate_twiss(tao, golden["location"], target, golden)
        return float(calcBMAG(design[0], design[1], bx, ax)), float(calcBMAG(design[2], design[3], by, ay))

    def residuals(factors):
        set_k1(factors)
        bx, ax, by, ay = _propagate_twiss(tao, golden["location"], target, golden)
        return np.concatenate([[np.log(bx/design[0]), ax - design[1], np.log(by/design[2]), ay - design[3]],
                               regularization*(factors - 1)])

    before = mismatch()
    fit = least_squares(residuals, np.ones(len(quads)), bounds=bounds)
    set_k1(fit.x)
    after = mismatch()
    result = {"factors": {q: float(f) for q, f in zip(quads, fit.x)}, "bmag_before": before, "bmag_after": after, "target": target}
    if verbose:
        print(f"L2 virtual match at {target}: BMAG {before[0]:.2f}/{before[1]:.2f} -> {after[0]:.2f}/{after[1]:.2f}; "
              f"K1 factors { {q: round(f, 3) for q, f in result['factors'].items()} }")
    if max(after) > 1.1:
        print(f"Warning: L2 virtual match reached only BMAG {after[0]:.2f}/{after[1]:.2f} at {target}")
    return result


# W-chicane sextupoles with horizontal movers (the S3 pair has none)
BC20_TUNE_SEXTUPOLES = ("S1EL", "S2EL", "S2ER", "S1ER")


def tune_bc20_sextupole_offsets(tao, beam=None, max_offset=2e-3, start="BEGBC20", target="ENDBC20",
                                sextupoles=BC20_TUNE_SEXTUPOLES, n_particles=2000, seed=0, verbose=False):
    '''
    Fit the x offsets of the W-chicane sextupoles (within +-max_offset of zero) to minimize the projected normalized
    emittances at `target` (eps_nx^2 + eps_ny^2), as the operators tune the dispersion after BC20 with the sextupole
    movers. The mover zero is unknown, so the offsets are a tuning knob; operators move them by up to ~2 mm.
    Offsets feed down quadrupole and dipole fields, so they cancel the residual dispersion of the chicane for the
    actual beam (energy spread, chirp, and the beam energy relative to P0C).

    beam: ParticleGroup at `start`; default: the beam that Tao saved at `start` in the last tracking.
    A random subset of n_particles is tracked from `start` to `target` without collective effects or ISR for each
    trial offset set (L-BFGS-B from zero offsets, so the fit finds the smallest offsets that work).
    The beam_init, track range, track type and bmad_com settings of Tao are restored afterwards; the fitted offsets
    are left set.
    Returns {"x_offsets_m": {name: offset}, "eps_n_before": (x, y), "eps_n_after": (x, y), "n_evaluations": n}.
    '''
    P = beam if beam is not None else getBeamAtElement(tao, start, tToZ=False)
    P = P[P.status == 1]
    if len(P) > n_particles:
        P = P[np.sort(np.random.default_rng(seed).choice(len(P), n_particles, replace=False))]
    gamma = P["mean_gamma"]
    beam_file = os.path.join(tao.filePathGlobal, "temp_beam", f"_bc20_tune_{os.getpid()}.h5")
    P.write(beam_file)

    com = tao.bmad_com()
    saved_com = {k: com[k] for k in ("csr_and_space_charge_on", "sr_wakes_on", "lr_wakes_on", "radiation_fluctuations_on")}
    saved_beam_file = tao.beam_init(0)["position_file"]
    saved_range = {k: tao.beam(0)[k] for k in ("track_start", "track_end")}
    tf = lambda b: "T" if b else "F"
    try:
        for k in saved_com:
            tao.cmd(f"set bmad_com {k} = F")
        tao.cmd(f"set beam_init position_file = {beam_file}")
        tao.cmd("reinit beam")
        tao.cmd(f"set beam_init track_start = {start}")
        tao.cmd(f"set beam_init track_end = {target}")
        tao.cmd("set global track_type = beam")
        n_eval = [0]

        def emittances(dx):
            tao.cmd("set global lattice_calc_on = F")      # one retracking per trial, not one per magnet
            for name, x in zip(sextupoles, dx):
                tao.cmd(f"set ele {name} X_OFFSET = {x}")
            tao.cmd("set global lattice_calc_on = T")
            n_eval[0] += 1
            alive = np.array(tao.bunch1(target, "state")) == 1
            c = {k: np.array(tao.bunch1(target, k))[alive] for k in ("x", "px", "y", "py", "pz")}
            emit = lambda u, pu: np.sqrt(np.linalg.det(np.cov(u, pu/(1 + c["pz"]))))
            return gamma*emit(c["x"], c["px"]), gamma*emit(c["y"], c["py"])

        before = emittances(np.zeros(len(sextupoles)))
        fit = minimize(lambda dx: 1e12*sum(e**2 for e in emittances(dx)), np.zeros(len(sextupoles)), method="L-BFGS-B",
                       bounds=[(-max_offset, max_offset)]*len(sextupoles), options={"eps": 2e-6, "maxiter": 60})
        after = emittances(fit.x)
    finally:
        tao.cmd("set global track_type = single")
        for k, v in saved_com.items():
            tao.cmd(f"set bmad_com {k} = {tf(v)}")
        tao.cmd(f"set beam_init position_file = {saved_beam_file}")
        tao.cmd(f"set beam_init track_start = {saved_range['track_start']}")
        tao.cmd(f"set beam_init track_end = {saved_range['track_end'] or 'END'}")
        if os.path.isfile(beam_file):
            os.remove(beam_file)
    result = {"x_offsets_m": {name: float(x) for name, x in zip(sextupoles, fit.x)},
              "eps_n_before": tuple(map(float, before)), "eps_n_after": tuple(map(float, after)), "n_evaluations": n_eval[0]}
    if verbose:
        print(f"BC20 sextupole tuning (+-{max_offset*1e3:.1f} mm): eps_nx/ny at {target} "
              f"{before[0]*1e6:.1f}/{before[1]*1e6:.1f} -> {after[0]*1e6:.1f}/{after[1]*1e6:.1f} um; "
              f"x offsets [mm] { {k: round(v*1e3, 3) for k, v in result['x_offsets_m'].items()} }")
    if any(abs(x) > 0.999*max_offset for x in fit.x):
        print(f"Note: BC20 sextupole tuning reached the {max_offset*1e3:.1f} mm limit")
    return result


def twiss_match_beam(tao, file, location, targets=None, verbose=False):
    '''
    Linear Twiss matching of the bunch in file+".h5" (rewritten in place): (x, px) and (y, py) are transformed so
    that the rms Twiss parameters of the bunch equal the targets. Emittances, energy, z/t and the particle
    distribution shape are unchanged (pmd_beamphysics ParticleGroup.twiss_match).
    targets: dict {"beta_x", "alpha_x", "beta_y", "alpha_y"} (m, rad). If None, the lattice Twiss at `location`
    (Tao normal modes a, b) are used.
    Returns the targets.
    '''
    if targets is None:
        tw = tao.ele_twiss(location)
        targets = {"beta_x": tw["beta_a"], "alpha_x": tw["alpha_a"], "beta_y": tw["beta_b"], "alpha_y": tw["alpha_b"]}
    P = ParticleGroup(file + ".h5")
    before = {plane: _beam_twiss(P, plane) for plane in ("x", "y")}
    for plane in ("x", "y"):
        P = P.twiss_match(beta=targets[f"beta_{plane}"], alpha=targets[f"alpha_{plane}"], plane=plane)
    P.write(file + ".h5")
    if verbose:
        for plane in ("x", "y"):
            b0, a0 = before[plane]
            b1, a1 = targets[f"beta_{plane}"], targets[f"alpha_{plane}"]
            print(f"Twiss match at {location}, {plane}: beta {b0:.2f} -> {b1:.2f} m, alpha {a0:.2f} -> {a1:.2f} (BMAG before {calcBMAG(b1, a1, b0, a0):.2f})")
    return targets


def _beam_twiss(P, plane):
    """(beta, alpha) from the (u, p_u/p0) covariance, the same definition ParticleGroup.twiss_match uses."""
    u, pu = (P.x, P.px) if plane == "x" else (P.y, P.py)
    t = twiss_calc(np.cov(u, pu/P["mean_p"], aweights=P.weight))
    return t["beta"], t["alpha"]


def run_initialized_sim(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], treat_dipoles_TF=False):
    '''
    If treat_dipoles_TF is False, this function just tracks from start to finish, saving the beam at the locations in "locations".
    If treat_dipoles_TF is True, before tracking it loads the constant fields from a "nominal" experiment lattice (default_bend_fields).

    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]

    if treat_dipoles_TF:
        tao = treat_dipoles(tao)
    trackBeam(tao, filepath=tao.filePathGlobal, trackStart = start, trackEnd = finish, autoLoadActiveFile=False)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind], tToZ=False)
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


def run_initialized_sim_edit_lattice_energy_for_dipoles(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], desired_P0Cs_MeV=[None,None,None,None]):
    '''
    Old function to change the bend energies without DB_Field.
    If desired_P0Cs_MeV are None, this function just tracks from start to finish, saving the beam at the locations in "locations".
    If desired_P0Cs_MeV is something else, it:
    tunes the cavities to the desired_P0Cs_MeV
    tracks to the BX0FBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BX0FEND
    tunes them to the desired_P0Cs_MeV
    tracks to the BC11CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC11CEND
    tunes them to the desired_P0Cs_MeV
    tracks to the BC14CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC14CEND
    tunes them to the desired_P0Cs_MeV
    tracks to the BC20CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC20CEND
    tunes them to the desired_P0Cs_MeV
    tracks to the finish

    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]

    current_start = start
    # injector
    if desired_P0Cs_MeV[0] is not None:
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV, change_only_L0B=True)
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FBEG", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BX0FBEG", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000], change_only_L0B=True)        
        current_start = "BX0FBEG"
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FEND", autoLoadActiveFile=False)
        #print(f'<x> inside of the dogleg: {tao.bunch_params("BPM10731")["centroid_vec_1"]}')
        getBeamAtElement(tao, "BX0FEND", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BX0FEND"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L1
    if desired_P0Cs_MeV[1] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CBEG", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BC11CBEG", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BC11CBEG"
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CEND", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BC11CEND", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BC11CEND"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L2
    if desired_P0Cs_MeV[2] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BEGBC14E", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BEGBC14E", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BEGBC14E"
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDBC14E", autoLoadActiveFile=False)
        getBeamAtElement(tao, "ENDBC14E", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDBC14E"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L3
    if desired_P0Cs_MeV[3] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BEGBC20", autoLoadActiveFile=False)
        getBeamAtElement(tao, "BEGBC20", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BEGBC20"
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDBC20", autoLoadActiveFile=False)
        getBeamAtElement(tao, "ENDBC20", tToZ=False).write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDBC20"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # Fin
    trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = finish, autoLoadActiveFile=False)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind], tToZ=False)
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


def run_initialized_sim_edit_bunch_energy(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], edited_bunch_energy_at_checkpoints_MeV=[None,None,None,None]):
    '''
    Tracks the bunch and changes the bunch energy at BX0FBEG, BC11CBEG, ENDL2F, ENDL3F_2 if any of edited_bunch_energy_at_checkpoints_MeV (list with 4 numbers) is not None.
    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]
    current_start = start
    if edited_bunch_energy_at_checkpoints_MeV[0] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BX0FBEG", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "BX0FBEG", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[0], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BX0FBEG"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[1] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "BC11CBEG", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "BC11CBEG", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[1], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BC11CBEG"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[2] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDL2F", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "ENDL2F", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[2], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDL2F"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[3] is not None:
        trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = "ENDL3F_2", autoLoadActiveFile=False)
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "ENDL3F_2", tToZ=False), pzMeV=edited_bunch_energy_at_checkpoints_MeV[3], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDL3F_2"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    
    trackBeam(tao, filepath=tao.filePathGlobal, trackStart = current_start, trackEnd = finish, autoLoadActiveFile=False)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind], tToZ=False)
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


def edit_energy_based_on_beam_all(tao, start, file, desired_beam_energies=None, change_only_L0B=True, change_file_pz=True, verbose=False, finalnumMacroParticles=5e4, tune_dipoles=True, dipole_fields=default_bend_fields):
    '''
    This function changes the cavity voltages so that the tracked beam has the "desired_P0Cs" <pz> between the cavities.
    desired_P0Cs: the desired <pz> in eV between the cavities. Must be None or a list of 4 numbers (dogleg, bc11, bc14, bc20).
    If None, the P0Cs from the lattice are used.
    start: start of the simulation. Need to be in the injector (use L0AFEND to avoid confusion).
    file: the bunch file to use for the tuning. Providing it is a must because the energy obviously depends on the charge and the size of the bunch.
    change_only_L0B: in the injector, scale only L0B (default). A beam that starts at L0AFEND does not go through L0A:
    scaling L0A only moves the reference energy at L0AFEND, and with change_file_pz the beam is then rescaled to it
    (e.g. an IMPACT beam from 68.6 to 67.3 MeV), which mismatches it to the injector quads.
    If changing both L0A and L0B, the input beam energy at L0A is adjusted accordingly later.
    change_file_pz: change the bunch energy at the "start" to match the "start" P0C (overrides the energy of the beam file).
    finalnumMacroParticles: the function uses "set_beam()" at the end. If you don't want to set the beam manually after the tuning,
    this option allows you to regulate the number of particles in that set beam (the one you provided with the "file").
    '''
    if desired_beam_energies is None:
        desired_beam_energies = [float(tao.ele_gen_attribs("BX0FBEG")["P0C"]), float(tao.ele_gen_attribs("BC11CBEG")["P0C"]), float(tao.ele_gen_attribs("ENDL2F")["P0C"]), float(tao.ele_gen_attribs("ENDL3F_2")["P0C"])]
    pre = 'temp_beam/'
    suf = 'temp'
    if verbose:
        print(f'P0C now: {desired_beam_energies}')
    
    # injector
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "BX0FBEG"]
    if tune_dipoles:
            tao = treat_dipoles(tao, dipole_fields)
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to BX0FBEG: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_inj(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[0]*1e-6, change_only_L0B=change_only_L0B)
    if verbose:
        print(f'edited P0C BX0FBEG: {tao.ele_gen_attribs("BX0FBEG")["P0C"]}')
    
    # L1
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "BC11CBEG"]

    if tune_dipoles:
            tao = treat_dipoles(tao, dipole_fields)
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to BC11CBEG: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L1(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[1]*1e-6)
    if verbose:
        print(f'edited P0C BC11CBEG: {tao.ele_gen_attribs("BC11CBEG")["P0C"]}')
    
    # L2
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "ENDL2F"]
    
    if tune_dipoles:
            tao = treat_dipoles(tao, dipole_fields)
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to ENDL2F: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL2F", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L2(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[2]*1e-6)
    if verbose:
        print(f'edited P0C ENDL2F: {tao.ele_gen_attribs("ENDL2F")["P0C"]}')
    
    # L3
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "ENDL3F_2"]
    
    if tune_dipoles:
            tao = treat_dipoles(tao, dipole_fields)
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to ENDL3F_2: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL2F", tToZ=False).pz)), float(np.mean(getBeamAtElement(tao, "ENDL3F_2", tToZ=False).pz))]}')
    
    tao = edit_energy_based_on_beam_L3(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[3]*1e-6)
    if verbose:
        print(f'edited P0C ENDL3F_2: {tao.ele_gen_attribs("ENDL3F_2")["P0C"]}')
    
    set_beam(tao, file, numMacroParticles=finalnumMacroParticles)
    return tao
