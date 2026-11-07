"""Build and edit a Tao lattice from a FACET-II DAQ experiment dataset.

Holds the BMAD-element-to-EPICS-PV maps (quadrupoles, sextupoles, cavities,
correctors, bends) and the functions that read a DAQ DATASET and push those
values into the lattice.

Moved from DAQdatasetToSimFunctions.py (all) and simulationFunctions.py
(get_tao_from_experiment).
"""

import os
from pathlib import Path
from facet2_daq import DATASET
import numpy as np
from pmd_beamphysics import ParticleGroup

from ...beam.generation import make_simple_bunch_standalone
from ...beam.manipulation import edit_bunch_parameters
from .lattice.linac import setLinacPhase, setLinacGradientAuto
from .lattice.set_lattice import setQuadkG, setSextkG
from .core import initializeTao
from .energy import save_dipoles, treat_dipoles
from .runs import set_beam, edit_energy_based_on_beam_all, run_initialized_sim_edit_bunch_energy, run_initialized_sim_edit_lattice_energy_for_dipoles, run_initialized_sim


## Initialize and run a simulation


### High end

def daq_scan_path(experiment, date, daq_root=None):
    """Folder of a DAQ experiment, <daq_root>/<experiment><date> (e.g. date="/2026/20260121").

    daq_root: root folder of the FACET-II DAQ data. If None, the FACET2_DAQ_ROOT environment variable is used.
    """
    root = daq_root or os.environ.get("FACET2_DAQ_ROOT")
    if not root:
        raise ValueError("The location of the FACET-II DAQ data is not set: pass daq_root=... or set the "
                         "FACET2_DAQ_ROOT environment variable (see docs/daq_settings.md).")
    return os.path.join(root, experiment + date)


def get_tao_from_experiment(experiment="", scan_number="", date="", start='L0AFEND', finish='PR11375', filepath=None, locationsToSave = [], daq_root=None,
                            csrTF=False, lscTF=False, file_ext = "", energy=None, N_in_simple_bunch=5e4, N_to_use_from_file=None, tune_dipoles_to_125_335_4500_10000_MeV=False, tune_dipoles=True,
                            correctors_coef=0, correctors_from_beg=False, run=True, gaussFromExternal=False, edit_only_energy_from_exp=False, energy_edit_on_beam=False, verbose=False,
                            lattice='setLattice_configs/2024-12-09_oneBunch_CSR-on_optimized.yml', moments=[None,None,None,None,None,None], means=[0,0,0,0,None], charge=1.6e-9, sr_wakes_on=False, lr_wakes_on=False,
                            desired_beam_energies_for_the_feedback=None, desired_P0Cs_MeV=[None,None,None,None], grid_size=[32,32,32], lsc_method="slice", csr_method="1_dim", n_bin=32,
                            edited_bunch_energy_at_checkpoints_MeV=[None, None, None, None], beam_edits=True):
    '''
    experiment: "BEAMPHYS" is an example.
    scan_number: DAQ scan number. "14438" is an example.
    date: the date when the scan was taken in a specific form. "/2026/20260121" is an example.
    daq_root: root folder of the FACET-II DAQ data. If None, the FACET2_DAQ_ROOT environment variable is used.

    start: where the simulation starts. This 1) can affect the initial bunch energy (see "energy"), 2) determines the start for the energy_edit_on_beam and run_initialized_sim functions.
    finish: determines the finish for the energy_edit_on_beam and run_initialized_sim functions.

    filepath: Path to the FACET2-S2E repository (the folder with bmad/, beams/, setLattice_configs/, temp_beam/).
    If None, it is taken from the location of the installed package (works with "pip install -e .").

    lattice: additional lattice settings to use.

    file_ext: the beam file. Not required (default is ""). If it is "", a Gaussian bunch will be created with "moments". "moments" in this case is required!
    N_to_use_from_file: randomly choose N_to_use_from_file particles from the external bunch.
    gaussFromExternal: if True, a Gaussian bunch will be created instead of the supplied bunch, with sizes being the same as in the supplied bunch.
    alpha in this case is set to 0, and the emittance is changed accordingly.
    N_in_simple_bunch: number of particles in the created Gaussian bunch (even if created from the external file with gaussFromExternal).

    locationsToSave: Usually, tao saves the bunch to RAM (I believe). The beam will be saved to the disk at the locationsToSave entries.
    Note that it will give an error if Tao does not save a provided location to RAM. To add the location to RAM, go to the quickstart -> initializeTao -> edit the "set beam add_saved_at" lines.

    csrTF, lscTF, sr_wakes_on, lr_wakes_on: bool settings for the collective effects. Work separately.
    grid_size: the grid size used by SC or CSR if they are 3d.
    lsc_method: off, fft_3d or slice.
    csr_method: off, steady_state_3d or 1_dim.
    n_bin: number of longitudinal slices in the slice/1_dim methods.

    energy: initial energy of the bunch in MeV (if positive number).
    Set energy=-1 to use the energy from the beam file, or energy=None to use the energy from the lattice / experiment (if experiment and scan_number are provided).

    moments: list of 6 numbers, the RMS sizes of the bunch in x, xp, y, yp, z, pz (in meters, radians, meters, radians, meters, MeV/c).
    Set a moment to -1 to keep the same as in the input file (applicable to each number). Default is -1 for all.

    means: list of 5 numbers, the means of the bunch in x, xp, y, yp, z (in meters, radians, meters, radians, meters).
    Set a mean to -1 to keep the same as in the input file (applicable to each number). Default is [0,0,0,0,-1] (I needed a centered bunch. Change this if needed).
    z and t are set to 0 in the set_beam() by default.
    
    desired_P0Cs_MeV: This settings allows using different cavity energies (from 125, 335, 4500, and 10000) while preserving the linac geometry.
    For example, setting it to [124, None, None, None], will adjust the injector and L1 cavities to have [124, 335, 4500, 10000] MeV. This will result in a non-zero <x> inside of the dogleg.

    energy_edit_on_beam: The beam feedback. If collective effects slow the bunch down, the cavity voltages will be adjusted to match the desired_beam_energies_for_the_feedback.
    desired_beam_energies_for_the_feedback: <pz> in eV that you would like to see before the dogleg, bc11, bc14, and bc20 (see the energy_edit_on_beam function).

    correctors_coef: what corrector strength to use from the DAQ database. -1/10 for the experimental values; 0 to turn them off.
    correctors_from_beg: if True, all DAQ saved correctors will be used. If False, correctors will be enabled from BX0FBEG.

    edit_only_energy_from_exp: if true, the quadrupoles, sextupoles, dipoles, and correctors will not be loaded from the DAQ database.

    tune_dipoles: if True, the dipole fields are adjusted using DB_FIELD to the corresponding angle and rho from the .tao lattice at some energies:
    Will tune to the DAQ values (it saves energy) if experiment and scan_number are provided, and to the default [125, 335, 4500, 10000] otherwise.
    tune_dipoles_to_125_335_4500_10000_MeV: if True, the dipole magnetic fields are set to [125, 335, 4500, 10000] even if the DAQ is provided.
    If False, the dipole fields are not adjusted and follow the lattice energy, as with initializeTao() and trackBeam().
    '''
    if filepath is None:
        # src/FACET2_S2E/simulators/bmad/experiment.py -> repository root (same rule as initializeTao)
        filepath = str(Path(__file__).resolve().parents[4])
    if not os.path.isfile(f"{filepath}/bmad/models/f2_elec/tao.init"):
        raise FileNotFoundError(f'No FACET2-S2E lattice found in "{filepath}". Pass filepath="/path/to/FACET2-S2E", '
                                'or install the package with "pip install -e ." so that the repository can be found automatically.')

    tao = initializeTao(filePath = filepath, loadCustomLatticeTF=True, csrTF=csrTF, lscTF=lscTF, latticeFile=lattice, bmad_grid_size=grid_size, verbose=verbose, sr_wakes_on=sr_wakes_on, lr_wakes_on=lr_wakes_on, lsc_method=lsc_method, csr_method=csr_method, n_bin=n_bin, autoLoadActiveFile=False)
    
    dipoleEnergies_MeV = [125, 335, 4500, 10000]
    # copy the experiment data
    if experiment!="" and scan_number!="":
        ds = DATASET("", experiment, scan_number, pathfull = daq_scan_path(experiment, date, daq_root))
        # check if magnets data is not needed. Dogleg energy is always 125 MeV (maybe need to change to the mean of 'BEND_IN10_661_BDES' and 'BEND_IN10_751_BDES' (they are in GeV), so that the magnet strengths are actually correct)
        if edit_only_energy_from_exp:
            tao = edit_energy_tao_based_on_experiment_database(tao, ds)
        else:
            tao = edit_tao_based_on_experiment_database(tao, ds, correctors_coef=correctors_coef, correctors_from_beg=correctors_from_beg)
            dipoleEnergies_MeV = save_dipole_energies_from_the_DAQ_database(ds)

    if tune_dipoles_to_125_335_4500_10000_MeV:
        dipoleEnergies_MeV = [125, 335, 4500, 10000]

    fields = save_dipoles(tao, dipoleEnergies_MeV)

    # tao.cmd(f'set global lattice_calc_on = T')
    # deal with the bunch
    current_e_start = tao.ele_gen_attribs(start)["P0C"]*1e-6 if energy is None else energy
    folder = filepath + "/"
    file = folder+ "temp_beam/temp"
    if beam_edits:
        if file_ext=='':
            moments = [0 if moment is None else moment for moment in moments]
            make_simple_bunch_standalone(N = N_in_simple_bunch, meanPzMeV = current_e_start, moments=moments, save_path = file, charge=charge)
        else:
            energy_from_file = None if energy==-1 else current_e_start
            edit_bunch_parameters(file_ext, pzMeV=energy_from_file, moments=moments, means=means, charge=charge, path_to_write=file)
            if gaussFromExternal:
                P = ParticleGroup(file+".h5")
                moments = [np.std(P.x),np.std(P.xp),np.std(P.y),np.std(P.yp),np.std(P.t)*3e8,np.std(P.pz)]
                make_simple_bunch_standalone(N = N_in_simple_bunch, meanPzMeV = current_e_start, moments=moments, save_path = file)
                edit_bunch_parameters(file, pzMeV=current_e_start, moments=moments, means=means, charge=charge, path_to_write=file)
    set_beam(tao, file, numMacroParticles=None if (gaussFromExternal or file_ext=='') else N_to_use_from_file)

    # energy feedback on beam
    if energy_edit_on_beam:
        tao = edit_energy_based_on_beam_all(tao, start, file, verbose=verbose, desired_beam_energies=desired_beam_energies_for_the_feedback, finalnumMacroParticles=N_to_use_from_file, tune_dipoles=tune_dipoles, dipole_fields=fields)

    if tune_dipoles:
        tao = treat_dipoles(tao, fields)

    # run the sim and save the bunch
    if run:
        pre = 'temp_beam/'
        suf = 'temp'
        if locationsToSave == []:
            locationsToSave = [start, finish]
        if edited_bunch_energy_at_checkpoints_MeV!=[None, None, None, None]:
            tao = run_initialized_sim_edit_bunch_energy(tao, start, finish, edited_bunch_energy_at_checkpoints_MeV=edited_bunch_energy_at_checkpoints_MeV, pre=pre, suf=suf, locations=locationsToSave)
        elif desired_P0Cs_MeV!=[None, None, None, None]:
            tao = run_initialized_sim_edit_lattice_energy_for_dipoles(tao, locationsToSave[0], locationsToSave[-1], pre, suf, locationsToSave, desired_P0Cs_MeV=desired_P0Cs_MeV)
        else:
            tao = run_initialized_sim(tao, locationsToSave[0], locationsToSave[-1], pre, suf, locationsToSave)
    return tao


## Edit lattice according to experiment

### BMAD to DAQ maps

#quadrupoles
bmad_quad_to_pv_map = {
    # s10
    'QA10361': ["nonBSA_List_S10", 'QUAD_IN10_361_BDES'],
    'QA10371': ["nonBSA_List_S10", 'QUAD_IN10_371_BDES'],
    'QE10425': ["nonBSA_List_S10", 'QUAD_IN10_425_BDES'],
    'QE10441': ["nonBSA_List_S10", 'QUAD_IN10_441_BDES'],
    'QE10511': ["nonBSA_List_S10", 'QUAD_IN10_511_BDES'],
    'QE10525': ["nonBSA_List_S10", 'QUAD_IN10_525_BDES'],
    'QM10631': ["nonBSA_List_S10", 'QUAD_IN10_631_BDES'],
    'QM10651': ["nonBSA_List_S10", 'QUAD_IN10_651_BDES'],
    'QB10731': ["nonBSA_List_S10", 'QUAD_IN10_731_BDES'],
    'QM10771': ["nonBSA_List_S10", 'QUAD_IN10_771_BDES'],
    'QM10781': ["nonBSA_List_S10", 'QUAD_IN10_781_BDES'],
    
    # s11
    'QA11132': ["nonBSA_List_S11", 'QUAD_LI11_132_BCON'],
    'Q11201': ["nonBSA_List_S11", 'QUAD_LI11_201_BCON'],
    'QA11265': ["nonBSA_List_S11", 'QUAD_LI11_265_BCON'],
    'Q11301': ["nonBSA_List_S11", 'QUAD_LI11_301_BCON'],
    'QM11312': ["nonBSA_List_S11", 'QUAD_LI11_312_BCON'],
    'CQ11317': ["nonBSA_List_S11", 'QUAD_LI11_317_BCON'],
    'SQ11340': ["nonBSA_List_S11", 'QUAD_LI11_340_BCON'],
    'CQ11352': ["nonBSA_List_S11", 'QUAD_LI11_352_BCON'],
    'QM11358': ["nonBSA_List_S11", 'QUAD_LI11_358_BCON'],
    'QM11362': ["nonBSA_List_S11", 'QUAD_LI11_362_BCON'],
    'QM11393': ["nonBSA_List_S11", 'QUAD_LI11_393_BCON'],

    # s12 - s18
    # NOT SAVED IN DAQ?

    # s19
    # MOST ARE NOT SAVED IN DAQ?
    'Q19851': ["nonBSA_List_S19", 'QUAD_LI19_851_BACT'],
    'Q19871': ["nonBSA_List_S19", 'QUAD_LI19_871_BACT'],

    # s20
    'SQ1': ['nonBSA_List_S20Magnets', 'LI20_QUAD_2086_BACT'],
    'Q1EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2060_BACT'],
    'Q2EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2130_BACT'],
    'Q3EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q3EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q4EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4EL_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q5EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2230_BACT'],
    'Q6E': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2251_BACT'],
    'Q5ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2230_BACT'],
    'Q4ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4ER_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q3ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q3ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q2ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2130_BACT'],
    'Q1ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2060_BACT'],
    #'SQ2': ['nonBSA_List_S20Magnets', 'LI20_QUAD_3015_BACT'], SQ2 is not in the BMAD model. But it is usually 0 anyway.
    'Q5FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3011_BACT'],
    'Q4FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3311_BACT'],
    'Q3FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3151_BACT'],
    'Q2FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_1910_BACT'],
    'Q1FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3204_BACT'],
    'Q0FF': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3031_BACT'],
    'Q0D': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3141_BACT'],
    'Q1D': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3261_BACT'],
    'Q2D': ['nonBSA_List_S20Magnets', 'LI20_LGPS_3091_BACT']
}


bmad_quad_to_pv_map_boost = {
    # s20
    'Q1EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2061_BACT'],
    'Q2EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2131_BACT'],
    'Q3EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2151_BACT'],
    'Q3EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2151_BACT'],
    'Q4EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q4EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q4EL_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q5EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2231_BACT'],

    'Q5ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2262_BACT'],
    'Q4ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q4ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q4ER_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q3ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2341_BACT'],
    'Q3ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2341_BACT'],
    'Q2ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2371_BACT'],
    'Q1ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2441_BACT']
}


# sextupoles
bmad_sextupoles_to_pv_map = {
    'S1EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2145_BACT'],
    'S2EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2165_BACT'],
    'S3EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2195_BACT'],
    'S3EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2195_BACT'],
    'S3ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2275_BACT'],
    'S3ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2275_BACT'],
    'S2ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2335_BACT'],
    'S1ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2365_BACT']
}


# sextupole offsets (in mm) (s1l, s2l, s2r, s1r)
sextupole_offsets_x_from_daq = [['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO552'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO502'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO517'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO567']]


sextupole_offsets_y_from_daq = [['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO557'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO507'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO522'],
                                ['nonBSA_List_S20Magnets', 'SIOC_SYS1_ML00_AO572']]


# injector cavities
def get_l0a_phase(database):
    """Read the L0A phase from the DAQ database and apply the FACET phase offset.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_S10RF"]['KLYS_LI10_31_SFB_PDES'])-20


def get_l0a_ampl(database):
    """Read the L0A amplitude from the database and convert to Tao voltage units.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_S10RF"]['KLYS_LI10_31_ADES'])*2.864664e6


def get_l0b_phase(database):
    """Read the L0B phase from the database.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_S10RF"]['KLYS_LI10_41_SFB_PDES'])


def get_l0b_ampl(database):
    """Read the L0B amplitude from the database and convert to Tao voltage units.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_S10RF"]['KLYS_LI10_41_ADES'])*2.864664e6


def get_l1_phase(database):
    """Read the L1 cavity phase from the database.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_LINAC_KLYS"]['KLYS_LI11_11_SSSB_PDES'])


def get_l2_phase(database):
    """Read the L2 cavity phase from the database.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_LINAC_KLYS"]['LI14_SBST_1_PHAS'])


def get_l3_phase(database):
    """Read the L3 cavity phase from the database.
    """
    return np.mean(database._data["scalars"]["nonBSA_List_LINAC_KLYS"]['LI19_SBST_1_PHAS'])


# L3
bmad_cavity_to_pv_map = {
    'K19_8A1': ['nonBSA_List_S20Magnets', 'LI19_KLYS_81_ADES'],
    'K19_8A2': ['nonBSA_List_S20Magnets', 'LI19_KLYS_81_ADES'],
    'K19_8A3': ['nonBSA_List_S20Magnets', 'LI19_KLYS_81_ADES']
}


bmad_corrector_to_pv_map_before_dogleg = {
    # Correctors before L0AFEND
    'YC10122': ["nonBSA_List_S10", 'YCOR_IN10_122_BDES'],
    'XC10121': ["nonBSA_List_S10", 'XCOR_IN10_121_BDES'],
    'XC10221': ["nonBSA_List_S10", 'XCOR_IN10_221_BDES'],
    'YC10222': ["nonBSA_List_S10", 'YCOR_IN10_222_BDES'],
    'YC10312': ["nonBSA_List_S10", 'YCOR_IN10_312_BDES'],
    'XC10311': ["nonBSA_List_S10", 'XCOR_IN10_311_BDES'],
    
    # Correctors after L0AFEND
    'YC10382': ["nonBSA_List_S10", 'YCOR_IN10_382_BDES'],
    'XC10381': ["nonBSA_List_S10", 'XCOR_IN10_381_BDES'],
    'YC10412': ["nonBSA_List_S10", 'YCOR_IN10_412_BDES'],
    'XC10411': ["nonBSA_List_S10", 'XCOR_IN10_411_BDES'],
    'YC10492': ["nonBSA_List_S10", 'YCOR_IN10_492_BDES'],
    'XC10491': ["nonBSA_List_S10", 'XCOR_IN10_491_BDES'],
    'XC10521': ["nonBSA_List_S10", 'XCOR_IN10_521_BDES'],
    'YC10522': ["nonBSA_List_S10", 'YCOR_IN10_522_BDES'],
    'XC10641': ["nonBSA_List_S10", 'XCOR_IN10_641_BDES'],
    'YC10642': ["nonBSA_List_S10", 'YCOR_IN10_642_BDES'],
}


bmad_corrector_to_pv_map = {    
    # Correctors after the dogleg beginning
    'XC10721': ["nonBSA_List_S10", 'XCOR_IN10_721_BDES'],
    'YC10722': ["nonBSA_List_S10", 'YCOR_IN10_722_BDES'],
    'XC10761': ["nonBSA_List_S10", 'XCOR_IN10_761_BDES'],
    'YC10762': ["nonBSA_List_S10", 'YCOR_IN10_762_BDES'],
    
    # correctors in sector 11
    'YC11105': ["nonBSA_List_S11", 'YCOR_LI11_105_BCON'],
    'XC11104': ["nonBSA_List_S11", 'XCOR_LI11_104_BCON'],
    'YC11141': ["nonBSA_List_S11", 'YCOR_LI11_141_BCON'],
    'XC11140': ["nonBSA_List_S11", 'XCOR_LI11_140_BCON'],
    'XC11202': ["nonBSA_List_S11", 'XCOR_LI11_202_BCON'],
    'YC11203': ["nonBSA_List_S11", 'YCOR_LI11_203_BCON'],
    'YC11273': ["nonBSA_List_S11", 'YCOR_LI11_273_BCON'],
    'XC11272': ["nonBSA_List_S11", 'XCOR_LI11_272_BCON'],
    'YC11305': ["nonBSA_List_S11", 'YCOR_LI11_305_BCON'],
    'XC11304': ["nonBSA_List_S11", 'XCOR_LI11_304_BCON'],
    'YC11321': ["nonBSA_List_S11", 'YCOR_LI11_321_BCON'],
    'YC11365': ["nonBSA_List_S11", 'YCOR_LI11_365_BCON'],
    'XC11398': ["nonBSA_List_S11", 'XCOR_LI11_398_BCON'],
    'YC11399': ["nonBSA_List_S11", 'YCOR_LI11_399_BCON'],
    
    # Sector 19 correctors
    'XC19202': ["nonBSA_List_S19", 'LI19_XCOR_202_BACT'],
    'XC19302': ["nonBSA_List_S19", 'LI19_XCOR_302_BACT'],
    'XC19402': ["nonBSA_List_S19", 'LI19_XCOR_402_BACT'],
    'XC19502': ["nonBSA_List_S19", 'LI19_XCOR_502_BACT'],
    'XC19602': ["nonBSA_List_S19", 'LI19_XCOR_602_BACT'],
    'XC19700': ["nonBSA_List_S19", 'LI19_XCOR_700_BACT'],
    'XC19802': ["nonBSA_List_S19", 'LI19_XCOR_802_BACT'],
    'XC19900': ["nonBSA_List_S19", 'LI19_XCOR_900_BACT'],

    'YC19203': ["nonBSA_List_S19", 'LI19_YCOR_203_BACT'],
    'YC19303': ["nonBSA_List_S19", 'LI19_YCOR_303_BACT'],
    'YC19403': ["nonBSA_List_S19", 'LI19_YCOR_403_BACT'],
    # 'YC57145': ["", ''], - not saved in DAQ
    # 'YC57146': ["", ''], - not saved in DAQ
    'YC19503': ["nonBSA_List_S19", 'LI19_YCOR_503_BACT'],
    'YC19603': ["nonBSA_List_S19", 'LI19_YCOR_603_BACT'],
    'YC19700': ["nonBSA_List_S19", 'LI19_YCOR_700_BACT'],
    'YC19803': ["nonBSA_List_S19", 'LI19_YCOR_803_BACT'],
    'YC19900': ["nonBSA_List_S19", 'LI19_YCOR_900_BACT'],

    'XC1996': ["nonBSA_List_S20Magnets", 'LI20_XCOR_1996_BACT'],

    # BC20 chicane, horizontal
    'XC1E':   ["nonBSA_List_S20Magnets", 'LI20_XCOR_2096_BACT'],
    'XC2E':   ["nonBSA_List_S20Magnets", 'LI20_XCOR_2176_BACT'],
    'XC3E':   ["nonBSA_List_S20Magnets", 'LI20_XCOR_2326_BACT'],
    'XC4E':   ["nonBSA_List_S20Magnets", 'LI20_XCOR_2396_BACT'],
    'XC2460': ["nonBSA_List_S20Magnets", 'LI20_XCOR_2460_BACT'],
    # BC20 chicane, vertical
    'YC1E':   ["nonBSA_List_S20Magnets", 'LI20_YCOR_2087_BACT'],
    'YC2181': ["nonBSA_List_S20Magnets", 'LI20_YCOR_2181_BACT'],
    'YC2E':   ["nonBSA_List_S20Magnets", 'LI20_YCOR_2227_BACT'],
    'YC3E':   ["nonBSA_List_S20Magnets", 'LI20_YCOR_2267_BACT'],
    'YC2321': ["nonBSA_List_S20Magnets", 'LI20_YCOR_2321_BACT'],
    'YCWIGE': ["nonBSA_List_S20Magnets", 'LI20_YCOR_2420_BACT'],
    # bend trims
    'XCB2LE': ["nonBSA_List_S20Magnets", 'LI20_BTRM_2111_BACT'],
    'XCB3LE': ["nonBSA_List_S20Magnets", 'LI20_BTRM_2241_BACT'],
    'XCB3RE': ["nonBSA_List_S20Magnets", 'LI20_BTRM_2261_BACT'],
    'XCB2RE': ["nonBSA_List_S20Magnets", 'LI20_BTRM_2391_BACT'],
    # final focus / spectrometer
    'XC1FF':  ["nonBSA_List_S20Magnets", 'LI20_XCOR_3026_BACT'],
    'YC1FF':  ["nonBSA_List_S20Magnets", 'LI20_YCOR_3017_BACT'],
    'YC2FF':  ["nonBSA_List_S20Magnets", 'LI20_YCOR_3057_BACT'],
    'XC3FF':  ["nonBSA_List_S20Magnets", 'LI20_XCOR_3086_BACT'],
    'XC1EX':  ["nonBSA_List_S20Magnets", 'LI20_XCOR_3276_BACT'],  # by elimination
}


### Lattice edit functions

def edit_tao_based_on_experiment_database(tao, dataset, correctors_coef=-1/10, correctors_from_beg=False):
    '''
    - Bend settings are disabled because setting B_FIELD in BMAD changes the positions of all downstream elements.
    - Correctors may be either all enabled (with correctors_coef != 0), enabled only from the dogleg beginning (correctors_from_beg=False), or turned off (correctors_coef=0).
    Cavities:
    - The phase set to all klystrons is the same (for a given cavity). But the EPICS databases suggest that in experiment there are two klystrons with +- large phase offset. It is disregarded here.
    - The voltage is also the same, and is tuned to match the default 125, 335, 4500, 10000 MeV.
    '''
    # Imported here (rather than at module level) to avoid a circular import with
    # simulationFunctions, which itself imports functions from this module.
    from .lattice.set_lattice import setAllWChicaneSextupolesXOffsets, setAllWChicaneSextupolesYOffsets

    # tao.cmd(f'set ele L0BF PHI0 = {l0bphase / 360.}')
    # tao.cmd(f'set ele L0BF VOLTAGE = {(61.0e6 + (mean_energy_MeV_lattice-125)*1e6) / math.cos(2*math.pi*l0bphase/360)}')

    tao = edit_energy_tao_based_on_experiment_database(tao, dataset)

    for k, v in bmad_quad_to_pv_map.items():
        #print(f'{k} base {np.mean(dataset._data["scalars"][v[0]][v[1]])}')
        quad_integrated_T = np.mean(dataset._data["scalars"][v[0]][v[1]])
        if k in bmad_quad_to_pv_map_boost:
            #print(f'{k} boost {np.mean(dataset._data["scalars"][bmad_quad_to_pv_map_boost[k][0]][bmad_quad_to_pv_map_boost[k][1]])}')
            quad_integrated_T += np.mean(dataset._data["scalars"][bmad_quad_to_pv_map_boost[k][0]][bmad_quad_to_pv_map_boost[k][1]])
        setQuadkG(tao, k, quad_integrated_T)

    # for k, v in bmad_bend_to_pv_map.items():
    #     bend_T = np.mean(dataset._data["scalars"][v[0]][v[1]])*tao.ele_gen_attribs(k)["ANGLE"]/tao.ele_gen_attribs(k)["L"]/0.299792458
    #     # print(f'{k} bend T from DAQ: {bend_T}')
    #     tao.cmd(f'set ele {k} B_FIELD = {bend_T}')
    #     # print(f'{k}: {tao.ele_gen_attribs(k)["B_FIELD"]}')


    for k, v in bmad_sextupoles_to_pv_map.items():
        sextupole_DAQ = np.mean(dataset._data["scalars"][v[0]][v[1]])
        setSextkG(tao, k, sextupole_DAQ)
        #print(sextupole_DAQ)
    
    # Offsets in m, ordered as (S1EL, S2EL, S3EL, S3ER, S2ER, S1ER). The DAQ has movers only for S1/S2 (s1l, s2l, s2r, s1r); S3 stays at 0.
    # Float arrays: an integer array would truncate the mm-scale offsets to 0.
    sextXOffsets = np.zeros(6)
    sextYOffsets = np.zeros(6)
    for i in range(2):
        sextXOffsets[i] = np.mean(dataset._data["scalars"][sextupole_offsets_x_from_daq[i][0]][sextupole_offsets_x_from_daq[i][1]])*1e-3
        sextYOffsets[i] = np.mean(dataset._data["scalars"][sextupole_offsets_y_from_daq[i][0]][sextupole_offsets_y_from_daq[i][1]])*1e-3
    for i in range(1, 3):
        # i=1 -> S1ER (s1r), i=2 -> S2ER (s2r)
        sextXOffsets[-i] = np.mean(dataset._data["scalars"][sextupole_offsets_x_from_daq[-i][0]][sextupole_offsets_x_from_daq[-i][1]])*1e-3
        sextYOffsets[-i] = np.mean(dataset._data["scalars"][sextupole_offsets_y_from_daq[-i][0]][sextupole_offsets_y_from_daq[-i][1]])*1e-3
    setAllWChicaneSextupolesXOffsets(tao, sextXOffsets[0], sextXOffsets[1], sextXOffsets[2], sextXOffsets[3], sextXOffsets[4], sextXOffsets[5])
    setAllWChicaneSextupolesYOffsets(tao, sextYOffsets[0], sextYOffsets[1], sextYOffsets[2], sextYOffsets[3], sextYOffsets[4], sextYOffsets[5])

    if correctors_from_beg:
        for k, v in bmad_corrector_to_pv_map_before_dogleg.items():
            tao.cmd(f'set ele {k} BL_KICK = {np.mean(dataset._data["scalars"][v[0]][v[1]])*correctors_coef}')  # need 1/10 to transform kG-m (unit of PV) to T-m (units of BMAD)
    for k, v in bmad_corrector_to_pv_map.items():
        tao.cmd(f'set ele {k} BL_KICK = {np.mean(dataset._data["scalars"][v[0]][v[1]])*correctors_coef}')  # need 1/10 to transform kG-m (unit of PV) to T-m (units of BMAD)
    
    return tao


# bends
bmad_bend_to_pv_map = {
    # s10
    # NOT SAVED IN DAQ?
    # 'BCX10451': ["", ''],
    # 'BCX10461': ["", ''],
    # 'BCX10475': ["", ''],
    # 'BCX10481': ["", ''],
    'BX10661': ["nonBSA_List_S10", 'BEND_IN10_661_BDES'],
    'BX10751': ["nonBSA_List_S10", 'BEND_IN10_751_BDES'],
    # s11
    'BCX11314': ["nonBSA_List_S11", 'BEND_LI11_314_BCON'],
    'BCX11331': ["nonBSA_List_S11", 'BEND_LI11_331_BCON'],
    'BCX11338': ["nonBSA_List_S11", 'BEND_LI11_338_BCON'],
    'BCX11355': ["nonBSA_List_S11", 'BEND_LI11_355_BCON'],
    # s14
    # NOT SAVED IN DAQ?
    # 'BCX14720': ["", ''],
    # 'BCX14796': ["", ''],
    # 'BCX14808': ["", ''],
    # 'BCX14883': ["", ''],
    # s20
    'B1LE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_1990_BACT'],
    'WIGE1': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2420_BACT'],
    'WIGE3': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2420_BACT'],
    'B1RE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_1990_BACT'],
    'B2LE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2110_BACT'],
    'B3LE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2240_BACT'],
    'B3RE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2240_BACT'],
    'B2RE': ["nonBSA_List_S20Magnets", 'LI20_LGPS_2110_BACT'],
    'WIGE2': ["nonBSA_List_S20Magnets", 'LI20_BTRM_2420_BACT']
}


def get_mean_energy_of_some_dipoles(dataset, dipoles):
    return np.mean(np.array([np.mean(dataset._data["scalars"][bmad_bend_to_pv_map[dipole][0]][bmad_bend_to_pv_map[dipole][1]]) for dipole in dipoles]))


def save_dipole_energies_from_the_DAQ_database(dataset):
    bendEnergiesMeV=[125, 335, 4500, 10000]
    bendEnergiesMeV[0] = get_mean_energy_of_some_dipoles(dataset, ["BX10661", "BX10751"])*1e3
    bendEnergiesMeV[1] = get_mean_energy_of_some_dipoles(dataset, ["BCX11314", "BCX11331", "BCX11338", "BCX11355"])*1e3
    bendEnergiesMeV[2] = 4500 # not saved in DAQ, so just set to the default value
    bendEnergiesMeV[3] = get_mean_energy_of_some_dipoles(dataset, ["B1LE", "B1RE", "B2LE", "B3LE", "B3RE", "B2RE"])*1e3
    return bendEnergiesMeV


def edit_energy_tao_based_on_experiment_database(tao, dataset):
    '''
    Cavities:
    - The phase set to all klystrons is the same (for a given cavity). But the EPICS databases suggest that in experiment there are two klystrons with +- large phase offset. It is disregarded here.
    - The voltage is also the same, and is tuned to match the default 125, 335, 4500, 10000 MeV.
    '''
    l0a_phase = get_l0a_phase(dataset)
    l0a_ampl = get_l0a_ampl(dataset)
    
    l0b_phase = get_l0b_phase(dataset)
    l0b_ampl = get_l0b_ampl(dataset)

    #print([l0a_phase, l0a_ampl, l0b_phase, l0b_ampl])

    tao.cmd(f'set ele L0AF PHI0 = {l0a_phase / 360.}')
    tao.cmd(f'set ele L0AF VOLTAGE = {l0a_ampl}')
    
    tao.cmd(f'set ele L0BF PHI0 = {l0b_phase / 360.}')
    tao.cmd(f'set ele L0BF VOLTAGE = {l0b_ampl}')

    edited_energy_dogleg_MeV = 125
    current_e_start = tao.ele_gen_attribs('BEGINNING')["P0C"]
    current_e_dogleg = tao.ele_gen_attribs('BX10661')["P0C"]
    coef_e = 1 + 1e6*(edited_energy_dogleg_MeV - current_e_dogleg*1e-6)/(current_e_dogleg-current_e_start)
    tao.cmd(f'set ele L0AF VOLTAGE = {l0a_ampl*coef_e}')
    tao.cmd(f'set ele L0BF VOLTAGE = {l0b_ampl*coef_e}')

    l1_phase = get_l1_phase(dataset)
    l2_phase = get_l2_phase(dataset)
    l3_phase = get_l3_phase(dataset)
    setLinacPhase(tao, "L1", l1_phase)
    setLinacGradientAuto(tao, "L1", (335-125)*1e6)
    setLinacPhase(tao, "L2", l2_phase)
    setLinacGradientAuto(tao, "L2", (4500-335)*1e6)
    setLinacPhase(tao, "L3", l3_phase)
    setLinacGradientAuto(tao, "L3", (10000-4500)*1e6)
    
    return tao
