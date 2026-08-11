from pytao import Tao
from pmd_beamphysics import ParticleGroup
import pmd_beamphysics.statistics

import numpy as np
import random

from FACET2_S2E.UTILITY_quickstart import loadConfig, trackBeamHelper
from FACET2_S2E.UTILITY_setLattice import setLattice, getBendkG, getQuadkG, getSextkG, setBendkG, setQuadkG, setSextkG, setXOffset, setYOffset, setKickerkG, getKickerkG, setBendGeVc, getBendGeVc
from FACET2_S2E.UTILITY_QPAD import QPAD_sim, run_QPAD

from FACET2_S2E.UTILITY_plotMod import plotMod, slicePlotMod, floorplanPlot
from FACET2_S2E.UTILITY_linacPhaseAndAmplitude import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto

from pathlib import Path
from os import path,environ
import os

filePathGlobal = None

def initializeTao(
    filePath = None,
    
    loadCustomLatticeTF = False,
    latticeFile = None,

    runQPAD = False,
    setQPADDefaultsFile = None,
    scratchPath = None,
    randomizeFileNames = False,
    
    bmad_grid_size = [32,32,64],
    lscTF = False,
    csrTF = False,
    transverseWakes = False,
    sr_wakes_on=True,
    lr_wakes_on=True,
    lsc_method="slice",
    csr_method="1_dim",
    n_bin=32,

    verbose = True,
    
    **kwargs
):

    """Initialize a Tao object

    Parameters
    ----------
    filePath : str, optional
        Path to the FACET-II lattice files. Defaults to the current working directory.

    loadCustomLatticeTF : bool
        Whether or not to run setLattice(). If False, the unmodified lattice specified by tao.init is loaded
    latticeFile : str
        Path to the file which setLattice() loads. If not specified uses defaults.yml. If specified, settings are added to (override) defaults.yml
    
    runQPAD : bool
        Whether or not to run QPAD in plasma from PENT to PEXIT

    scratchPath : str
        Path to write scratch files. If used, typically set to "/tmp"
    randomizeFileNames : bool
        Add random hashes to file names. Allows for parallel operation
    
    csrTF, lscTF, sr_wakes_on, lr_wakes_on : bool
        Enable or disable corresponding collective effects
    bmad_grid_size: the grid size used by SC or CSR if they are 3d.
    lsc_method: off, fft_3d or slice.
    csr_method: off, steady_state_3d or 1_dim.
    n_bin: number of longitudinal slices in the slice/1_dim methods.
    transverseWakes : bool
        Enable or disable transverse wakefields within linac sections. Impacts the SR wakes in L0, L1, and K

    IMPACT is disabled in this function! Run it separately if needed.
        
    Returns
    -------
    Tao
        Configured Tao object ready for beam tracking.
    """

    
    #######################################################################
    #Set file path
    #######################################################################
    global filePathGlobal
    
    if not filePath:
        filePath = str(Path(__file__).parent.parent.parent)

    if not scratchPath:
        scratchPath = filePath


    
    os.environ['FACET2_LATTICE'] = filePath
    filePathGlobal = filePath
    
    if verbose:
        print('Environment set to: ', environ['FACET2_LATTICE']) 

    
    #######################################################################
    #Launch and configure Tao
    #######################################################################
    if transverseWakes: 
        if verbose:
            print("Transverse wakes enabled!")
        tao=Tao('-init {:s}/bmad/models/f2_elec/tao_transverseWakesOn.init -noplot'.format(environ['FACET2_LATTICE'])) 
    else:
        tao=Tao('-init {:s}/bmad/models/f2_elec/tao.init -noplot'.format(environ['FACET2_LATTICE'])) 
        if verbose:
            print('-init {:s}/bmad/models/f2_elec/tao.init -noplot'.format(environ['FACET2_LATTICE']))

    tao.filePathGlobal = filePathGlobal #Put this into the tao object immediately. Needed early in the initialization
    
    tao.cmd("set beam add_saved_at = DTOTR, XTCAVF, M2EX, PR10571, PR10711, CN2069, YCWIGE, BEGL1F, ENDL1F, BC11CBEG, BC11CEND, BPM10425, PR10465, PR10471, BPM10511, BPM10525, PR10571, BPM10581, BZ10596, BPM10631, BPM10651, PR10711, BPM10731") #The beam is saved at all MARKER elements already; this list just supplements
    #tao.cmd("set beam saved_at = MARKER::*, BPM10731")

    # Lattice
    if loadCustomLatticeTF:
        if verbose:
            print("Overwriting lattice with setLattice()")
        if latticeFile is not None:
            importedSettings = loadConfig(latticeFile, filePathGlobal)
            setLattice(tao, filePath=filePath, **importedSettings)
        else:
            setLattice(tao, verbose = True) #Set lattice to Nathan's latest default config
        
    else:
        if verbose:
            print("Base Tao lattice")

    # tao calculation methods, aperture
    #tao.cmd(f'set global rad_int_calc_on = T')
    tao.cmd(f'set global lattice_calc_on = T')
    tao.cmd(f'set bmad_com aperture_limit_on = F')
    
    # collective effects
    tao = applyBMADCollectiveEffectSettings(tao=tao, csrTF=csrTF, lscTF=lscTF, sr_wakes_on=sr_wakes_on, lr_wakes_on=lr_wakes_on, bmad_grid_size=bmad_grid_size, verbose=verbose, lsc_method=lsc_method, csr_method=csr_method, n_bin=n_bin)
    

    #######################################################################
    #Import or generate input beam file
    #######################################################################


    if randomizeFileNames:
        #True-random path for this particular instance
        randomPath = str(int.from_bytes(os.urandom(8), "big"))
        activeFilePath = f'{scratchPath}/beams/activeBeamFile_{randomPath}.h5'
        patchFilePath = f'{scratchPath}/beams/patchBeamFile_{randomPath}.h5'
        qpadSimPath = f'{scratchPath}/beams/qpad_sim_{randomPath}'
    else:
        activeFilePath = f'{scratchPath}/beams/activeBeamFile.h5'
        patchFilePath = f'{scratchPath}/beams/patchBeamFile.h5'
        qpadSimPath = f'{scratchPath}/beams/qpad_sim'

    # Create 'beams' folder if it doesn't exist
    os.makedirs(f"{scratchPath}/beams", exist_ok=True)

    # create 'qpad' sim folder if it doesn't exist
    if(run_QPAD):
        os.makedirs(qpadSimPath, exist_ok=True)

    
    #Save things into the tao object
    tao.activeFilePath = activeFilePath
    tao.patchFilePath = patchFilePath
    tao.qpadSimPath = qpadSimPath
    tao.runQPAD = runQPAD
    tao.QPADDefaultsFile = setQPADDefaultsFile
    #tao.activeBeam = activeBeam


    return tao


def applyBMADCollectiveEffectSettings(tao=None, csrTF=False, lscTF=False, bmad_grid_size=[32,32,64], sr_wakes_on=True, lr_wakes_on=True,
                                      verbose=True, lsc_method="slice", csr_method="1_dim", n_bin=32):
    '''
    LSC: off, fft_3d or slice
    CSR: off, steady_state_3d or 1_dim
    '''
    # CSR and SC
    tao.cmd(f'call {filePathGlobal}/bmad/models/f2_elec/scripts/Activate_CSR.tao')
    if csrTF:
        tao.cmd(f'set bmad_com csr_and_space_charge_on = T')
        if not lscTF:
            tao.cmd(f'set ele * space_charge_method = off')
            tao.cmd(f'set ele * csr_method = {csr_method}')
            if verbose:
                print('CSR on, SC off')
        else:
            tao.cmd(f'set ele * space_charge_method = {lsc_method}')
            tao.cmd(f'set ele * csr_method = {csr_method}')
            if verbose:
                print('CSR on, SC on')
    else:
        tao.cmd(f'set ele * csr_method = off')
        if not lscTF:
            tao.cmd(f'set bmad_com csr_and_space_charge_on = F')
            tao.cmd(f'set ele * space_charge_method = off')
            tao.cmd(f'set ele * csr_method = off')
            if verbose:
                print('CSR off, SC off')
        else:
            tao.cmd(f'set bmad_com csr_and_space_charge_on = T')
            tao.cmd(f'set ele * space_charge_method = {lsc_method}')
            tao.cmd(f'set ele * csr_method = off')
            if verbose:
                print('CSR off, SC on')

    # Collective effect calculation: Wakes, SC and CSR mech sizes
    tao.cmd(f'set bmad_com lr_wakes_on = {"T" if lr_wakes_on else "F"}')
    tao.cmd(f'set bmad_com sr_wakes_on = {"T" if sr_wakes_on else "F"}')
    tao.cmd(f'set space_charge_com space_charge_mesh_size = {" ".join(list(map(str, bmad_grid_size)))}')
    tao.cmd(f'set space_charge_com csr3d_mesh_size = {" ".join(list(map(str, bmad_grid_size)))}')
    tao.cmd(f"set space_charge_com {'n_bin'} = {n_bin}")
    tao.cmd(f"set space_charge_com {'ds_track_step'} = {1e-2}")

    # tao calculation methods, aperture
    #tao.cmd(f'set global rad_int_calc_on = T')
    tao.cmd(f'set global lattice_calc_on = T')
    tao.cmd(f'set bmad_com aperture_limit_on = F')
    return tao


    
# def reinitActiveBeam(tao):
#     #Take the beam stored in the tao object (tao.activeBeam), save it to a file, load and reinit tao with that file
    
#     (tao.activeBeam).write(tao.activeFilePath)
    
#     tao.cmd(f'set beam_init position_file={tao.activeFilePath}')
#     tao.cmd('reinit beam')
#     #os.remove(tao.activeFilePath)    

# def reinitPatchBeam(tao, P):
#     #Take the provided beam, save it to a file, load and reinit tao with that file
    
#     P.write(tao.patchFilePath)
    
#     tao.cmd(f'set beam_init position_file={tao.patchFilePath}')
#     tao.cmd('reinit beam')
#     #os.remove(tao.patchFilePath)


def trackBeam(
    tao,
    trackStart = "L0AFEND",
    trackEnd = "end",
    laserHeater = False,
    centerDL10 = False,
    centerBC14 = False,
    assertBC14Energy = False,
    centerBC20 = False,
    assertBC20Energy = False,
    allCollimatorRules = None,
    centerMFFF = False,
    verbose = False,
    plasmaSIM = False,
    **kwargs,
):
    """Tracks the beam in activeBeamFile.h5 through the lattice presently in tao from trackStart to trackEnd

    Some special options are available but disabled by default
    * Centering
     * At some selected treaty points, remove net offsets to transverse position and angle
    * Assert energy
     * Centering must be enabled. Can either set True (for default energy at that point) or the desired energy in eV. This is effectively a virtual energy feedback
    * Laser heater
     * Refer to addLHmodulation(). Need to pass additional options to trackBeam() as **kwargs
    * BC20 collimators
     * Refer to collimateBeam(). Collimator positions passed as allCollimatorRules
    """
    global filePathGlobal

    tao.cmd(f'set beam_init track_start = {trackStart}')
    tao.cmd(f'set beam_init track_end = {trackEnd}')
    if verbose: print(f"Set track_start = {trackStart}, track_end = {trackEnd}")


    #Adding S-location checks so center* commands won't trigger unnecessarily 
    trackStartS  = tao.ele_param(trackStart,"ele.s")['ele_s']
    trackEndS    = tao.ele_param(trackEnd,"ele.s")['ele_s']
    laserHeaterS = tao.ele_param("HTRUNDF","ele.s")['ele_s']
    DL10ENDS     = tao.ele_param("ENDDL10","ele.s")['ele_s']
    BC14BEGS     = tao.ele_param("BEGBC14_1","ele.s")['ele_s']
    BC20BEGS     = tao.ele_param("BEGBC20","ele.s")['ele_s']
    BC20COLLS    = tao.ele_param("CN2069","ele.s")['ele_s']
    PENTS        = tao.ele_param("PENT","ele.s")['ele_s']
    MFFFS        = tao.ele_param("MFFF","ele.s")['ele_s']
    PEXITS        = tao.ele_param("PEXT","ele.s")['ele_s']
    
    if laserHeater and trackStartS < laserHeaterS < trackEndS:
        #Will track from start to HTRUNDF, get the beam, modify it, export it, import it, update track_start and track_end
        tao.cmd(f'set beam_init track_end = HTRUNDF')
        if verbose: print(f"Set track_end = HTRUNDF")
        
        if verbose: print(f"Tracking!")
        trackBeamHelper(tao)

        P = getBeamAtElement(tao, "HTRUNDF", tToZ = False)

        PAfterLHmodulation, deltagamma, t = addLHmodulation(P, **kwargs,);
        
        writeBeam(PAfterLHmodulation, tao.patchFilePath)
        if verbose: print(f"Beam with LH modulation written to {tao.patchFilePath}")

        tao.cmd(f'set beam_init position_file={tao.patchFilePath}')
        tao.cmd('reinit beam')
        if verbose: print(f"Loaded {tao.patchFilePath}")

        tao.cmd(f'set beam_init track_start = HTRUNDF')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = HTRUNDF, track_end = {trackEnd}")

    if centerDL10 and trackStartS < DL10ENDS < trackEndS:
        tao.cmd(f'set beam_init track_end = ENDDL10')
        if verbose: print(f"Set track_end = ENDDL10")

        if verbose: print(f"Tracking!")
        trackBeamHelper(tao)

        P = getBeamAtElement(tao, "ENDDL10", tToZ = False)

        PMod = centerBeam(P)
        
        writeBeam(PMod, tao.patchFilePath)
        if verbose: print(f"Beam centered at ENDDL10 written to {tao.patchFilePath}")

        tao.cmd(f'set beam_init position_file={tao.patchFilePath}')
        tao.cmd('reinit beam')
        if verbose: print(f"Loaded {tao.patchFilePath}")

        tao.cmd(f'set beam_init track_start = ENDDL10')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = ENDDL10, track_end = {trackEnd}")
    
    if centerBC14 and trackStartS < BC14BEGS < trackEndS:
        tao.cmd(f'set beam_init track_end = BEGBC14_1')
        if verbose: print(f"Set track_end = BEGBC14_1")

        if verbose: print(f"Tracking!")
        trackBeamHelper(tao)

        P = getBeamAtElement(tao, "BEGBC14_1", tToZ = False)

        if assertBC14Energy:
            if type(assertBC14Energy) is bool: 
                assertBC14Energy = 4.5e9
            if verbose: print(f"""Also setting BC14 energy = {1e-9 * assertBC14Energy} GeV, from {1e-9 * P["mean_energy"]} GeV""")
            PMod = centerBeam(P, assertEnergy = assertBC14Energy)
        else:
            PMod = centerBeam(P)
        
        writeBeam(PMod, tao.patchFilePath)
        if verbose: print(f"Beam centered at BEGBC14 written to {tao.patchFilePath}")

        tao.cmd(f'set beam_init position_file={tao.patchFilePath}')
        tao.cmd('reinit beam')
        if verbose: print(f"Loaded {tao.patchFilePath}")

        tao.cmd(f'set beam_init track_start = BEGBC14_1')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = BEGBC14_1, track_end = {trackEnd}")

    if centerBC20 and trackStartS < BC20BEGS < trackEndS:
        tao.cmd(f'set beam_init track_end = BEGBC20')
        if verbose: print(f"Set track_end = BEGBC20")

        if verbose: print(f"Tracking!")
        trackBeamHelper(tao)

        P = getBeamAtElement(tao, "BEGBC20", tToZ = False)

        if assertBC20Energy:
            if type(assertBC20Energy) is bool: 
                assertBC20Energy = 10e9
            if verbose: print(f"""Also setting BC20 energy = {1e-9 * assertBC20Energy} GeV, from {1e-9 * P["mean_energy"]} GeV""")
            PMod = centerBeam(P, assertEnergy = assertBC20Energy)
        else:
            PMod = centerBeam(P)
        
        writeBeam(PMod, tao.patchFilePath)
        if verbose: print(f"Beam centered at BEGBC20 written to {tao.patchFilePath}")

        tao.cmd(f'set beam_init position_file={tao.patchFilePath}')
        tao.cmd('reinit beam')
        if verbose: print(f"Loaded {tao.patchFilePath}")

        tao.cmd(f'set beam_init track_start = BEGBC20')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = BEGBC20, track_end = {trackEnd}")


    if allCollimatorRules and trackStartS < BC20COLLS < trackEndS:
        tao.cmd(f'set beam_init track_end = CN2069')
        if verbose: print(f"Set track_end = CN2069")

        if verbose: print(f"Tracking!")
        trackBeamHelper(tao)

        P = getBeamAtElement(tao, "CN2069", tToZ = False)

        PMod = collimateBeam(P, allCollimatorRules)
        
        writeBeam(PMod, tao.patchFilePath)
        if verbose: print(f"Collimated beam written to {tao.patchFilePath}. Rules: {allCollimatorRules}")

        tao.cmd(f'set beam_init position_file={tao.patchFilePath}')
        tao.cmd('reinit beam')
        if verbose: print(f"Loaded {tao.patchFilePath}")

        tao.cmd(f'set beam_init track_start = CN2069')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = CN2069, track_end = {trackEnd}")


    if centerMFFF and trackStartS < MFFFS < trackEndS:
        tao.cmd(f'set beam_init track_end = MFFF')
        if verbose: print(f"Set track_end = MFFF")

        if verbose: print(f"Tracking!")
        trackBeamHelper(tao)

        P = getBeamAtElement(tao, "MFFF", tToZ = False)

        PMod = centerBeam(P)
        
        writeBeam(PMod, tao.patchFilePath)
        if verbose: print(f"Beam centered at MFFF written to {tao.patchFilePath}")

        tao.cmd(f'set beam_init position_file={tao.patchFilePath}')
        tao.cmd('reinit beam')
        if verbose: print(f"Loaded {tao.patchFilePath}")

        tao.cmd(f'set beam_init track_start = MFFF')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = MFFF, track_end = {trackEnd}")


    if plasmaSIM and trackStartS < PEXITS < trackEndS:
        ## propagate to PEXIT
        tao.cmd(f'set beam_init track_end = PEXT')
        if verbose: print(f"Set track_end = PEXT")

        if verbose: print(f"Tracking!")
        trackBeamHelper(tao)

        P = getBeamAtElement(tao, "PENT", tToZ = False)

        
        PENT_to_plasma = 0.25 # todo: specify in lattice config

        # ballistic propagation from PENT to plasma
        ballisticPropagation(P, PENT_to_plasma) 
        # run plasma simulation
        P2, lsim = run_QPAD(tao, P, defaultsFile = f"{filepath}/" + tao.QPADDefaultsFile)
        # ballistic propagation from plasma to PEXIT
        ds = max(PEXITS - (PENTS + PENT_to_plasma + lsim), 0.0)
        ballisticPropagation(P2, ds)
        writeBeam(P2, tao.patchFilePath)
        
        tao.cmd(f'set beam_init position_file={tao.patchFilePath}')
        tao.cmd('reinit beam')
        if verbose: print(f"Loaded {tao.patchFilePath}")

        tao.cmd(f'set beam_init track_start = PEXT')
        tao.cmd(f'set beam_init track_end = {trackEnd}')
        if verbose: print(f"Set track_start = PEXT, track_end = {trackEnd}")


    if verbose: print(f"Tracking!")
    trackBeamHelper(tao)

    if verbose: print(f"trackBeam() exiting")


    #For backwards compatibility, return to activeBeamFile. Might be unnecessary
    # tao.cmd(f'set beam_init position_file={filePathGlobal}/beams/activeBeamFile.h5')
    # tao.cmd('reinit beam')

# def trackBeamLEGACY(tao):
#     #This is the pre-2024-08-23 version of trackBeam(), retained for debugging purposes. Can be deleted
    
#     tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
#     tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation
        

def getBeamAtElement(tao, eleString):
    P = ParticleGroup(data=tao.bunch_data(eleString))
    P = P[P.status == 1]
    return P

