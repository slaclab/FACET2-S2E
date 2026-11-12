"""Tao initialization and beam tracking.

Moved from UTILITY_quickstart.py. These functions share the module-level
``filePathGlobal``, so they are kept together.
"""

from pytao import Tao
from os import environ
from pmd_beamphysics import ParticleGroup
from pathlib import Path
import OpenPMD_to_Bmad.Update_h5_file as pmd2bmad
import os

from ...beam.manipulation import modifyAndSaveInputBeam, centerBeam, collimateBeam, ballisticPropagation
from ...beam.microbunching import addLHmodulation
from ..impact import runImpact
from ..qpad import run_QPAD
from .lattice.set_lattice import setLattice
from .config import loadConfig


filePathGlobal = None


def initializeTao(
    filePath = None,

    # Used with autoLoadActiveFile=True (initializeTao also loads the beam)
    runSetLatticeTF = True,
    setLatticeDefaultsFile = None, 

    numMacroParticles = None,
    inputBeamFilePathSuffix = None,

    runImpactTF = False,
    runQPAD = False,

    setQPADDefaultsFile = None,
    scratchPath = None,
    randomizeFileNames = False,

    autoLoadActiveFile = True,
    # Used with autoLoadActiveFile=False (lattice only; the beam is set separately)
    loadCustomLatticeTF = False,
    latticeFile = None,

    bmad_grid_size = [32,32,64],
    lscTF = False,
    csrTF = False,
    transverseWakes = False,
    sr_wakes_on=True,
    lr_wakes_on=True,
    lsc_method="slice",
    csr_method="1_dim",
    n_bin=32,
    n_rf_steps=100,

    verbose = True,
    
    **kwargs
):

    """Initialize a Tao object

    Parameters
    ----------
    filePath : str, optional
        Path to the FACET-II lattice files. Defaults to the current working directory.

    runSetLatticeTF : bool
        Whether or not to run setLattice(). If False, the unmodified lattice specified by tao.init is loaded
    setLatticeDefaultsFile : str
        Path to the file which setLattice() loads. If not specified uses defaults.yml

    numMacroparticles : int
        The number of macroparticles to simulate
    runImpactTF : bool
        Whether or not to run IMPACT-T to generate an initial beam
    inputBeamFilePathSuffix : str
        Relative path from filePath to a file containing an intial beam
    
    runQPAD : bool
        Whether or not to run QPAD in plasma from PENT to PEXIT

    scratchPath : str
        Path to write scratch files. If used, typically set to "/tmp"
    randomizeFileNames : bool
        Add random hashes to file names. Allows for parallel operation
    
    csrTF : bool
        Enable or disable coherent synchrotron radiation (CSR) in bends    
    transverseWakes : bool
        Enable or disable transverse wakefields within linac sections

    autoLoadActiveFile : bool
        If True (default), initializeTao() also prepares the input beam (from inputBeamFilePathSuffix, or from IMPACT-T with runImpactTF), writes it to beams/activeBeamFile.h5 and loads it into Tao; the lattice is set with runSetLatticeTF/setLatticeDefaultsFile and CSR is switched with csrTF, as used by trackBeam().
        If False, only the lattice (loadCustomLatticeTF/latticeFile) and the collective effects (each one separately, see below) are set up, and the beam is loaded later, e.g. with set_beam() or get_tao_from_experiment().
        
    # For autoLoadActiveFile=False:

    loadCustomLatticeTF : bool
        Whether or not to run setLattice(). If False, the unmodified lattice specified by tao.init is loaded
    latticeFile : str
        Path to the file which setLattice() loads. If not specified uses defaults.yml. If specified, settings are added to (override) defaults.yml

    csrTF, lscTF, sr_wakes_on, lr_wakes_on : bool
        Enable or disable corresponding collective effects
    bmad_grid_size: the grid size used by SC or CSR if they are 3d.
    lsc_method: off, fft_3d or slice.
    csr_method: off, steady_state_3d or 1_dim.
    n_bin: number of longitudinal slices in the slice/1_dim methods.
    transverseWakes : bool
        Enable or disable transverse wakefields within linac sections. Impacts the SR wakes in L0, L1, and K

    n_rf_steps : int or None, optional
        RF steps per accelerating structure for all lcavities (Bmad n_rf_steps), 100 by default. None keeps the
        lattice's values: 1000, and 5000 in L0 and LI11 (facet2-lattice 2026). Every lattice recalculation tracks
        through these steps, so with the lattice's values a DAQ load with the energy feedback takes ~25 min instead of
        ~3-5 min. For E320_17418, 100 steps gave the same beam up to BEGBC20 as 1000/5000; Bmad's default of 10 gave a
        2% shorter bunch after BC14.

    IMPACT is disabled in this function! Run it separately if needed.
    
    # All:

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
        filePath = str(Path(__file__).parent.parent.parent.parent.parent)

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

    if n_rf_steps is not None:
        tao.cmd(f"set ele lcavity::* n_rf_steps = {int(n_rf_steps)}")
    
    tao.cmd("set beam add_saved_at = DTOTR, XTCAVF, M2EX, PR10571, PR10711, CN2069, YCWIGE") #The beam is saved at all MARKER elements already; this list just supplements

    if autoLoadActiveFile:
        #tao.cmd(f'set beam_init track_end = {lastTrackedElement}') #See track_start and track_end values with `show beam`
        tao.cmd(f'set beam_init track_end = end')
        #print(f"Tracking to {lastTrackedElement}")

        tao.cmd(f'call {filePath}/bmad/models/f2_elec/scripts/Activate_CSR.tao')
        if csrTF: 
            tao.cmd('csron')
            print("CSR on")
        else:
            tao.cmd('csroff')
            print("CSR off")


        if runSetLatticeTF:
            print("Overwriting lattice with setLattice() defaults")
            setLattice(tao, verbose = True,  defaultsFile = setLatticeDefaultsFile) #Set lattice to my latest default config
            
        else:
            print("Not using setLattice(). Golden lattice")
    

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
        
        if runImpactTF:
            if not numMacroParticles:
                print("Define numMacroParticles to run Impact")
                return
                    
            runImpact(
                filePath = filePath,
                numMacroParticles = numMacroParticles,
                **kwargs
            )

            inputBeamFilePath = f'{filePath}/beams/ImpactBeam.h5'

        else:
            if inputBeamFilePathSuffix:
                inputBeamFilePath = f'{filePath}{inputBeamFilePathSuffix}'
                
            else: #If tracking wasn't requested and a beamfile wasn't specified just grab a random beam... assume the user only wants to do single-particle sims
                print("WARNING! No beam file is specified!")
                #inputBeamFilePath = f'{filePath}/beams/activeBeamFile.h5'
                inputBeamFilePath = f'{filePath}/beams/L0AFEND_facet2-lattice.h5'

            if numMacroParticles:
                print(f"Number of macro particles = {numMacroParticles}")
            else:
                print(f"Number of macro particles defined by input file")

        #Create the beam
        modifyAndSaveInputBeam(
                inputBeamFilePath,
                numMacroParticles = (None if runImpactTF else numMacroParticles),
                outputBeamFilePath = activeFilePath
        )

        tao.cmd(f'set beam_init position_file={activeFilePath}')
        tao.cmd('reinit beam')
        print(f"Beam created, written to {activeFilePath}, and reinit to tao")


        #Save things into the tao object
        tao.inputBeamFilePath = inputBeamFilePath
        tao.activeFilePath = activeFilePath
        tao.patchFilePath = patchFilePath
        tao.qpadSimPath = qpadSimPath
        tao.runQPAD = runQPAD
        tao.QPADDefaultsFile = setQPADDefaultsFile
        #tao.activeBeam = activeBeam
    
    # autoLoadActiveFile=False: lattice and collective effects only; the beam is set separately
    else:
        # Lattice
        if loadCustomLatticeTF:
            if verbose:
                print("Overwriting lattice with setLattice()")
            if latticeFile is not None:
                importedSettings = loadConfig(latticeFile, filePathGlobal)
                setLattice(tao, filePath=filePath, **importedSettings)
            else:
                setLattice(tao, verbose = True) # settings from setLattice_configs/defaults.yml
            
        else:
            if verbose:
                print("Base Tao lattice")
    
        # tao calculation methods, aperture
        #tao.cmd(f'set global rad_int_calc_on = T')
        tao.cmd(f'set global lattice_calc_on = T')
        tao.cmd(f'set bmad_com aperture_limit_on = F')
        
        # collective effects
        tao = applyBMADCollectiveEffectSettings(tao=tao, csrTF=csrTF, lscTF=lscTF, sr_wakes_on=sr_wakes_on, lr_wakes_on=lr_wakes_on, bmad_grid_size=bmad_grid_size, verbose=verbose, lsc_method=lsc_method, csr_method=csr_method, n_bin=n_bin)

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
    filepath,
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
    autoLoadActiveFile = True,
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

    if autoLoadActiveFile:
            tao.cmd(f'set beam_init position_file={tao.activeFilePath}')
            tao.cmd('reinit beam')
            if verbose: print(f"Loaded {tao.activeFilePath}")
    
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


    # Update z?


def trackBeamHelper(tao):
    """Wrap some of the tao commands with a try/except. This way if tracking doesn't work, we failsafe to track_type = single"""
    try:
        tao.cmd('set global track_type = beam') #set "track_type = single" to return to single particle
    except:
        print("Beam tracking failed. Resetting track_type = single")
        tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation
        raise #Rethrow the error. Adding this line causes trackBeam() to see the error and also fail instead of potentially entering a weird state

    tao.cmd('set global track_type = single') #return to single to prevent accidental long re-evaluation

    return


        

def getBeamAtElement(tao, eleString, tToZ = True):
    """Queries tao for the beam at an element
    
    Parameters
    ----------
    tao : pytao object

    eleString : str, int
        Either the name or lattice index of the element where the beam is to be found

    tToZ : bool
        Set P.z to -ct

    Returns
    -------
    ParticleGroup beam
    """
    
    P = ParticleGroup(data=tao.bunch_data(eleString))
    P = P[P.status == 1]

    #Naive implementation for "typical" beams. ParticleGroup has .drift_to_z but I couldn't get it to work...
    if tToZ:
        P.z = -299792458 * P["delta_t"]
        #P.t = 0 * P.t #I haven't decided the best practice for this yet. Technically the beam is not self-consistent without t being set to zero but not doing so is convenient for backwards compatibility
        
    return P


def writeBeam(P, fileName):
    """ Writes the beam as an h5 with E. Cropp's timeOffset fix """
    P.write(fileName)
    
    pmd2bmad.OpenPMD_to_Bmad(fileName)


def makeBeamActiveBeamFile(P, tao = None):

    global filePathGlobal

    if tao:
        writeBeam(P, tao.activeFilePath)
    else:
        print(f"WARNING! No tao object provided. Writing beam to {filePathGlobal}/beams/activeBeamFile.h5... hope that's what you wanted")
        writeBeam(P, f"{filePathGlobal}/beams/activeBeamFile.h5")
