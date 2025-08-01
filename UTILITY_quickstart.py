from pytao import Tao
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.patches as mpatches
import re
import io
from os import path,environ
import pandas as pd
import random
from IPython.display import display, clear_output, update_display
import bayes_opt
import shutil
from scipy.special import jn as besselj
from scipy.interpolate import interp1d
import json

import scipy
from scipy.optimize import curve_fit

from pmd_beamphysics import ParticleGroup
#from pmd_beamphysics.statistics import resample_particles
import pmd_beamphysics.statistics

from UTILITY_plotMod import plotMod, slicePlotMod, floorplanPlot
from UTILITY_linacPhaseAndAmplitude import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto
from UTILITY_modifyAndSaveInputBeam import modifyAndSaveInputBeam
from UTILITY_setLattice import setLattice, getBendkG, getQuadkG, getSextkG, setBendkG, setQuadkG, setSextkG, setXOffset, setYOffset, setKickerkG, getKickerkG, setBendGeVc, getBendGeVc
from UTILITY_impact import runImpact
from UTILITY_OpenPMDtoBmad import OpenPMD_to_Bmad
from UTILITY_finalFocusSolver import finalFocusSolver

import os
import yaml

filePathGlobal = None

def initializeTao(
    filePath = None,
    
    runSetLatticeTF = True,
    setLatticeDefaultsFile = None, 
    
    numMacroParticles = None,
    runImpactTF = False,
    inputBeamFilePathSuffix = None,
    
    scratchPath = None,
    randomizeFileNames = False,
    
    csrTF = False,
    transverseWakes = False,
    
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
    
    scratchPath : str
        Path to write scratch files. If used, typically set to "/tmp"
    randomizeFileNames : bool
        Add random hashes to file names. Allows for parallel operation
    
    csrTF : bool
        Enable or disable coherent synchrotron radiation (CSR) in bends    
    transverseWakes : bool
        Enable or disable transverse wakefields within linac sections

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
        filePath = os.getcwd()

    if not scratchPath:
        scratchPath = filePath


        
    os.environ['FACET2_LATTICE'] = filePath
    filePathGlobal = filePath
    
    print('Environment set to: ', environ['FACET2_LATTICE']) 

    
    #######################################################################
    #Launch and configure Tao
    #######################################################################
    if transverseWakes: 
        print("Transverse wakes enabled!")
        tao=Tao('-init {:s}/bmad/models/f2_elec/tao_transverseWakesOn.init -noplot'.format(environ['FACET2_LATTICE'])) 
    else:
        tao=Tao('-init {:s}/bmad/models/f2_elec/tao.init -noplot'.format(environ['FACET2_LATTICE'])) 

    tao.filePathGlobal = filePathGlobal #Put this into the tao object immediately. Needed early in the initialization
    
    tao.cmd("set beam add_saved_at = DTOTR, XTCAVF, M2EX, PR10571, PR10711, CN2069, YCWIGE") #The beam is saved at all MARKER elements already; this list just supplements


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
    else:
        activeFilePath = f'{scratchPath}/beams/activeBeamFile.h5'
        patchFilePath = f'{scratchPath}/beams/patchBeamFile.h5'

    # Create 'beams' folder if it doesn't exist
    os.makedirs(f"{scratchPath}/beams", exist_ok=True)
    
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
    #tao.activeBeam = activeBeam


    


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
    
    
    centerDL10 = False,
    centerBC14 = False,
    centerBC20 = False,
    centerMFFF = False,
    
    assertBC14Energy = False,
    assertBC20Energy = False,
    
    laserHeater = False,
    allCollimatorRules = None,
    
    verbose = False,
    **kwargs,
):
    """Tracks the beam in activeBeamFile.h5 through the lattice presently in tao from trackStart to trackEnd
    
    Parameters
    ----------
    tao : pytao object
    
    trackStart, trackEnd : str
        Element names where tracking starts and ends

    center* : bool
        Option to dump the beam at that location, center it, and reload it

    assert*Energy : bool, float
        Option to also correct the median beam energy. If True, uses default values (e.g. 4.5 GeV in BC14, 10 GeV in BC20). Sets to arbitrary value if float

    laserHeater : dict
        If not False, laser heater is enabled. Dictionary is passed to addLHmodulation(). See addLHmodulation() for options

    allCollimatorRules : list of lists
        If not False, BC20 notch and/or jaw collimators enabled. list is passed to collimateBeam(). See collimateBeam() for details

    verbose : bool
        Print progress info
    """
    
    global filePathGlobal


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
    MFFFS        = tao.ele_param("MFFF","ele.s")['ele_s']

    
    if laserHeater and trackStartS < laserHeaterS < trackEndS:
        #Will track from start to HTRUNDF, get the beam, modify it, export it, import it, update track_start and track_end
        tao.cmd(f'set beam_init track_end = HTRUNDF')
        if verbose: print(f"Set track_end = HTRUNDF")
        
        if verbose: print(f"Tracking!")
        trackBeamHelper(tao)

        P = getBeamAtElement(tao, "HTRUNDF", tToZ = False)

        PAfterLHmodulation, deltagamma, t = addLHmodulation(P, **laserHeater);
        
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

def nudgeMacroparticleWeights(
    PInput,
    trailingBunchFraction = None,
    trailingBunchType = None
):
    """
    This is NOT a robust function. Don't trust it to do what you want
    Presently splits based on z and a user-specified charge ratio. Lots of things can go wrong if you aren't careful!
    
    Borrowing stuff from 2024-03-29_nudgeMacroparticleWeights.ipynb
    """

    P = PInput.copy()

    zVals = (P["delta_z"]).copy()
    zVals = np.sort(zVals)
    
    splitZ = zVals[int(trailingBunchFraction * len(zVals))] 
    
    
    
    startingWeight = P.weight[0]
    startingWeight
    
    witnessWeight = 0.999*startingWeight
    driverWeight = 1.001*startingWeight
    
    if trailingBunchType == "witness":
        trailingBunchWeight = witnessWeight
        leadingBunchWeight = driverWeight
    if trailingBunchType == "driver": 
        trailingBunchWeight = driverWeight
        leadingBunchWeight = witnessWeight
    
    newWeightArr = np.full(np.size(P.weight), -1.1)
    for i in range(np.size(newWeightArr)):
        if P["delta_z"][i] < splitZ:
            newWeightArr[i] = trailingBunchWeight
        else:
            newWeightArr[i] = leadingBunchWeight
    
    P.weight = newWeightArr

    return P

def getDriverAndWitness(P):
    """Splits a beam by unique weights into  drive and witness beam objects
    
    See, e.g. "2024-07-01 Nudge Macroparticle Weights.ipynb" for details
    """
    
    weights = np.sort(np.unique(P.weight))
    if len(weights) != 2:
        print("WARNING! Expected drive/witness structure not found")
        return
    PWitness = P[P.weight == weights[0]]
    PDrive = P[P.weight == weights[1]]
    return PDrive, PWitness

def writeBeam(P, fileName):
    """ Writes the beam as an h5 with E. Cropp's timeOffset fix """
    P.write(fileName)
    
    OpenPMD_to_Bmad(fileName)

def makeBeamActiveBeamFile(P, tao = None):

    global filePathGlobal

    if tao:
        writeBeam(P, tao.activeFilePath)
    else:
        print(f"WARNING! No tao object provided. Writing beam to {filePathGlobal}/beams/activeBeamFile.h5... hope that's what you wanted")
        writeBeam(P, f"{filePathGlobal}/beams/activeBeamFile.h5")

def smallestInterval(nums, percentage=0.9):
    """Give the smallest interval containing a desired percentage of provided points"""
    nums = nums.copy()  # Create a copy to avoid modifying the original list
    nums.sort()
    n = len(nums)
    k = int(n * percentage)
    min_range = float('inf')
    interval = (None, None)
    
    for i in range(n - k + 1):
        current_range = nums[i + k - 1] - nums[i]
        if current_range < min_range:
            min_range = current_range
            interval = (nums[i], nums[i + k - 1])
    
    return interval[1]-interval[0]


def smallestIntervalImpliedSigma(nums, percentage=0.9):
    """Given a smallestInterval, infer the sigma assuming the distribution was Gaussian

    See "Discussion of alternative emittance and spot size calculations.ipynb"
    See also "2024-07-01 RMS vs FWHM at PENT.ipynb"
    """
    interval = smallestInterval(nums, percentage)
    intervalToSigmaFactor = scipy.special.erfinv(percentage) * (2 * np.sqrt(2))
    return interval/intervalToSigmaFactor

def smallestIntervalImpliedEmittanceModelFunction(z, sigmax, sigmaxp, rho):

    return np.sqrt(np.clip(
        sigmax**2 + 2 * z * rho * sigmax * sigmaxp + z**2 * sigmaxp**2,
        0, np.inf))


def smallestIntervalImpliedEmittance(P, plane = "x", percentage = 0.9, verbose = False):
    """Use a virtual quad scan and smallestInterval measurements to calculate a beam's emittance"""
    
    #zValues = np.arange(-20, 20, 0.1)
    zValues = np.array([-131.072, -65.536, -32.768, -16.384, -8.192, -4.096, -2.048, -1.024, 
               -0.512, -0.256, -0.128, -0.064, -0.032, -0.016, -0.008, -0.004, 
               -0.002, -0.001, 0.0, 0.001, 0.002, 0.004, 0.008, 0.016, 0.032, 
               0.064, 0.128, 0.256, 0.512, 1.024, 2.048, 4.096, 8.192, 
               16.384, 32.768, 65.536, 131.072]) #(-Reverse[#])~Join~{0}~Join~# &@PowerRange[0.001, 200, 2]
    
    if plane == "x":
        sigmaXResults = [ smallestIntervalImpliedSigma(P.x + z * P.xp, percentage = percentage) for z in zValues]
        sigmaXResultsExact = [ np.std(P.x + z * P.xp) for z in zValues]
    elif plane == "y":
        sigmaXResults = [ smallestIntervalImpliedSigma(P.y + z * P.yp, percentage = percentage) for z in zValues]
        sigmaXResultsExact = [ np.std(P.y + z * P.yp) for z in zValues]
    else:
        return

    #sigmaXResults = [ smallestIntervalImpliedSigma(P.x + z * P.xp, percentage = percentage) for z in zValues]
    #sigmaXResultsExact = [ np.std(P.x + z * P.xp) for z in zValues]
    
    
    # Fit the model to the data
    popt, pcov = curve_fit(smallestIntervalImpliedEmittanceModelFunction, zValues, sigmaXResults, p0=[1, 1, 0])
    
    # Extract optimal parameters
    sigmax_opt, sigmaxp_opt, rho_opt = popt


    
    emit_opt = np.sqrt(np.clip( 
        sigmax_opt**2 * sigmaxp_opt**2 - (rho_opt * sigmax_opt * sigmaxp_opt)**2 ,
        0, np.inf))


    if verbose:
        print(f"""True sigma_x, sigma_xp, rho: {P.std("x")}, {P.std("xp")}, {P["cov_x__xp"] / (P.std("x") * P.std("xp"))}""")
        print(f"Optimizer parameters: sigma_x = {sigmax_opt}, sigma_xp = {sigmaxp_opt}, rho = {rho_opt}")
    

        plt.scatter(zValues, sigmaXResultsExact, label='True rms')
        plt.scatter(zValues, sigmaXResults, label='Inferred rms')
        plt.plot(zValues, smallestIntervalImpliedEmittanceModelFunction(zValues, *popt), label='Fitted function', color='red')
        plt.xlabel('Drift [m]')
        plt.ylabel('Sigma_x [m]')
        plt.legend()
        plt.show()

        print(f"""Actual emittance: \t {P["norm_emit_x"]}""")
        print(f"""Fit emittance: \t\t {emit_opt * P["mean_gamma"]}""")

    return emit_opt * P["mean_gamma"]

def emittance(P, plane = "x", fraction = 0.9):
    """Just a wrapper for the OpenPMD functions which let you specify the fraction used in the emittance calculation"""
    return P.twiss(plane = plane, fraction = fraction)[f"norm_emit_{plane}"]
    
def displayMatrix(matrix):
    display(pd.DataFrame(matrix).style.hide(axis="index").hide(axis="columns"))

def getMatrix(tao, start, end, order = 1, startOffset = 0, endOffset = 0, print = False):
    """Return zero or first order transport matrix from start to end, with offsets. Optionally print in a human readable format"""
    
    startS = tao.ele_head(start)["s"] + startOffset
    endS = tao.ele_head(end)["s"] + endOffset

    #This disgusting workaround is required because PyTao implementation of taylor_map doesn't support -s mode
    if order == 0: 
        allStrings = tao.show(f"taylor_map -order 0 -s {startS} {endS}")
        transportMatrix = np.array([float(str) for str in allStrings[0].split()])
    
    elif order == 1:
        allStrings = tao.show(f"taylor_map -order 1 -s {startS} {endS}")
        splitStrings = [str.split() for str in allStrings]
        transportMatrix = np.array([ [float(str) for str in row[0:6]] for row in splitStrings[2:] ])

    else:
        print("Invalid matrix order requested")
        return
    

    if print:
        displayMatrix(transportMatrix)
        
    return transportMatrix

def getMatrixLEGACY(tao, start, end, order = 1, print = False):
    """Return zero or first order transport matrix from start to end. Optionally print in a human readable format"""

    if order == 0: 
        transportMatrix = (tao.matrix(start, end))["vec0"]
    elif order == 1:
        transportMatrix = (tao.matrix(start, end))["mat6"]
    else:
        print("Invalid matrix order requested")
        return
    
    
    if print:
        #display(pd.DataFrame(transportMatrix).style.hide(axis="index").hide(axis="columns"))
        displayMatrix(transportMatrix)
        
    return transportMatrix

def setLatticeAndGetMatrix(tao, start, end, startOffset = 0, endOffset = 0, defaultSettings = None, overrideSettings = {}):
    """
    Load lattice based on default and override settings. Then provide the transfer matrix between two elements, optionally with offsets
    If no defaultSettings are specified, the golden lattice from defaults.yml will be loaded
    """
    
    if not defaultSettings:
        defaultSettings = loadConfig(f"{tao.filePathGlobal}/setLattice_configs/defaults.yml")
        
    setLattice(tao, **( defaultSettings | overrideSettings ) )
   
    return getMatrix(tao, start, end, startOffset = startOffset, endOffset = endOffset)


# Historical version which only allows gaussian laser time profiles
# def addLHmodulation(
#     inputBeam, 
#     #Elaser, 
#     showplots=False,
#     laserHeater_laserEnergy = 0.5e-3,
#     laserHeater_sigma_t =  (2 / 2.355) * 1e-12,
#     laserHeater_offset = -0.5
# ):
#     """ From C. Emma, 2024-08-23 """
#     # Hardcode FACET-II laser and undulator parameters
#     # Laser parameters
#     Elaser = laserHeater_laserEnergy
#     lambda_laser = 760e-9
#     sigmar_laser = 200e-6
#     sigmat_laser = laserHeater_sigma_t # (2 / 2.355) * 1e-12
#     Plaser = Elaser / np.sqrt(2 * np.pi * sigmat_laser**2)
#     offset = laserHeater_offset #-0.5  # laser to e-beam offset if you want you can add it
#     # Undulator parameters
#     K = 1.1699
#     lambdaw = 0.054
#     Nwig = 9
#     # Electron beam
#     outputBeam = inputBeam.copy()
#     x = inputBeam.x - np.mean(inputBeam.x)
#     y = inputBeam.y - np.mean(inputBeam.y)
#     gamma = inputBeam.gamma
#     gamma0 = np.mean(inputBeam.gamma)
#     t = inputBeam.t-np.mean(inputBeam.t);
#     # Calculated parameters
#     lambda_r = lambdaw / 2 / gamma0**2 * (1 + K**2 / 2)  # Assumes planar undulator
#     omega = 2 * np.pi * 299792458 / lambda_r  # Resonant frequency
#     JJ = besselj(0, K**2 / (4 + 2 * K**2)) - besselj(1, K**2 / (4 + 2 * K**2))
#     totalLaserEnergy = np.sqrt(2 * np.pi * sigmat_laser**2) * Plaser
#     # Laser is assumed Gaussian with peak power Plaser
#     # This formula from eq. 8 Huang PRSTAB 074401 2004
#     mod_amplitude = np.sqrt(Plaser / 8.7e9) * K * lambdaw * Nwig / gamma0 / sigmar_laser * JJ
#     #print(mod_amplitude / np.sqrt(Plaser))
#     # offset = 1.0  # temporal offset between laser and e-beam in units of laser wavelengths
#     # Calculate induced modulation deltagamma
#     deltagamma = mod_amplitude * np.exp(-0.25 * (x**2 + y**2) / sigmar_laser**2) * \
#                  np.sin(omega * t + offset * 2 * np.pi) * \
#                  np.exp(-0.5 * ((t - offset * sigmat_laser) / sigmat_laser)**2)
#     outputBeam.gamma = inputBeam.gamma + deltagamma
#     return outputBeam, deltagamma, t


def addLHmodulation(
    inputBeam, 
    Elaser = None, 
    PowerProfile=None, 
    tvector=None, 
    case='Gaussian', 
    sigmat_laser=None, 
    offset=None, 
    showplots=False
):
    """Impose an energy modulation from a *laser heater* (LH) on an electron beam.

    The routine supports two use-cases that mirror how the FACET-II LH is
    typically modelled:

    * **Gaussian** (default) – The laser pulse is assumed to have a
      temporally-Gaussian power profile with total pulse energy ``Elaser``.
    * **arbitrary** – An arbitrary temporal power profile is supplied via
      ``PowerProfile``/``tvector``

    In both cases the laser is assumed transversely Gaussian with a
    hard-coded rms laser spot size and the undulator/LH hardware parameters
    installed at FACET-II.

    From C. Emma

    Parameters
    ----------
    inputBeam : ParticleGroup beam

    Elaser : float
        Laser pulse energy *[J]*.  Used only when ``case == 'Gaussian'``.
        
    PowerProfile : list, optional
        Instantaneous laser power *[W]* sampled at the points given by
        ``tvector``.  Required when ``case == 'arbitrary'``.
    
    tvector : list, optional
        Time grid *[s]* associated with ``PowerProfile``.  Required when
        ``case == 'arbitrary'``.
        
    case : {'Gaussian', 'arbitrary'}, default 'Gaussian'
        Choice of temporal profile model.
        
    sigmat_laser : float, optional
        Rms laser pulse length *[s]* for the Gaussian case.  If omitted,
        a 6 ps (FWHM) pulse is assumed.
        
    offset : float, optional
        Temporal offset *[s]* between the laser pulse centre and
        ``np.mean(inputBeam.t)`` in the Gaussian case.  Defaults to ``0``.
    
    showplots : bool
        Print plots

    Returns
    -------
    outputBeam : ParticleGroup beam
        Input beam with modulation applied
    deltagamma : ndarray
        Per-particle induced energy deviation ``Δγ``.
    t : ndarray
        The mean-subtracted time coordinate that was used in the modulation
        calculation (identical to ``inputBeam.t - np.mean(inputBeam.t)``).

    """

    
    # Hardcode FACET-II laser and undulator parameters
    lambda_laser = 760e-9
    sigmar_laser = 200e-6
    K = 1.1699
    lambdaw = 0.054
    Nwig = 9
    
    # Electron beam calculations
    x = inputBeam.x - np.mean(inputBeam.x)
    y = inputBeam.y - np.mean(inputBeam.y)
    t = inputBeam.t-np.mean(inputBeam.t);
    gamma = inputBeam.gamma
    gamma0 = np.nanmean(inputBeam.gamma)     

    if gamma0 == 0 or np.isnan(gamma0):
        print("Warning: gamma0 is either zero or NaN. Returning None.")
        return None    

    # Calculated parameters for arbitrary case
    lambda_r = lambdaw / (2 * gamma0**2) * (1 + K**2 / 2)
    omega = 2 * np.pi * 299792458 / lambda_r
    JJ = besselj(0, K**2 / (4 + 2 * K**2)) - besselj(1, K**2 / (4 + 2 * K**2))

    if case == 'arbitrary':
        if PowerProfile is None or tvector is None:
            raise ValueError("For 'arbitrary' case, both PowerProfile and tvector must be provided.")
        
        # Calculate the laser power profile normalized
        PlaserPeak = np.max(PowerProfile)
        normalizedPowerProfile = PowerProfile / PlaserPeak

        # Laser parameters    
        mod_amplitude = np.sqrt(PlaserPeak / 8.7e9) * K * lambdaw * Nwig / gamma0 / sigmar_laser * JJ

        if np.isnan(mod_amplitude):
            print("NaN value in LH mod_amplitude.")
            return None

        # Interpolate Power value at the location of the particle t
        interpolation_function = interp1d(tvector, normalizedPowerProfile, kind='linear', fill_value='extrapolate')
        PlaserInterp = interpolation_function(t)
        
        if np.any(np.isnan(PlaserInterp)):
            print("NaN values found in PlaserInterp.")
            return None

        if showplots:
            plt.scatter(t, PlaserInterp)
            plt.title("Interpolated Laser Power")
            plt.xlabel("Time")
            plt.ylabel("Normalized Laser Power")
            plt.show()

        # Calculate induced modulation for arbitrary case - Laser is assumed transversely Gaussian 
        deltagamma = mod_amplitude * np.sqrt(PlaserInterp) * \
                     np.exp(-0.25 * (x**2 + y**2) / sigmar_laser**2) * np.sin(omega * t)
        deltapz = deltagamma / gamma0

    elif case == 'Gaussian':
        # Set default values if not provided
        if sigmat_laser is None:
            sigmat_laser = (6 / 2.355) * 1e-12  # Default value for sigmat_laser
        if offset is None:
            offset = 0.0  # Default value for offset

        Plaser = Elaser / np.sqrt(2 * np.pi * sigmat_laser**2)
        
        # Laser is assumed transversely Gaussian with peak power Plaser
        mod_amplitude = np.sqrt(Plaser / 8.7e9) * K * lambdaw * Nwig / gamma0 / sigmar_laser * JJ
        
        # Calculate induced modulation for Gaussian case
        deltagamma = mod_amplitude * np.exp(-0.25 * (x**2 + y**2) / sigmar_laser**2) * \
                     np.sin(omega * t ) * \
                     np.exp(-0.25 * ((t - offset ) / sigmat_laser)**2)

    else:
        raise ValueError("Invalid case provided. Use 'arbitrary' or 'Gaussian'.")

    # Update the outputBeam with the new gamma values
    outputBeam = inputBeam.copy()
    outputBeam.gamma = inputBeam.gamma + deltagamma
    
    return outputBeam, deltagamma, t
    
def centerBeam(
    P,
    centerType = "median",
    assertEnergy = None
):
    """
    Shifts x, y, xp, and yp of a beam to zero
    centerType is either "median" or "mean"
    """
    
    PMod = P
    if centerType == "median":
        PMod.x = P.x - np.median(P.x)
        PMod.y = P.y - np.median(P.y)
        PMod.px = P.px - np.median(P.px)
        PMod.py = P.py - np.median(P.py)
        if assertEnergy:
            PMod.pz = P.pz * assertEnergy / np.median(P.pz)
        
        return PMod
        
    if centerType == "mean":
        PMod.x = P.x - np.mean(P.x)
        PMod.y = P.y - np.mean(P.y)
        PMod.px = P.px - np.mean(P.px)
        PMod.py = P.py - np.mean(P.py)
        if assertEnergy:
            PMod.pz = P.pz * assertEnergy / np.mean(P.pz)
                    
        return PMod

    return


def calcBMAG(b0, a0, b, a):
    #From Lucretia
    #For a bit more detail, see "BMAG from Lucretia.nb"
    #Not validated!!!

    # function [B,Bpsi]=bmag(b0,a0,b,a)
    # %
    # % [B,Bpsi]=bmag(b0,a0,b,a);
    # %
    # % Compute BMAG and its phase from Twiss parameters
    # %
    # % INPUTs:
    # %
    # %   b0 = matched beta
    # %   a0 = matched alpha
    # %   b  = mismatched beta
    # %   a  = mismatched alpha
    # %
    # % OUTPUTs:
    # %
    # %   B    = mismatch amplitude
    # %   Bpsi = mismatch phase (deg)

    g0 = (1 + a0 ** 2) / b0
    g  = (1 + a ** 2) / b
    B  = (b0 * g - 2.0 * a0 * a + g0 * b) / 2

    return B


def collimateBeam(
    P,
    allCollimatorRules = None
):
    """
    allCollimatorRules is a list of lists. Each sublist should have exactly two elements for the lower and upper x position of a collimator
    Arbitrarily many collimators can be defined this way; therefore it works for notch and/or jaw collimators
    """
    PMod = P.copy()


    for collimatorRange in allCollimatorRules:

        print(collimatorRange)
        all_indices = np.arange(len(PMod.x))
        killedIndices = np.where(np.logical_and(PMod.x > collimatorRange[0], PMod.x < collimatorRange[1]))[0]
        survivingIndices = np.setdiff1d(all_indices, killedIndices)
        
        # OpenPMD checks the length so I can't just remove the "killed" particles
        # Also, for compatibility, I don't want to change either the weight or status of the killed particles
        filtered_data = {
            "x": PMod.x[survivingIndices],
            "y": PMod.y[survivingIndices],
            "z": PMod.z[survivingIndices],
            "px": PMod.px[survivingIndices],
            "py": PMod.py[survivingIndices],
            "pz": PMod.pz[survivingIndices],
            "t": PMod.t[survivingIndices], 
            "status": PMod.status[survivingIndices], 
            "weight": PMod.weight[survivingIndices], 
            "species": PMod.species
        }
        
        # Create a new ParticleGroup instance with the filtered data
        PMod = ParticleGroup(data=filtered_data)
        print(f"New particle count: {len(PMod.x)}")
        print(f"{len(PMod.x)}")

    return PMod


def sortIndices(lst):
    #Returns the indices of the sorted elements, e.g. [1, 3, 5, 2, 4] --> [0, 3, 1, 4, 2]
    return [i for i, _ in sorted(enumerate(lst), key=lambda x: x[1])]

def sliceBeam(
    P,
    sortKey = None,
    numBeamlets = None
):
    """Sort a beam by sortKey, then slice it into numBeamlets of equal count"""
    sortedIndices = sortIndices(P[sortKey])
    
    subsetIndices = np.array_split(sortedIndices, numBeamlets)
    
    resultBeamlets = []
    
    for activeSubsetIndices in subsetIndices:
        PMod = P.copy()
        
        # OpenPMD checks the length so I can't just remove the "killed" particles
        # Also, for compatibility, I don't want to change either the weight or status of the killed particles
        filtered_data = {
            "x": PMod.x[activeSubsetIndices],
            "y": PMod.y[activeSubsetIndices],
            "z": PMod.z[activeSubsetIndices],
            "px": PMod.px[activeSubsetIndices],
            "py": PMod.py[activeSubsetIndices],
            "pz": PMod.pz[activeSubsetIndices],
            "t": PMod.t[activeSubsetIndices], 
            "status": PMod.status[activeSubsetIndices], 
            "weight": PMod.weight[activeSubsetIndices], 
            "species": PMod.species
        }
        
        # Create a new ParticleGroup instance with the filtered data
        PMod = ParticleGroup(data=filtered_data)
        #print(f"New particle count: {len(PMod.x)}")
        #print(f"{len(PMod.x)}")
    
        resultBeamlets.append(PMod)

    return resultBeamlets


def getSingleBeamSlice(
    P,
    sortKey = None,
    minVal = None,
    maxVal = None
):
    """Return a beamlet of particles which satisfy the inequality """

    # Get indices where sortKey is within the given range
    mask = (P[sortKey] >= minVal) & (P[sortKey] <= maxVal)
    
    if not np.any(mask):
        raise ValueError("No particles found in the specified range.")
    
    # Filter data based on the mask
    filtered_data = {
        "x": P.x[mask],
        "y": P.y[mask],
        "z": P.z[mask],
        "px": P.px[mask],
        "py": P.py[mask],
        "pz": P.pz[mask],
        "t": P.t[mask],
        "status": P.status[mask],
        "weight": P.weight[mask],
        "species": P.species
    }


    PMod = ParticleGroup(data=filtered_data)

    return PMod

def loadConfig(file, loaded_files=None):
    """Code to load nested config files... ChatGPT is the author, beware!"""
    if loaded_files is None:
        loaded_files = set()
    if file in loaded_files:
        return {}  # Avoid circular imports
    loaded_files.add(file)

    with open(file, 'r') as f:
        data = yaml.safe_load(f) or {}

    # Handle includes
    includes = data.pop('include', [])
    merged_data = {}
    for include_file in includes:
        merged_data.update(loadConfig(include_file, loaded_files))
    
    merged_data.update(data)  # Later settings override earlier ones
    return merged_data

from scipy.optimize import curve_fit

def gaussian_with_offset(x, A, mu, sigma, offset):
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2)) + offset

def fit_gaussian_sigma( data, bin_size = 1e-6 ):
    # Compute histogram
    data = np.array(data)
    bins = np.arange(min(data), max(data) + bin_size, bin_size)
    hist, edges = np.histogram(data, bins=bins)
    bin_centers = (edges[:-1] + edges[1:]) / 2

    # Initial guess: amplitude, mean, sigma, offset
    A_guess = max(hist)
    mu_guess = np.mean(data)
    sigma_guess = np.std(data)
    offset_guess = min(hist)
    p0 = [A_guess, mu_guess, sigma_guess, offset_guess]

    # Fit Gaussian
    try:
        popt, _ = curve_fit(gaussian_with_offset, bin_centers, hist, p0=p0)
        _, _, sigma_fit, _ = popt
        return sigma_fit
    except RuntimeError:
        return None  # Fit did not converge

def getBeamSpecs(
    P, 
    targetTwiss = None,
    calculateGaussianFits = False               
):
    """
    Returns a collection of convenient beam parameters as a dictionary
    Will automatically detect and add extra measurements for two-bunch beams
    
    targetTwiss can either be in the form [betaX, alphaX, betaY, alphaY] or
    for a very limited number of treaty point elements, can instead provide the element name. This will use the golden lattice targetTwiss
    Presently defined: "PR10571", "BEGBC20", "MFFF", "PENT"

    Optionally bin and fit gaussians to the x, y, and z position data. A bit expensive so left off by default
    """
    
    savedData = {}

    
    #A silly trick; set() will give back unique values
    bunchCount = len(set(P.weight))
    
    if bunchCount == 1:
        PDrive = P.copy()
        beamsToEvaluate = ["PDrive"]
    elif bunchCount == 2:
        PDrive, PWitness = getDriverAndWitness(P)
        beamsToEvaluate = ["PDrive", "PWitness"]
    else:
        print("bunchCount doesn't make sense. Aborting")
        return



    if targetTwiss:
        if isinstance(targetTwiss, str): 
            if targetTwiss == "PR10571":
                #PR10571 lucretia live model lattice 2024-10-16
                targetBetaX = 5.7
                targetBetaY = 2.6
                targetAlphaX = -2.1
                targetAlphaY = 0.0
                
            elif targetTwiss == "BEGBC20":
                #BEGBC20 lucretia live model lattice 2024-10-16
                targetBetaX = 11.5
                targetBetaY = 27.3
                targetAlphaX = 0.7
                targetAlphaY = 1.2
            
            elif targetTwiss == "MFFF":
                #MFFF lucretia live model lattice 2024-10-16
                targetBetaX = 11.6
                targetAlphaX = -0.64
                targetBetaY = 25.2
                targetAlphaY = -1.6
        
            elif targetTwiss == "PENT":
                #PENT lucretia live model lattice 2024-10-16
                targetBetaX = 0.5
                targetAlphaX = 0.0
                targetBetaY = 0.5
                targetAlphaY = 0.0

            else:
                print("Not a valid treaty point. Aborting")
                #return

                
                #Invalid treaty point; setting to None to avoid BMAG evaluation
                targetTwiss = None


        else:
            targetBetaX, targetAlphaX, targetBetaY, targetAlphaY = targetTwiss

    
    
    for PActiveStr in beamsToEvaluate:
        PActive = locals()[PActiveStr]

        
        # for val in ["mean_x", "mean_y", "sigma_x", "sigma_y", "mean_xp", "mean_yp"]:
        #     savedData[f"{PActiveStr}_{val}"] = PActive[val]

        
        savedData[f"{PActiveStr}_median_x"] = np.median(PActive.x)
        savedData[f"{PActiveStr}_median_y"] = np.median(PActive.y)

        savedData[f"{PActiveStr}_median_xp"] = np.median(PActive.xp)
        savedData[f"{PActiveStr}_median_yp"] = np.median(PActive.yp)
        savedData[f"{PActiveStr}_median_energy"] = np.median(PActive.energy)
        
        savedData[f"{PActiveStr}_sigmaSI90_x"] = smallestIntervalImpliedSigma(PActive.x, percentage = 0.90)
        savedData[f"{PActiveStr}_sigmaSI90_y"] = smallestIntervalImpliedSigma(PActive.y, percentage = 0.90)
        savedData[f"{PActiveStr}_sigmaSI90_z"] = smallestIntervalImpliedSigma(PActive.t * 3e8, percentage=0.9)

        if calculateGaussianFits:
            savedData[f"{PActiveStr}_fitSigma_x"] = fit_gaussian_sigma( PActive.x,       bin_size = 1e-6 )
            savedData[f"{PActiveStr}_fitSigma_y"] = fit_gaussian_sigma( PActive.y,       bin_size = 1e-6 )
            savedData[f"{PActiveStr}_fitSigma_z"] = fit_gaussian_sigma( PActive.t * 3e8, bin_size = 1e-6 )

        savedData[f"{PActiveStr}_sigmaSI90_xp"] = smallestIntervalImpliedSigma(PActive.xp, percentage = 0.90)
        savedData[f"{PActiveStr}_sigmaSI90_yp"] = smallestIntervalImpliedSigma(PActive.yp, percentage = 0.90)
        savedData[f"{PActiveStr}_sigmaSI90_energy"] = smallestIntervalImpliedSigma(PActive.energy, percentage = 0.90)

        savedData[f"{PActiveStr}_emitSI90_x"] = smallestIntervalImpliedEmittance(PActive, plane = "x", percentage = 0.90)
        savedData[f"{PActiveStr}_emitSI90_y"] = smallestIntervalImpliedEmittance(PActive, plane = "y", percentage = 0.90)

        savedData[f"{PActiveStr}_norm_emit_x"] = (PActive.twiss(plane = "x", fraction = 0.9))["norm_emit_x"]
        savedData[f"{PActiveStr}_norm_emit_y"] = (PActive.twiss(plane = "y", fraction = 0.9))["norm_emit_y"]

        if bunchCount == 2:
            savedData[f"{PActiveStr}_zCentroid"] = np.median(PActive.t * 3e8)

        savedData[f"{PActiveStr}_charge_nC"] = PActive.charge * 1e9


        PActiveTwiss = PActive.twiss(plane = "x", fraction = 0.9) | PActive.twiss(plane = "y", fraction = 0.9)

        if targetTwiss: 
            savedData[f"{PActiveStr}_BMAG_x"] = calcBMAG(targetBetaX, targetAlphaX, PActiveTwiss["beta_x"], PActiveTwiss["alpha_x"])
            savedData[f"{PActiveStr}_BMAG_y"] = calcBMAG(targetBetaY, targetAlphaY, PActiveTwiss["beta_y"], PActiveTwiss["alpha_y"])
    
            # Get BMAGs by energy slice
            slicedBeamlets = sliceBeam( PActive , sortKey = "pz", numBeamlets = 5 )
    
            slicedTwiss =  [ ( beamlet.twiss(plane = "x", fraction = 0.9) | beamlet.twiss(plane = "y", fraction = 0.9) ) for beamlet in slicedBeamlets ] 
            
            savedData[f"{PActiveStr}_sliced_BMAG_x"] = [ calcBMAG(targetBetaX, targetAlphaX, beamletTwiss["beta_x"], beamletTwiss["alpha_x"]) for beamletTwiss in slicedTwiss ]
            savedData[f"{PActiveStr}_sliced_BMAG_y"] = [ calcBMAG(targetBetaY, targetAlphaY, beamletTwiss["beta_y"], beamletTwiss["alpha_y"]) for beamletTwiss in slicedTwiss ]

    if bunchCount == 2:
        savedData["bunchSpacing"] = savedData["PWitness_zCentroid"] - savedData["PDrive_zCentroid"]

        savedData["transverseCentroidOffset"] = np.sqrt(
                (savedData["PDrive_median_x"] - savedData["PWitness_median_x"])**2 + 
                (savedData["PDrive_median_y"] - savedData["PWitness_median_y"])**2
            )

    #savedData["lostChargeFraction"] = 1 - (P.charge / PInit.charge)

    return savedData



#Here's a version that would work if the axes are coupled.... they really, really, really shouldn't ever be though
def launchTwissCorrectionObjective(params, tao, evalElement, targetBetaX, targetAlphaX, targetBetaY, targetAlphaY):
    betaSetX, alphaSetX, betaSetY, alphaSetY = params
    
    try:
        #Prevent recalculation until changes are made
        tao.cmd("set global lattice_calc_on = F")
        
        tao.cmd(f"set element beginning beta_a = {betaSetX}")
        tao.cmd(f"set element beginning alpha_a = {alphaSetX}")
        tao.cmd(f"set element beginning beta_b = {betaSetY}")
        tao.cmd(f"set element beginning alpha_b = {alphaSetY}")
        
        #Reenable lattice calculations
        tao.cmd("set global lattice_calc_on = T")
    
    except: #If Bmad doesn't like the proposed solution, don't crash, give a bad number
        return 1e20
    
    return (tao.ele_twiss(evalElement)[f"beta_a"] - targetBetaX) ** 2 + (tao.ele_twiss(evalElement)[f"alpha_a"] - targetAlphaX) ** 2 + (tao.ele_twiss(evalElement)[f"beta_b"] - targetBetaY) ** 2 + (tao.ele_twiss(evalElement)[f"alpha_b"] - targetAlphaY) ** 2

def launchTwissCorrection(tao, 
                          evalElement = None, 
                          targetBetaX = None, 
                          targetAlphaX = None, 
                          targetBetaY = None, 
                          targetAlphaY = None
                         ):
    """
    This function will update the BEGINNING twiss values (set in bmad/models/f2_elec/f2_elec.lat.bmad, e.g. BEGINNING[BETA_A] =  1.39449126865854395E-001) to achieve an arbitrary match at an arbitrary element.

    If no element is specified, the function will create the default golden lattice match at PR10571
    """
    
    from scipy.optimize import minimize

    if not evalElement:
        print("No evalElement provided. Assuming golden lattice PR10571")
        evalElement = "PR10571"
        targetBetaX = 5.73666431
        targetAlphaX = -2.14411559
        targetBetaY = 2.57530302
        targetAlphaY = 0.01016211

    # Perform optimization using Nelder-Mead
    result = minimize(
        launchTwissCorrectionObjective, 
        [0.137, 0.954, 0.406, 2.20], #Starting point
        #[101, -26, 101, -26],
        method='Nelder-Mead',
        bounds = [(1e-9, 1e3), (-100, 100), (1e-9, 1e3), (-100, 100)],
        args = (tao, evalElement, targetBetaX, targetAlphaX, targetBetaY, targetAlphaY)
    )


    #Apply best result to the lattice
    betaSetX, alphaSetX, betaSetY, alphaSetY = result.x
    
    #Prevent recalculation until changes are made
    tao.cmd("set global lattice_calc_on = F")
    
    tao.cmd(f"set element beginning beta_a = {betaSetX}")
    tao.cmd(f"set element beginning alpha_a = {alphaSetX}")
    tao.cmd(f"set element beginning beta_b = {betaSetY}")
    tao.cmd(f"set element beginning alpha_b = {alphaSetY}")
    
    #Reenable lattice calculations
    tao.cmd("set global lattice_calc_on = T")
                          
    print("Optimization Results:")
    print(f"Optimal Parameters: {result.x}")
    print(f"Objective Function Value at Optimal Parameters: {result.fun}")
    print(f"Number of Iterations: {result.nit}")
    print(f"Converged: {result.success}")

    return


def generalizedEmittanceSolverObjective(params, data):
    betaI, alphaI, emittanceGeo = params
    
    errorComponents = []

    for shot in data:
        # Twiss transfer matrix for beta
        # R11^2 \[Beta] - 2 R11 R12 \[Alpha] +  + R12^2 \[Gamma]
        term1 = betaI * shot["R11"] ** 2
        term2 = -2 * alphaI * shot["R11"] * shot["R12"]
        term3 =  ( (1 + alphaI ** 2) / betaI ) * shot["R12"] ** 2

        # (beta * emit_geo) == sigma^2
        errorComponent = ( term1 + term2 + term3 ) * emittanceGeo - shot["sigma"] ** 2

        #Add all error terms in quadrature
        errorComponents.append(errorComponent ** 2)
    
        
    
    return np.sum(errorComponents)

def generalizedEmittanceSolver(
    data,
    energyGeV = None,
    verbose = False,
    initialGuess = [0.5, 0.5, 1e-9],
    **kwargs
):
    """
    `data` should be a list of dictionaries, each of which should contain at least "R11", "R12", and "sigma" corresponding to the R-matrix terms for the transfer of interest
    and the beam sigma at the downstream screen.

    The the initial beta, alpha, and emittance are used as fit parameters to explain the observations
    The typical Twiss transfer is applied for each case and compared to the observed spot size

    If this isn't giving the values you expect, make sure the single-particle Twiss values are what you expect! (Consider running launchTwissCorrection())
    """
    
    from scipy.optimize import minimize


    # Perform optimization using Nelder-Mead
    result = minimize(
        generalizedEmittanceSolverObjective, 
        initialGuess, #Starting point
        method='Nelder-Mead',
        args = (data, ),
        **kwargs
    )


    if verbose:
        print("Optimization Results:")
        print(f"Optimal Parameters: {result.x}")
        print(f"Objective Function Value at Optimal Parameters: {result.fun}")
        print(f"Number of Iterations: {result.nit}")
        print(f"Converged: {result.success}")

    output = {"beta" : result.x[0], "alpha" : result.x[1], "emitGeo" : result.x[2]}

    if energyGeV:
        #Sloppy, ultrarel only
        output["emit"] = result.x[2] * energyGeV * 1000 / 0.511
    
    return output

def disableAutoQuadEnergyCompensation(tao):
    """
    The golden lattice, by default, has the quads set according to the design K1 rather than a fixed gradient.
    For "typical" simulations, this is a good approach. It's basically assuming that we LEM the machine
    For some edge cases, like jitter simulations, we don't want the magnets to be changing though
    """ 
    
    tao.cmd("set ele QUAD::* FIELD_MASTER = T")

    return

def disableAutoMagnetEnergyCompensation(tao):
    """
    The golden lattice, by default, has the magnets set according to the design K# rather than a fixed field.
    For "typical" simulations, this is a good approach. It's basically assuming that we LEM the machine
    For some edge cases, like jitter simulations, we don't want the magnets to be changing though
    """ 
    
    tao.cmd("set ele Solenoid::* FIELD_MASTER = T")
    
    tao.cmd("set ele Sbend::* FIELD_MASTER = T")
    tao.cmd("set ele Quadrupole::* FIELD_MASTER = T")
    tao.cmd("set ele Sextupole::* FIELD_MASTER = T")
    tao.cmd("set ele Multipole::* FIELD_MASTER = T")

    tao.cmd("set ele Wiggler::* FIELD_MASTER = T")

    tao.cmd("set ele VKicker::* FIELD_MASTER = T")
    tao.cmd("set ele HKicker::* FIELD_MASTER = T")

    return

def applyOtherConfig(tao, configArr):
    """
    The format for other_configs is an array with rows
    [ elementName, attributeName, setValue ] 
    """
    
    #Prevent recalculation until changes are made
    tao.cmd("set global lattice_calc_on = F")

    try: 
        for row in configArr:
            tao.cmd(f"""set ele {row[0]} {row[1]} = {row[2]}""")

    except:
        print("WARNING! At least one assignment has failed!")

    #Prevent recalculation until changes are made
    tao.cmd("set global lattice_calc_on = T")