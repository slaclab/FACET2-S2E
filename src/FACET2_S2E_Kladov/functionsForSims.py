import math
from scipy.stats import moment
from scipy.stats import gennorm
from scipy.special import gamma
from scipy.optimize import curve_fit
from scipy.ndimage import gaussian_filter1d
import pprint
from copy import copy
import matplotlib.pyplot as plt

#import mplstyle
from matplotlib.ticker import AutoMinorLocator
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from Experimental_functions import *

# from FACET2_S2E_Kladov.UTILITY_quickstart import (
#     initializeTao,
#     trackBeam,
#     getBeamAtElement,
# )

from FACET2_S2E_Kladov.UTILITY_quickstart import *

from FACET2_S2E.UTILITY_linacPhaseAndAmplitude import matchStringWrapper



"""Utilities for BMAD-based FACET2-S2E simulation workflows.

This module contains beam preparation, lattice editing, energy tuning,
scan drivers, and plotting helpers used for simulation workflows
connected to the experimental data (DAQ scans).
"""

## Bunch support functions

### Create a bunch

def make_simple_bunch(file, n = 0, save_path = ''):
    """Create a Gaussian bunch with statistics matched to an input beam.

    The transverse and momentum distributions are drawn from normal
    distributions matched to the input beam rms values.
    The transverse emittance is increased (alpha = 0, rms are the same);
    The longitudinal emittance is kept approximately the same (alpha = 0, rms by pz is taken from a 0.1um slice).

    Parameters:
        file: Base path of input beam file without the .h5 extension.
        n: Number of macro particles to generate. If 0, the input beam size is used.
        save_path: Output file path without extension. If "", file+'_simple.h5' is used.

    Returns:
        ParticleGroup: Generated bunch.
    """
    beam = ParticleGroup(file + '.h5')
    
    N = np.size(beam.x)
    N = N if n==0 else n
    charge = beam.charge
    data = {'x': np.zeros(N), 'px': np.zeros(N), 'y': np.zeros(N), 'py': np.zeros(N), 'z': np.zeros(N), 'pz': np.zeros(N), 't': np.zeros(N), 'status': np.ones(N).astype(int), 'weight': np.ones(N)*charge/N, 'species': 'electron', 'id': np.arange(N).astype(int)}
    P1 = ParticleGroup(data = data)
    
    Pslice = cut_length(beam, length = 1e-7)
    
    P1.x = np.random.normal(0, 1*moment(beam.x, moment=2) ** 0.5, N)
    P1.y = np.random.normal(0, 1*moment(beam.y, moment=2) ** 0.5, N)
    P1.px = np.random.normal(0, 1*moment(beam.px, moment=2) ** 0.5, N)
    P1.py = np.random.normal(0, 1*moment(beam.py, moment=2) ** 0.5, N)
    P1.pz = np.random.normal(np.mean(beam.pz), 1*moment(Pslice.pz, moment=2) ** 0.5, N)
    P1.t = np.random.normal(0, 1*moment(beam.t, moment=2) ** 0.5, N)
    
    match_impact_file = file + '_simple' + '.h5' if save_path=='' else save_path + '.h5'
    P1.write(match_impact_file)
    return P1


def make_simple_bunch_flatter(file, n = 0, save_path = ''):
    """Create a flatter bunch using a generalized normal time distribution.

    Similar to make_simple_bunch, but the longitudinal time coordinate is
    sampled from a generalized normal distribution to produce a flatter
    longitudinal profile.

    Parameters:
        file: Base path of input beam file without the .h5 extension.
        n: Number of macro particles to generate. If 0, the input beam size is used.
        save_path: Output file path without extension. If empty, file+'_simple.h5' is used.
    """
    beam = ParticleGroup(file + '.h5')
    
    N = np.size(beam.x)
    N = N if n==0 else n
    charge = beam.charge
    data = {'x': np.zeros(N), 'px': np.zeros(N), 'y': np.zeros(N), 'py': np.zeros(N), 'z': np.zeros(N), 'pz': np.zeros(N), 't': np.zeros(N), 'status': np.ones(N).astype(int), 'weight': np.ones(N)*charge/N, 'species': 'electron', 'id': np.arange(N).astype(int)}
    P1 = ParticleGroup(data = data)
    
    Pslice = cut_length(beam, length = 1e-7)
    
    P1.x = np.random.normal(0, 1*moment(beam.x, moment=2) ** 0.5, N)
    P1.y = np.random.normal(0, 1*moment(beam.y, moment=2) ** 0.5, N)
    P1.px = np.random.normal(0, 1*moment(beam.px, moment=2) ** 0.5, N)
    P1.py = np.random.normal(0, 1*moment(beam.py, moment=2) ** 0.5, N)
    P1.pz = np.random.normal(np.mean(beam.pz), 1*moment(Pslice.pz, moment=2) ** 0.5, N)
    P1.t = gennorm.rvs(4, size=N)*((moment(beam.t, moment=2)/(gamma(3/4)/gamma(1/4))) ** 0.5)
    
    match_impact_file = file + '_simple' + '.h5' if save_path=='' else save_path + '.h5'
    P1.write(match_impact_file)
    return P1


def make_simple_bunch_standalone(N = 0, meanPzMeV = 125 , moments=[0.3e-3, 0.2e-3, 0.4e-3, 0.2e-3, 0.58e-3, 0], charge = 1e-9, save_path = '', means=[0,0,0,0,0,0]):
    """Create a Gaussian bunch from explicit statistical moments.

    Parameters:
        N: Number of macro particles.
        meanPzMeV: Mean longitudinal momentum in MeV/c.
        moments: RMS values in x, xp, y, yp, z, pz.
        charge: bunch charge.
        save_path: Output file path without extension.
        means: Mean values for x, xp, y, yp, z, pz.

    Returns:
        ParticleGroup: Generated bunch.
    """

    N = int(N)
    data = {'x': np.zeros(N), 'px': np.zeros(N), 'y': np.zeros(N), 'py': np.zeros(N), 'z': np.zeros(N), 'pz': np.zeros(N), 't': np.zeros(N), 'status': np.ones(N).astype(int), 'weight': np.ones(N)*charge/N, 'species': 'electron', 'id': np.arange(N).astype(int)}
    P1 = ParticleGroup(data = data)
    
    P1.x = np.random.normal(means[0], moments[0], N)
    P1.px = np.random.normal(means[1], moments[1]*meanPzMeV*1e6, N)
    P1.y = np.random.normal(means[2], moments[2], N)
    P1.py = np.random.normal(means[3], moments[3]*meanPzMeV*1e6, N)
    P1.t = np.random.normal(means[4], moments[4], N)/3e8
    P1.pz = np.random.normal(meanPzMeV*1e6, moments[5], N)
    
    match_impact_file = save_path + '.h5'
    P1.write(match_impact_file)
    return P1

def make_simple_bunch_theory_from_bunch_sims(bunch_file, mean_lattice_P0C_MeV, means_shift=[0,0,0,0,0,0]):
    """Convert a BMAD bunch into theory coordinates for map-based calculations.

    Parameters:
        bunch_file: Base path of the bunch file without extension.
        mean_lattice_P0C_MeV: Reference lattice momentum in MeV/c (the bunch will have this <pz>).
        means_shift: Shifts to apply to x, xp, y, yp, z, delta.

    Returns:
        np.ndarray: Nx6 array in the order [x, xp, y, yp, z, delta].
    """
    sim_bunch = ParticleGroup(bunch_file+".h5")
    return np.stack((sim_bunch.x+means_shift[0], sim_bunch.xp+means_shift[1], sim_bunch.y+means_shift[2], sim_bunch.yp+means_shift[3], -3e8*sim_bunch.t, (sim_bunch.pz*1e-6-mean_lattice_P0C_MeV)/mean_lattice_P0C_MeV), axis=1)


## Modify the bunch as a whole (sizes, means, chirps, correlations)

def modifyInputBeamSimple(inputBeamFilePath, numMacroParticles = None, timeCenterTF = True):
    """Prepare an input beam for Tao by optionally downsampling and centering. Almost the same as Nathans', but without Twiss matching.

    The beam is drift_to_z(), set z=0, and optionally time-centered.
    If numMacroParticles is provided, the beam is randomly subsampled and weights are adjusted.

    Parameters:
        inputBeamFilePath: Path to the input beam file, including extension.
        numMacroParticles: Target number of macroparticles.
        timeCenterTF: If True, subtract the mean time to center the bunch and avoid cavity phase mismatch.

    Returns:
        ParticleGroup: Modified beam ready for use with Tao.
    """
    P = ParticleGroup(inputBeamFilePath)

    if numMacroParticles:
        if numMacroParticles>0:
            initialImportSize = np.size(P.x)
            numMacroParticles = int(numMacroParticles)
            P = P[random.sample(range(initialImportSize), numMacroParticles)]
            P.weight = P.weight * (initialImportSize / numMacroParticles)

    P.drift_to_z()
    P.z = np.zeros(np.size(P.x))
    # Time center
    if timeCenterTF:
        P.t=P.t-np.mean(P.t) # This is OK because present beam doesn't have different weights; np.unique(P.weight)
        
    return P

def sqrtm_psd(M, tol=1e-14):
    """Compute the positive-semidefinite square root of a symmetric matrix.
    
    Returns:
        tuple: (matrix_sqrt, eigenvalues)
    """
    w, V = np.linalg.eigh(M)
    w_clipped = np.clip(w, 0.0, None)  # allow zero
    return V @ np.diag(np.sqrt(w_clipped)) @ V.T, w

def invsqrtm_psd(M, tol=1e-14):
    """Compute the inverse square root of a symmetric matrix, with small eigenvalues treated as zero.
    
    Returns:
        tuple: (matrix_sqrt, eigenvalues)
    """
    w, V = np.linalg.eigh(M)
    w_inv = np.zeros_like(w)
    mask = w > tol
    w_inv[mask] = 1.0 / np.sqrt(w[mask])
    # zero eigenvalues remain zero → projection
    return V @ np.diag(w_inv) @ V.T, w


def edit_bunch_parameters_from_PG(P_arg, pzMeV=None, moments=[None, None, None, None, None, None], correlations=[None,None,None], means=[0,0,0,0,0],
                                  charge=-1, betaX=None, alphaX=None, emittanceX=None, betaY=None, alphaY=None, emittanceY=None, path_to_write='temp_beam/temp_e'):
    '''
    moments are for x, xp, y, yp, z, pz
    means are for x, xp, y, yp, z
    xp and yp are in radians
    moments are RMS sizes

    if betaX and alphaX are supplied, the X phase space will have the emittance corresponding to moments[0] (size x = sqrt{epsilon beta}), and moments[1] will be overwritten. Same for Y.
    '''
    P = P_arg.copy()
    
    if correlations[0] is None:
        correlations[0] = np.mean((P.x - np.mean(P.x))*(P.xp - np.mean(P.xp)))/np.std(P.x - np.mean(P.x))*np.std(P.xp - np.mean(P.xp)) if (np.std(P.xp - np.mean(P.xp))!=0 and np.std(P.x - np.mean(P.x))!=0) else 0
    if correlations[1] is None:
        correlations[1] = np.mean((P.y - np.mean(P.y))*(P.yp - np.mean(P.yp)))/np.std(P.y - np.mean(P.y))*np.std(P.yp - np.mean(P.yp)) if (np.std(P.yp - np.mean(P.yp))!=0 and np.std(P.y - np.mean(P.y))!=0) else 0
    if correlations[2] is None:
        correlations[2] = np.mean((P.pz - np.mean(P.pz))*(P.t - np.mean(P.t))*3e8)/np.std(P.pz - np.mean(P.pz))*np.std((P.t - np.mean(P.t))*3e8) if (np.std((P.t - np.mean(P.t))*3e8)!=0 and np.std(P.pz - np.mean(P.pz))!=0) else 0
    
    sigmaMatrixX=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixY=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixZ=np.array([[-1,-1],[-1,-1]], dtype=float)
    sigmaMatrixX[0][0] = -1 if moments[0]==None else moments[0]**2
    sigmaMatrixX[1][1] = -1 if moments[1]==None else moments[1]**2
    sigmaMatrixY[0][0] = -1 if moments[2]==None else moments[2]**2
    sigmaMatrixY[1][1] = -1 if moments[3]==None else moments[3]**2
    sigmaMatrixZ[0][0] = -1 if moments[4]==None else moments[4]**2
    sigmaMatrixZ[1][1] = -1 if moments[5]==None else moments[5]**2
    
    # charge
    if charge!=-1:
        P.charge = charge
    N = len(P.x)
    
    # longitudinal momentum (pz), t and z
    if pzMeV is not None and pzMeV>0:
        P.pz = P.pz*1e6*pzMeV/np.mean(P.pz)

    meanpz = np.mean(P.pz)
    P.pz = P.pz - meanpz
    means[4] = means[4] if means[4]!=None else np.mean(P.t)*3e8
    P.t = P.t - np.mean(P.t)

    if not ((sigmaMatrixZ[1,1]==-1) and (sigmaMatrixZ[0,0]==-1)):
        if sigmaMatrixZ[0][0]==-1:
            sigmaMatrixZ[0][0] = np.std(P.t*3e8)**2
        if sigmaMatrixZ[1][1]==-1:
            sigmaMatrixZ[1][1] = np.std(P.pz)**2
        sigmaMatrixZ[0][1] = correlations[2]*np.sqrt(sigmaMatrixZ[0][0]*sigmaMatrixZ[1][1])
        sigmaMatrixZ[1][0] = sigmaMatrixZ[0][1]
        if np.std(P.t)==0 or np.std(P.pz)==0:
            P.t = np.random.normal(0, 1, N)/3e8
            P.pz = np.random.normal(0, 1, N)
        Z = np.vstack((-P.t*3e8, P.pz))
        Sigma_current = np.cov(Z, bias=True)
        S_target_sqrt, _ = sqrtm_psd(sigmaMatrixZ)
        S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
        T = S_target_sqrt @ S_current_invsqrt
        Z_new = T @ Z
        P.t = -Z_new[0]/3e8
        P.pz = Z_new[1]

    P.pz = P.pz + meanpz
    P.t = P.t + means[4]/3e8
            
    # means before
    means[0] = means[0] if means[0]!=None else np.mean(P.x)
    means[1] = means[1] if means[1]!=None else np.mean(P.xp)
    means[2] = means[2] if means[2]!=None else np.mean(P.y)
    means[3] = means[3] if means[3]!=None else np.mean(P.yp)
    P.x = P.x - np.mean(P.x)
    P.px = P.px - np.mean(P.px)
    P.y = P.y - np.mean(P.y)
    P.py = P.py - np.mean(P.py)

    #Apply linear matching X
    if (betaX is not None) and (alphaX is not None):
        current_sqrt_emmitance = np.sqrt(np.sqrt( np.mean(P.x**2)*np.mean(P.xp**2) - np.mean(P.x*P.xp)**2 ))
        if current_sqrt_emmitance==0:
            print("edit_bunch_parameters_from_PG: Zero emittance in X. Filling with Gaussian with the target emittance.")
            P.x = np.random.normal(0, np.sqrt(emittanceX), N)
            P.px = np.random.normal(0, np.mean(P.pz)*np.sqrt(emittanceX), N)
        P.twiss_match(plane='x', beta = betaX, alpha = alphaX, inplace=True)
        if current_sqrt_emmitance!=0:
            if emittanceX is not None:
                P.x = P.x*np.sqrt(emittanceX)/current_sqrt_emmitance
                P.px = P.px*np.sqrt(emittanceX)/current_sqrt_emmitance
            
    #if doing second moments matrix instead of beta alpha emittance
    else:
        if not ((sigmaMatrixX[1,1]==-1) and (sigmaMatrixX[0,0]==-1)):
            if sigmaMatrixX[0][0]==-1:
                sigmaMatrixX[0][0] = np.std(P.x)**2
            if sigmaMatrixX[1][1]==-1:
                sigmaMatrixX[1][1] = np.std(P.xp)**2
            sigmaMatrixX[0][1] = correlations[0]*np.sqrt(sigmaMatrixX[0][0]*sigmaMatrixX[1][1])
            sigmaMatrixX[1][0] = sigmaMatrixX[0][1]
            if np.std(P.x)==0 or np.std(P.xp)==0:
                P.x = np.random.normal(0, 1, N)
                P.px = np.random.normal(0, np.mean(P.pz)*1, N)
            X = np.vstack((P.x, P.xp))
            Sigma_current = np.cov(X, bias=True)
            S_target_sqrt, _ = sqrtm_psd(sigmaMatrixX)
            S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
            T = S_target_sqrt @ S_current_invsqrt
            X_new = T @ X
            P.x = X_new[0]
            P.px = X_new[1]*np.mean(P.pz)

    if (betaY is not None) and (alphaY is not None):
        current_sqrt_emmitance = np.sqrt(np.sqrt( np.mean(P.y**2)*np.mean(P.yp**2) - np.mean(P.y*P.yp)**2 ))
        if current_sqrt_emmitance==0:
            print("edit_bunch_parameters_from_PG: Zero emittance in Y. Filling with Gaussian with the target emittance.")
            P.y = np.random.normal(0, np.sqrt(emittanceY), N)
            P.py = np.random.normal(0, np.mean(P.pz)*np.sqrt(emittanceY), N)
        P.twiss_match(plane='y', beta = betaY, alpha = alphaY, inplace=True)
        if current_sqrt_emmitance!=0:
            if emittanceY is not None:
                P.y = P.y*np.sqrt(emittanceY)/current_sqrt_emmitance
                P.py = P.py*np.sqrt(emittanceY)/current_sqrt_emmitance
    else:
        if not ((sigmaMatrixY[1,1]==-1) and (sigmaMatrixY[0,0]==-1)):
            if sigmaMatrixY[0][0]==-1:
                sigmaMatrixY[0][0] = np.std(P.y)**2
            if sigmaMatrixY[1][1]==-1:
                sigmaMatrixY[1][1] = np.std(P.yp)**2
            sigmaMatrixY[0][1] = correlations[1]*np.sqrt(sigmaMatrixY[0][0]*sigmaMatrixY[1][1])
            sigmaMatrixY[1][0] = sigmaMatrixY[0][1]
            if np.std(P.y)==0 or np.std(P.yp)==0:
                P.y = np.random.normal(0, 1, N)
                P.py = np.random.normal(0, np.mean(P.pz)*1, N)
            Y = np.vstack((P.y, P.yp))
            Sigma_current = np.cov(Y, bias=True)
            S_target_sqrt, _ = sqrtm_psd(sigmaMatrixY)
            S_current_invsqrt, _ = invsqrtm_psd(Sigma_current)
            T = S_target_sqrt @ S_current_invsqrt
            Y_new = T @ Y
            P.y = Y_new[0]
            P.py = Y_new[1]*np.mean(P.pz)

    # means after
    P.x = P.x + means[0]
    P.px = P.px + means[1]*np.mean(P.pz)
    P.y = P.y + means[2]
    P.py = P.py + means[3]*np.mean(P.pz)

    P.write(path_to_write+".h5")
    return P

def edit_bunch_parameters(file_ext, pzMeV=None, moments=[None,None,None,None,None,None], correlations=[None,None,None], means=[0,0,0,0,0], charge=-1,
                          betaX=None, alphaX=None, emittanceX=None, betaY=None, alphaY=None, emittanceY=None, path_to_write='temp_beam/temp_e'):
    '''Edit the bunch parameters.
    file_ext: Base file path without the .h5 extension.

    moments are for x, xp, y, yp, z, pz
    means are for x, xp, y, yp, z
    xp and yp are in radians
    moments are RMS sizes

    if betaX and alphaX are supplied, the X phase space will have the emittance corresponding to moments[0] (size x = sqrt{epsilon beta}), and moments[1] will be overwritten. Same for Y.
    '''
    return edit_bunch_parameters_from_PG(ParticleGroup(file_ext + ".h5"), pzMeV=pzMeV, moments=moments, correlations=correlations, means=means, charge=charge,
                                  betaX=betaX, alphaX=alphaX, emittanceX=emittanceX, betaY=betaY, alphaY=alphaY, emittanceY=emittanceY, path_to_write=path_to_write)

def cut_length(particle_group, length = 0, drift_to_z = True):
    """Return a slice of the beam around its mean arrival time.

    Parameters:
        particle_group: Input ParticleGroup.
        length: Full longitudinal window in meters.
        drift_to_z: If False, the beam will be drift_to_t to <t> after the slicing.

    Returns:
        ParticleGroup: Sliced beam.
    """

    P = particle_group.copy()
    P.drift_to_z()
    indexes_to_leave = []
    meanT = np.mean(P.t)
    for (i,p) in enumerate(P):
        if(np.abs(p.t-meanT)<(length/(3e8))):
            indexes_to_leave.append(i)
    indices = np.array(indexes_to_leave)
    Ptemp = P[indices]
    if not drift_to_z:
        Ptemp.drift_to_t()
    return Ptemp


## Initialize and run a simulation


### High end

def get_tao_from_experiment(experiment="", scan_number="", date="", start='L0AFEND', finish='PR11375', filepath="/sdf/group/facet/kladov/FACET2_S2E", locationsToSave = [],
                            csrTF=False, lscTF=False, file_ext = "", energy=None, N_in_simple_bunch=5e4, N_to_use_from_file=None, tune_dipoles_to_125_335_4500_10000_MeV=True,
                            correctors_coef=0, correctors_from_beg=False, run=True, gaussFromExternal=False, edit_only_energy_from_exp=False, energy_edit_on_beam=False, verbose=False,
                            lattice='setLattice_configs/2024-10-22_oneBunch-Copy1.yml', moments=[None,None,None,None,None,None], means=[0,0,0,0,None], charge=1.6e-9, sr_wakes_on=False, lr_wakes_on=False,
                            desired_beam_energies_for_the_feedback=None, desired_P0Cs_MeV=[None,None,None,None], grid_size=[32,32,32], lsc_method="slice", csr_method="1_dim", n_bin=32,
                            edited_bunch_energy_at_checkpoints_MeV=[None, None, None, None]):
    '''
    experiment: "BEAMPHYS" is an example.
    scan_number: DAQ scan number. "14438" is an example.
    date: the date when the scan was taken in a specific form. "/2026/20260121" is an example.

    start: where the simulation starts. This 1) can affect the initial bunch energy (see "energy"), 2) determines the start for the energy_edit_on_beam and run_initialized_sim functions.
    finish: determines the finish for the energy_edit_on_beam and run_initialized_sim functions.

    filepath: Path to the package. I haven't found a way to do this automatically yet.

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

    tune_dipoles_to_125_335_4500_10000_MeV: if True, the dipole magnetic fields are set to constants, corresponding to the angle and rho from the .tao lattice at 125, 335, 4500 and 10000 MeV.
    If False, the simulation is the same as Nathan's, where the dipole strength changes with the lattice energy.
    Note that with tune_dipoles_to_125_335_4500_10000_MeV=False, setting the dipoles to the values from DAQ will not give the desired effect.
    This setting was disabled because of the unsolved lattice autoupdate issue (that is present in the default sim too).
    The problem workaround is using desired_P0Cs_MeV for now.
    '''
    tao = initializeTao(filePath = filepath, loadCustomLatticeTF=True, csrTF=csrTF, lscTF=lscTF, latticeFile=lattice, bmad_grid_size=grid_size, verbose=verbose, sr_wakes_on=sr_wakes_on, lr_wakes_on=lr_wakes_on, lsc_method=lsc_method, csr_method=csr_method, n_bin=n_bin)

    # copy the experiment data
    if experiment!="" and scan_number!="":
        ds = DATASET("", experiment, scan_number, pathfull = "".join(["/sdf/data/ad/fs/transition/nfs/slac/g/facet/matlab/data_prod/nas-li20-pm00/", experiment, date]))
        # check if magnets data is not needed. Dogleg energy is always 125 MeV (maybe need to change to the mean of 'BEND_IN10_661_BDES' and 'BEND_IN10_751_BDES' (they are in GeV), so that the magnet strengths are actually correct)
        if edit_only_energy_from_exp:
            tao = edit_energy_tao_based_on_experiment_database(tao, ds)
        else:
            tao = edit_tao_based_on_experiment_database(tao, ds, correctors_coef=correctors_coef, correctors_from_beg=correctors_from_beg)

    # if tune_dipoles_to_125_335_4500_10000_MeV:
    #     tao = treat_dipoles(tao)

    # tao.cmd(f'set global lattice_calc_on = T')
    # deal with the bunch
    current_e_start = tao.ele_gen_attribs(start)["P0C"]*1e-6 if energy is None else energy
    folder = filepath + "/"
    file = folder+ "temp_beam/temp"
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
        tao = edit_energy_based_on_beam_all(tao, start, file, verbose=verbose, desired_beam_energies=desired_beam_energies_for_the_feedback, finalnumMacroParticles=N_to_use_from_file)

    # run the sim and save the bunch
    if run:
        pre = 'temp_beam/'
        suf = 'temp'
        if locationsToSave == []:
            locationsToSave = [start, finish]
        if edited_bunch_energy_at_checkpoints_MeV!=[None, None, None, None]:
            tao = run_initialized_sim_edit_bunch_energy(tao, start, finish, edited_bunch_energy_at_checkpoints_MeV=edited_bunch_energy_at_checkpoints_MeV, pre=pre, suf=suf, locations=locationsToSave)
        else:
            tao = run_initialized_sim(tao, locationsToSave[0], locationsToSave[-1], pre, suf, locationsToSave, desired_P0Cs_MeV=desired_P0Cs_MeV)
    return tao

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

def run_initialized_sim(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], desired_P0Cs_MeV=[None,None,None,None]):
    '''
    tunes the cavities to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the BX0FBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BX0FEND
    tunes them to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the BC11CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC11CEND
    tunes them to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the BC14CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC14CEND
    tunes them to the provided edited_bunch_energy_at_checkpoints_MeV
    tracks to the BC20CBEG
    tunes them back to 125, 335, 4500, 10000
    tracks to the BC20CEND
    tunes them to the provided edited_bunch_energy_at_checkpoints_MeV
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
        trackBeam(tao, trackStart = current_start, trackEnd = "BX0FBEG")
        getBeamAtElement(tao, "BX0FBEG").write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000], change_only_L0B=True)        
        current_start = "BX0FBEG"
        trackBeam(tao, trackStart = current_start, trackEnd = "BX0FEND")
        #print(f'<x> inside of the dogleg: {tao.bunch_params("BPM10731")["centroid_vec_1"]}')
        getBeamAtElement(tao, "BX0FEND").write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BX0FEND"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L1
    if desired_P0Cs_MeV[1] is not None:
        trackBeam(tao, trackStart = current_start, trackEnd = "BC11CBEG")
        getBeamAtElement(tao, "BC11CBEG").write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BC11CBEG"
        trackBeam(tao, trackStart = current_start, trackEnd = "BC11CEND")
        getBeamAtElement(tao, "BC11CEND").write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BC11CEND"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L2
    if desired_P0Cs_MeV[2] is not None:
        trackBeam(tao, trackStart = current_start, trackEnd = "BEGBC14E")
        getBeamAtElement(tao, "BEGBC14E").write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BEGBC14E"
        trackBeam(tao, trackStart = current_start, trackEnd = "ENDBC14E")
        getBeamAtElement(tao, "ENDBC14E").write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDBC14E"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # L3
    if desired_P0Cs_MeV[3] is not None:
        trackBeam(tao, trackStart = current_start, trackEnd = "BEGBC20")
        getBeamAtElement(tao, "BEGBC20").write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000])
        current_start = "BEGBC20"
        trackBeam(tao, trackStart = current_start, trackEnd = "ENDBC20")
        getBeamAtElement(tao, "ENDBC20").write(tao.filePathGlobal+"/"+"temp_beam/temp.h5")
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDBC20"
        tune_to_P0Cs(tao, desired_P0Cs_MeV=desired_P0Cs_MeV)
    # Fin
    trackBeam(tao, trackStart = current_start, trackEnd = finish)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind])
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


def run_initialized_sim_edit_bunch_energy(tao, start, finish, pre='temp_beam/', suf='temp', locations=[], edited_bunch_energy_at_checkpoints_MeV=[None,None,None,None]):
    '''
    Tracks the bunch and changes the bunch energy at BX0FBEG, BC11CBEG, ENDL2F, ENDL3F_2 if any of edited_bunch_energy_at_checkpoints_MeV (list with 4 numbers) is not -1.
    '''
    if locations==[]:
        locations = [start, finish]
    if locations==None:
        locations=[]
    current_start = start
    if edited_bunch_energy_at_checkpoints_MeV[0] is not None:
        trackBeam(tao, trackStart = current_start, trackEnd = "BX0FBEG")
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "BX0FBEG"), pzMeV=edited_bunch_energy_at_checkpoints_MeV[0], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BX0FBEG"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[1] is not None:
        trackBeam(tao, trackStart = current_start, trackEnd = "BC11CBEG")
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "BC11CBEG"), pzMeV=edited_bunch_energy_at_checkpoints_MeV[1], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "BC11CBEG"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[2] is not None:
        trackBeam(tao, trackStart = current_start, trackEnd = "ENDL2F")
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "ENDL2F"), pzMeV=edited_bunch_energy_at_checkpoints_MeV[2], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDL2F"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    if edited_bunch_energy_at_checkpoints_MeV[3] is not None:
        trackBeam(tao, trackStart = current_start, trackEnd = "ENDL3F_2")
        edit_bunch_parameters_from_PG(getBeamAtElement(tao, "ENDL3F_2"), pzMeV=edited_bunch_energy_at_checkpoints_MeV[3], means=[None, None, None, None, None], path_to_write=tao.filePathGlobal+"/"+"temp_beam/temp")
        current_start = "ENDL3F_2"
        set_beam(tao, tao.filePathGlobal+"/"+"temp_beam/temp")

    
    trackBeam(tao, trackStart = current_start, trackEnd = finish)

    for ind in range(len(locations)):
        P = getBeamAtElement(tao, locations[ind])
        P.write(tao.filePathGlobal+"/"+pre+locations[ind]+suf +'.h5')

    return tao


### Correct the lattice to match the desired Pz

def tune_to_P0Cs(tao, desired_P0Cs_MeV=[125, 335, 4500, 10000], change_only_L0B=False):
    '''
    This function scales the cavities to match the desired_P0Cs_MeV.
    change_only_L0B: see the "edit_energy_based_on_beam_inj" function.
    '''
    desired_P0Cs_MeV = [desired_P0Cs_MeV[0] if desired_P0Cs_MeV[0] is not None else 125,
                        desired_P0Cs_MeV[1] if desired_P0Cs_MeV[1] is not None else 335,
                        desired_P0Cs_MeV[2] if desired_P0Cs_MeV[2] is not None else 4500,
                        desired_P0Cs_MeV[3] if desired_P0Cs_MeV[3] is not None else 10000]
    if change_only_L0B:
        current_pz_location = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
        current_pz_start = tao.ele_gen_attribs('L0AFEND')["P0C"]*1e-6
        coef_e = 1 + (desired_P0Cs_MeV[0] - current_pz_location)/(current_pz_location-current_pz_start)
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    else:
        current_pz_location = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
        current_pz_start = tao.ele_gen_attribs('BEGINNING')["P0C"]*1e-6
        coef_e = 1 + (desired_P0Cs_MeV[0] - current_pz_location)/(current_pz_location-current_pz_start)
        l0aVoltage = tao.ele_gen_attribs('L0AF')["VOLTAGE"]
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0AF VOLTAGE = {l0aVoltage*coef_e}')
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')

    setLinacGradientAuto(tao, "L1", (desired_P0Cs_MeV[1]-desired_P0Cs_MeV[0])*1e6)
    setLinacGradientAuto(tao, "L2", (desired_P0Cs_MeV[2]-desired_P0Cs_MeV[1])*1e6)
    setLinacGradientAuto(tao, "L3", (desired_P0Cs_MeV[3]-desired_P0Cs_MeV[2])*1e6)

    return tao


#### Treat dipoles

def treat_dipoles(tao):
    '''
    A test function to deal with the dipoles. Hopefully I will find the solution...
    '''
    initial_energy_bx = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
    initial_energy_bc11 = tao.ele_gen_attribs('BC11CBEG')["P0C"]*1e-6
    initial_energy_bc14 = tao.ele_gen_attribs('ENDL2F')["P0C"]*1e-6
    initial_energy_bc20 = tao.ele_gen_attribs('ENDL3F_2')["P0C"]*1e-6

    current_pz_location = tao.ele_gen_attribs('BX0FBEG')["P0C"]*1e-6
    current_pz_start = tao.ele_gen_attribs('BEGINNING')["P0C"]*1e-6
    coef_e = 1 + (125 - current_pz_location)/(current_pz_location-current_pz_start)
    l0aVoltage = tao.ele_gen_attribs('L0AF')["VOLTAGE"]
    l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
    tao.cmd(f'set ele L0AF VOLTAGE = {l0aVoltage*coef_e}')
    tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')

    setLinacGradientAuto(tao, "L1", (335-125)*1e6)
    setLinacGradientAuto(tao, "L2", (4500-335)*1e6)
    setLinacGradientAuto(tao, "L3", (10000-4500)*1e6)

    elems = get_element_array(tao, "BEGINNING", "END", values_to_show=["SBend"])[:,1]
    for ele in elems:
        tao.cmd(f'set ele {ele} FIELD_MASTER = {True}')
    
    tao.cmd(f'set global lattice_calc_on = F')

    tao.cmd(f'set ele L0AF VOLTAGE = {l0aVoltage}')
    tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage}')

    setLinacGradientAuto(tao, "L1", (initial_energy_bc11-initial_energy_bx)*1e6)
    setLinacGradientAuto(tao, "L2", (initial_energy_bc14-initial_energy_bc11)*1e6)
    setLinacGradientAuto(tao, "L3", (initial_energy_bc20-initial_energy_bc14)*1e6)

    return tao

fields = {
    'BCX10451': 0.4399498913496881,
    'BCX10461': -0.4399498913496881,
    'BCX10475': -0.4399498913496881,
    'BCX10481': 0.4399498913496881,
    'BX10661': 0.6242944753331842,
    'BX10751': 0.6242944753331842,
    'BCX11314': 0.5167328829297726,
    'BCX11331': -0.5167328829297726,
    'BCX11338': -0.5167328829297726,
    'BCX11355': 0.5167328829297726,
    'BCX14720': 1.145800533192734,
    'BCX14796': -1.145800533192734,
    'BCX14808': -1.145800533192734,
    'BCX14883': 1.145800533192734,
    'B1LE': -0.7088897708589302,
    'B2LE': 0.5998061022132172,
    'B3LE': -0.6450166589406902,
    'B3RE': -0.6450166589406902,
    'B2RE': 0.5998061022132172,
    'WIGE1': 0.3417645361988085,
    'WIGE2': -0.3417645361988085,
    'WIGE3': 0.3417645361988085,
    'B1RE': -0.7088897708589302,
    'B5D36': -0.2046605187706695
}

def treat_dipoles1(tao):
    '''
    A test function to deal with the dipoles. Hopefully I will find the solution...
    '''
    tao.cmd(f'set global lattice_calc_on = F')

    elems = get_element_array(tao, "BEGINNING", "END", values_to_show=["SBend"])[:,1]
    for k, v in fields.items():
        #tao.cmd(f'set ele {ele} FIELD_MASTER = {True}')
        tao.cmd(f'set ele {k} B_FIELD = {v}')

    #tao.cmd(f'set global lattice_calc_on = T')

    return tao


#### Adjust based on the beam


def edit_energy_based_on_beam_inj(tao, location="BX0FBEG", desiredPzMeV=125, change_only_L0B=False):
    '''
    Scales the injector cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    change_only_L0B: scale only L0B. I think that at FACET we do exactly that.
    If changing both L0A and L0B, the input beam energy at L0A is adjusted accordingly later.
    '''
    if change_only_L0B:
        current_pz_location = np.mean(getBeamAtElement(tao, location).pz)
        current_pz_start = np.mean(getBeamAtElement(tao, "L0AFEND").pz)
        coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    else:
        current_pz_location = np.mean(getBeamAtElement(tao, location).pz)
        current_pz_start = tao.ele_gen_attribs('BEGINNING')["P0C"]
        #current_pz_location = tao.ele_gen_attribs(location)["P0C"]
        coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
        l0aVoltage = tao.ele_gen_attribs('L0AF')["VOLTAGE"]
        l0bVoltage = tao.ele_gen_attribs('L0BF')["VOLTAGE"]
        tao.cmd(f'set ele L0AF VOLTAGE = {l0aVoltage*coef_e}')
        tao.cmd(f'set ele L0BF VOLTAGE = {l0bVoltage*coef_e}')
    return tao

def edit_energy_based_on_beam_L1(tao, location="BC11CBEG", desiredPzMeV=335):
    '''
    Scales the L1 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'BX0FBEG').pz)
    activeMatchStrings = matchStringWrapper(tao, "L1")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao

def edit_energy_based_on_beam_L2(tao, location="ENDL2F", desiredPzMeV=4500):
    '''
    Scales the L2 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'BC11CBEG').pz)
    activeMatchStrings = matchStringWrapper(tao, "L2")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao

def edit_energy_based_on_beam_L3(tao, location="ENDL3F_2", desiredPzMeV=10000):
    '''
    Scales the L3 cavities to match the desired beam energy at the dogleg (desiredPzMeV) in MeV.
    location: where to look at the beam energy (it changes with s if wake fields or CSR are enabled)
    '''
    current_pz_location = np.mean(getBeamAtElement(tao, location).pz)
    current_pz_start = np.mean(getBeamAtElement(tao, 'ENDL2F').pz)
    activeMatchStrings = matchStringWrapper(tao, "L3")
    coef_e = 1 + 1e6*(desiredPzMeV - current_pz_location*1e-6)/(current_pz_location-current_pz_start)
    for i in activeMatchStrings:
        g = tao.ele_gen_attribs(i)["GRADIENT"]
        tao.cmd(f'set ele {i} GRADIENT = {g*coef_e}')
    return tao

def edit_energy_based_on_beam_all(tao, start, file, desired_beam_energies=None, change_only_L0B=False, change_file_pz=True, verbose=False, finalnumMacroParticles=5e4):
    '''
    This function changes the cavity voltages so that the tracked beam has the "desired_P0Cs" <pz> between the cavities.
    desired_P0Cs: the desired <pz> in eV between the cavities. Must be None or a list of 4 numbers (dogleg, bc11, bc14, bc20).
    If None, the P0Cs from the lattice are used.
    start: start of the simulation. Need to be in the injector (use L0AFEND to avoid confusion).
    file: the bunch file to use for the tuning. Providing it is a must because the energy obviously depends on the charge and the size of the bunch.
    change_only_L0B: in the injector, scale only L0B. I think that at FACET we do exactly that.
    If changing both L0A and L0B, the input beam energy at L0A is adjusted accordingly later.
    change_file_pz: change the bunch energy at the "start" to match the "start" P0C.
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
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to BX0FBEG: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG").pz))]}')
    
    tao = edit_energy_based_on_beam_inj(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[0]*1e-6, change_only_L0B=change_only_L0B)
    if verbose:
        print(f'edited P0C BX0FBEG: {tao.ele_gen_attribs("BX0FBEG")["P0C"]}')
    
    # L1
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "BC11CBEG"]
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to BC11CBEG: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG").pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG").pz))]}')
    
    tao = edit_energy_based_on_beam_L1(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[1]*1e-6)
    if verbose:
        print(f'edited P0C BC11CBEG: {tao.ele_gen_attribs("BC11CBEG")["P0C"]}')
    
    # L2
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "ENDL2F"]
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to ENDL2F: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG").pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG").pz)), float(np.mean(getBeamAtElement(tao, "ENDL2F").pz))]}')
    
    tao = edit_energy_based_on_beam_L2(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[2]*1e-6)
    if verbose:
        print(f'edited P0C ENDL2F: {tao.ele_gen_attribs("ENDL2F")["P0C"]}')
    
    # L3
    if change_file_pz:
        edit_bunch_parameters(file, pzMeV=tao.ele_gen_attribs(start)["P0C"]*1e-6, moments=[None,None,None,None,None,None], means=[0,0,0,0,0], charge=-1, path_to_write=file)
    set_beam(tao, file, numMacroParticles = 5e4)
    locations = [start, "ENDL3F_2"]
    tao = run_initialized_sim(tao, locations[0], locations[-1], pre, suf, locations)
    if verbose:
        print(f'beam to ENDL3F_2: {[float(np.mean(getBeamAtElement(tao, "BX0FBEG").pz)), float(np.mean(getBeamAtElement(tao, "BC11CBEG").pz)), float(np.mean(getBeamAtElement(tao, "ENDL2F").pz)), float(np.mean(getBeamAtElement(tao, "ENDL3F_2").pz))]}')
    
    tao = edit_energy_based_on_beam_L3(tao, location=locations[-1], desiredPzMeV=desired_beam_energies[3]*1e-6)
    if verbose:
        print(f'edited P0C ENDL3F_2: {tao.ele_gen_attribs("ENDL3F_2")["P0C"]}')
    
    set_beam(tao, file, numMacroParticles=finalnumMacroParticles)
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
    #'Q1EL' - same as Q1ER?
    #'SQ1': should be ['nonBSA_List_S20Magnets', 'LI20_LGPS_2086_BACT'], but not saved in DAQ?,
    'Q1EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2060_BACT'],
    'Q2EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2130_BACT'],
    'Q3EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q3EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q4EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4EL_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q5EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2230_BACT'],
    'Q5ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2230_BACT'],
    'Q6E': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2251_BACT'],
    'Q4ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q4ER_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2200_BACT'],
    'Q3ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q3ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2150_BACT'],
    'Q2ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2130_BACT'],
    'Q1ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2060_BACT'],
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
    #'Q1EL' - same as Q1ER?
    'Q1EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2441_BACT'],
    'Q2EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2131_BACT'],
    'Q3EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2151_BACT'],
    'Q3EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2151_BACT'],
    'Q4EL_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q4EL_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q4EL_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2201_BACT'],
    'Q5EL': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2231_BACT'],
    'Q5ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2262_BACT'],
    #'Q6E': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2251_BACT'],
    'Q4ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q4ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q4ER_3': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2281_BACT'],
    'Q3ER_1': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2341_BACT'],
    'Q3ER_2': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2341_BACT'],
    'Q2ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2371_BACT'],
    'Q1ER': ['nonBSA_List_S20Magnets', 'LI20_LGPS_2441_BACT']
}

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
    'YC11399': ["nonBSA_List_S11", 'YCOR_LI11_399_BCON']
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
    # tao.cmd(f'set ele L0BF PHI0 = {l0bphase / 360.}')
    # tao.cmd(f'set ele L0BF VOLTAGE = {(61.0e6 + (mean_energy_MeV_lattice-125)*1e6) / math.cos(2*math.pi*l0bphase/360)}')

    tao = edit_energy_tao_based_on_experiment_database(tao, dataset)

    for k, v in bmad_quad_to_pv_map.items():
        quad_integrated_T = np.mean(dataset._data["scalars"][v[0]][v[1]])
        if k in bmad_quad_to_pv_map_boost:
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
    
    sextXOffsets = np.array([0,0,0,0,0,0])
    sextYOffsets = np.array([0,0,0,0,0,0])
    for i in range(2):
        sextXOffsets[i] = np.mean(dataset._data["scalars"][sextupole_offsets_x_from_daq[i][0]][sextupole_offsets_x_from_daq[i][1]])*1e-3
        sextYOffsets[i] = np.mean(dataset._data["scalars"][sextupole_offsets_y_from_daq[i][0]][sextupole_offsets_y_from_daq[i][1]])*1e-3
    for i in range(2):
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


## Scans

def make_1d_scan(tao, mean=0, nscan=21, scan_span=5e-3, function_to_change_tao_in_scan=None, function_to_get_results_in_scan=None, plot=True, label="y", xlabel="x axis", ylabel="y axis", **kwargs):
    """Perform a 1D parameter scan over Tao and optionally plot the results.
    """
    scan_values = scan_span*(np.arange(nscan)-(nscan-1)/2)/(nscan-1)+mean
    output = []
    for value_ind in range(nscan):
        tao = function_to_change_tao_in_scan(tao, scan_values[value_ind], **kwargs)
        output.append(function_to_get_results_in_scan(tao, **kwargs))
    output = np.array(output)
    if plot:
        make_a_plot(scan_values, output, label=label, x_label=xlabel, y_label=ylabel, cartesian_axes=[False, False], axes_location=[0, 0])
    return scan_values, output


def make_comparison_dz_2nd_order(tao, beam_file='temp_beam/temp', start='L0AFEND', finish='PR11375', means_shift=[0,0,0,0,0,0], edited_bunch_energy_at_checkpoints_MeV=[-1,-1,-1,-1], theory=True, **kwargs):
    """Compare simulation and second-order theory for longitudinal bunch size growth."""
    # sim
    set_beam(tao, beam_file)
    run_initialized_sim(tao, start, finish, edited_bunch_energy_at_checkpoints_MeV=edited_bunch_energy_at_checkpoints_MeV)
    
    dzsim = 3e8*(getBeamAtElement(tao, finish).t - np.mean(getBeamAtElement(tao, finish).t))
    size_sim = np.sqrt(np.mean(dzsim**2) - np.mean(dzsim)**2)

    # theory
    if theory:
        beam = make_simple_bunch_theory_from_bunch_sims(beam_file+"_e", np.mean(ParticleGroup(beam_file+"_e.h5").pz)*1e-6, means_shift=means_shift)
        
        r5i = np.array([float(get_rij(tao, start, finish, 5, j+1)) for j in range(6)])
        t5ij = np.zeros((6,6))
        for j in range(6):
            for k in range(6):
                t5ij[j][k] = float(get_tijk(tao, start, finish, 5, j+1, k+1))
        nonlinear_impact = np.zeros(len(beam))
        for j in range(6):
            for k in range(6):
                nonlinear_impact += (t5ij[j][k]*(beam[:, j]*beam[:, k]) if j>=k else 0)
        dz = np.sum([r5i[j]*beam[:, j] for j in range(6)], axis=0) + nonlinear_impact
        dz = dz - np.mean(dz)
        size_theory = np.sqrt(np.mean(dz**2) - np.mean(dz)**2)
        
        return size_sim, size_theory
    else:
        return size_sim
    

## Simulation parameters functions

def get_element_array(tao, beg, end, values_to_show=[], values_to_remove=[], marginl=0, marginr=0):
    """Return a subset of lattice elements between two locations, filtered by element type."""
    keys = tao.lat_list("*", "ele.key")
    ss = tao.lat_list("*", "ele.s")
    names = tao.lat_list("*", "ele.name")
    
    elements = np.stack([ss, names, keys], axis=1)
    
    location1 = np.argwhere(elements==beg)[0,0]-marginl
    location2 = np.argwhere(elements==end)[0,0]+marginr
    
    trunc_array = elements[location1:location2+1]
    show_array = trunc_array[np.isin(trunc_array[:, 2], values_to_show)] if (len(values_to_show)>0) else trunc_array
    clean_array = show_array[~np.isin(trunc_array[:, 2], values_to_remove)] if (len(values_to_remove)>0) else show_array
    
    return clean_array

def get_tijk(tao, loc1, loc2, i_0, j_0, k_0):
    """Read a second-order Taylor map coefficient (T) from Tao between two locations."""
    n = 6
    t5ijterms = [
        [
            tuple(1 if k == i or k == j else 0 for k in range(n)) if i != j
            else tuple(2 if k == i else 0 for k in range(n))
            for j in range(n)
        ]
        for i in range(n)
    ]
    mapterms = tao.taylor_map(loc1, loc2, order='2', verbose=False, as_dict=True, raises=True)[i_0]
    return mapterms[t5ijterms[j_0-1][k_0-1]] if t5ijterms[j_0-1][k_0-1] in mapterms else 0

def get_rij(tao, loc1, loc2, i, j):
    """Read an R-matrix element from Tao between two locations."""
    r5i = np.zeros(6)
    s = tao.cmd("".join(["show matrix ", loc1, " ", loc2]))[i+1]
    numeric_part = s.split(':')[0]
    nums = [float(x) for x in numeric_part.split()]
    r5i = np.array(nums)
    return r5i[j-1]


### Sextupole settings

def setAllWChicaneSextupoles(tao, S1ELkG, S2ELkG, S3ELkG, S3ERkG, S2ERkG, S1ERkG):
    """Set all chicane sextupole strengths in the Tao lattice."""
    setSextkG(tao, "S1EL",   S1ELkG)
    setSextkG(tao, "S2EL",   S2ELkG)
    setSextkG(tao, "S3EL_1", S3ELkG)
    setSextkG(tao, "S3EL_2", S3ELkG)
    setSextkG(tao, "S3ER_1", S3ERkG)
    setSextkG(tao, "S3ER_2", S3ERkG)
    setSextkG(tao, "S2ER",   S2ERkG)
    setSextkG(tao, "S1ER",   S1ERkG)
    return tao

def setAllWChicaneSextupolesXOffsets(tao, S1EL_dx, S2EL_dx, S3EL_dx, S3ER_dx, S2ER_dx, S1ER_dx):
    """Set all chicane sextupole horizontal offsets in the Tao lattice."""
    tao.cmd(f'set ele {"S1EL"} X_OFFSET = {S1EL_dx}')
    tao.cmd(f'set ele {"S2EL"} X_OFFSET = {S2EL_dx}')
    tao.cmd(f'set ele {"S3EL_1"} X_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3EL_2"} X_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3ER_1"} X_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S3ER_2"} X_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S2ER"} X_OFFSET = {S2ER_dx}')
    tao.cmd(f'set ele {"S1ER"} X_OFFSET = {S1ER_dx}')
    return tao

def setAllWChicaneSextupolesYOffsets(tao, S1EL_dx, S2EL_dx, S3EL_dx, S3ER_dx, S2ER_dx, S1ER_dx):
    """Set all chicane sextupole vertical offsets in the Tao lattice."""
    tao.cmd(f'set ele {"S1EL"} Y_OFFSET = {S1EL_dx}')
    tao.cmd(f'set ele {"S2EL"} Y_OFFSET = {S2EL_dx}')
    tao.cmd(f'set ele {"S3EL_1"} Y_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3EL_2"} Y_OFFSET = {S3EL_dx}')
    tao.cmd(f'set ele {"S3ER_1"} Y_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S3ER_2"} Y_OFFSET = {S3ER_dx}')
    tao.cmd(f'set ele {"S2ER"} Y_OFFSET = {S2ER_dx}')
    tao.cmd(f'set ele {"S1ER"} Y_OFFSET = {S1ER_dx}')
    return tao


## Plotting functions

def enable_plt_styling():
    """Enable APS/PRAB-like matplotlib styling for plots.
    """
    # APS / PRAB-like matplotlib configuration
    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 14,
        "axes.titlesize": 14,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 11,
        "axes.linewidth": 1.0,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
    })


### Display a bunch

# New PlotMod for horizontal plotting

def plotMod(particle_group, key1='t', key2='p', 
                  bins=None,
                  *,
                  xlim=None,
                  ylim=None,
                  tex=True,
                  nice=True,
            fig=None, outer=None, i=None, z_from_t=False,
                  **kwargs):
    """
    Derived from openPMD-beamphysics marginal_plot()
    """    

    #plt.close('all')
    
    CMAP0 = copy(plt.get_cmap('viridis'))
    CMAP0.set_under(CMAP0(0))  # set under-color to the lowest colormap color
    CMAP1 = copy(plt.get_cmap('plasma'))

    plt.ioff()
    
    if not bins:
        n = len(particle_group)
        bins = int(np.sqrt(n/4) )

    key1changed = False
    key2changed = False
    if z_from_t:
        if key1=='z':
            key1='delta_t'
            key1changed = True
        if key2=='z':
            key2='delta_t'
            key2changed = True

    # Scale to nice units and get the factor, unit prefix
    x = particle_group[key1]
    y = particle_group[key2]

    if key1changed:
        x = -3e8*x
    if key2changed:
        y = -3e8*x
    
    # Form nice arrays
    x, f1, p1, xmin, xmax = pmd_beamphysics.units.plottable_array(x, nice=nice, lim=xlim)
    y, f2, p2, ymin, ymax = pmd_beamphysics.units.plottable_array(y, nice=nice, lim=ylim)

    if key1changed:
        x = f1*x
    if key2changed:
        y = f2*x
    
    w = particle_group['weight']
    
    u1 = particle_group.units(key1).unitSymbol
    u2 = particle_group.units(key2).unitSymbol
    ux = p1+u1
    uy = p2+u2
    
    labelx = pmd_beamphysics.labels.mathlabel(key1, units=ux, tex=tex)
    labely = pmd_beamphysics.labels.mathlabel(key2, units=uy, tex=tex)

    if key1changed:
        labelx = "z (m)"
    if key2changed:
        labely = "z"

    if (fig is None or outer is None or i is None):
        fig = plt.figure(**kwargs)
        gs = GridSpec(4,4)
        ax_joint = fig.add_subplot(gs[1:4,0:3])
        ax_marg_x = fig.add_subplot(gs[0,0:3])
        ax_marg_y = fig.add_subplot(gs[1:4,3])
    else:
        gs = GridSpecFromSubplotSpec(
            4, 4,
            subplot_spec=outer[i],
            wspace=0.0,
            hspace=0.0
        )
        
    ax_joint = fig.add_subplot(gs[1:4, 0:3])
    ax_marg_x = fig.add_subplot(gs[0, 0:3], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(gs[1:4, 3], sharey=ax_joint)

    # Set the joint plot background color to match the bottom end of the colormap
    #ax_joint.set_facecolor(CMAP0(0))
    ax_joint.set_facecolor('white')
    
    # Plot the hexbin
    ax_joint.hexbin(x, y, C=w, reduce_C_function=np.sum, gridsize=bins, cmap=CMAP0, vmin=1e-20)
    
    # Top histogram
    hist, bin_edges = np.histogram(x, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = pmd_beamphysics.units.nice_array(hist/hist_width)
    ax_marg_x.bar(hist_x, hist_y, hist_width, color='gray')
    if u1 == 's':
        _, hist_prefix = pmd_beamphysics.units.nice_scale_prefix(hist_f/f1)
        ax_marg_x.set_ylabel(f'{hist_prefix}A')
    else:   
        ax_marg_x.set_ylabel(pmd_beamphysics.labels.mathlabel(f'{hist_prefix}C/{ux}'))

    # Side histogram
    hist, bin_edges = np.histogram(y, bins=bins, weights=w)
    hist_x = bin_edges[:-1] + np.diff(bin_edges) / 2
    hist_width =  np.diff(bin_edges)
    hist_y, hist_f, hist_prefix = pmd_beamphysics.units.nice_array(hist/hist_width)
    ax_marg_y.barh(hist_x, hist_y, hist_width, color='gray')
    ax_marg_y.set_xlabel(pmd_beamphysics.labels.mathlabel(f'{hist_prefix}C/{uy}'))

    # Turn off tick labels on marginals
    plt.setp(ax_marg_x.get_xticklabels(), visible=False)
    plt.setp(ax_marg_y.get_yticklabels(), visible=False)
    
    # Set labels on joint
    ax_joint.set_xlabel(labelx)
    ax_joint.set_ylabel(labely)
    
    if xlim:
        ax_joint.set_xlim(xmin/f1, xmax/f1)      
        ax_marg_x.set_xlim(xmin/f1, xmax/f1)
        
    if ylim:
        ax_joint.set_ylim(ymin/f2, ymax/f2)     
        ax_marg_y.set_ylim(ymin/f2, ymax/f2)
    
    return ax_joint, ax_marg_x, ax_marg_y

# Prints specified 2d spaces, plots are arranged horizontally. Uncomment a section for vertical plotting.
def print_result(particle_group, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, drift_to_z_in_cutting = True, z_from_t=False):
    """Display 2D phase-space plots and print basic bunch moments for a ParticleGroup."""
    P = particle_group.copy()
    #P.drift_to_z()
    if length != 0:
        Pt = cut_length(P, length, drift_to_z = drift_to_z_in_cutting)
    else:
        Pt = P

    l = len(couples)
    fig = plt.figure(figsize=(7*l,6))
    
    # outer layout: 1 row, l columns
    outer = GridSpec(1, l, wspace=0.3)
    
    for (i,couple) in enumerate(couples):
        plotMod(Pt, couple[0], couple[1], bins=300, fig=fig, outer=outer, i=i, z_from_t=z_from_t)

    plt.show()

    # if column-arranging is needed
    # for (i,couple) in enumerate(couples):
    #     #display(plotMod(Pt, couple[0], couple[1],  bins=300))
    #     plt.subplot(1,len(couples[:,1]),i+1)
    #     plotMod(Pt, couple[0], couple[1],  bins=300)
    #     plt.show()
        
    if moments:
        deltas = (P.gamma-np.mean(P.gamma))/np.mean(P.gamma)
        energies = P.pz
        energies = P.energy
        if sliceEnergyMoment:
            Pslice = cut_length(P, length = 1e-7)
            deltas = (Pslice.gamma-np.mean(Pslice.gamma))/np.mean(Pslice.gamma)
            energies = Pslice.pz
        if energyInDelta:
            sigmapz = moment(deltas, moment=2) ** 0.5
        else:
            sigmapz = moment(energies, moment=2) ** 0.5
        print([float(moment(Pt.x, moment=2) ** 0.5), float(moment(Pt.xp, moment=2) ** 0.5), float(moment(Pt.y, moment=2) ** 0.5), float(moment(Pt.yp, moment=2) ** 0.5), float(moment(Pt.z, moment=2) ** 0.5), float(sigmapz), float(moment(Pt.t, moment=2) ** 0.5)])


def print_result_from_tao(tao_local, location, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, z_from_t=False):
    """Display results (2d distribution plots) for a beam extracted from Tao at a given location."""
    print_result(getBeamAtElement(tao_local, location), length = length, couples = couples, moments = moments, sliceEnergyMoment=sliceEnergyMoment, energyInDelta=energyInDelta, z_from_t=z_from_t)

def print_result_from_file(file, length = 0, couples = [['t','pz']], moments = True, sliceEnergyMoment=False, energyInDelta=False, z_from_t=False):
    """Display results (2d distribution plots) for a beam loaded from an HDF5 file."""
    print_result(ParticleGroup(file), length = length, couples = couples, moments = moments, sliceEnergyMoment=sliceEnergyMoment, energyInDelta=energyInDelta, z_from_t=z_from_t)


## Make a nice plot from arrays

def normalize_arrays(arrays):
    """Normalize input arrays into a list of numpy arrays for plotting utilities."""
    # Case 1: single NumPy array
    if isinstance(arrays, np.ndarray):
        if arrays.ndim == 1:
            return [arrays]          # wrap single dataset
        elif arrays.ndim == 2:
            return list(arrays)      # split into rows
        else:
            raise ValueError("Array must be 1D or 2D")

    # Case 2: iterable of arrays (list/tuple/etc.)
    try:
        return [np.asarray(a) for a in arrays]
    except TypeError:
        raise ValueError("Input must be an array or iterable of arrays")

def make_a_plot(x, y, errs=None, aspect_ratio=0.67, label=r"$\sin(x)$", x_label=r"$x$", y_label=r"$y$", cartesian_axes=[True, True], axes_location=[0, 0], colors=['blue', 'black']):
    """Create a publication-quality line plot from x/y arrays with optional error bands."""
    x = normalize_arrays(x)
    y = normalize_arrays(y)
    if errs is not None:
        errs = normalize_arrays(errs)
    nLines = len(x)
    if nLines==1:
        label=[label]
        
    if len(y_label)==2:
        fig, ax1 = plt.subplots(figsize=(6.0, 6.0*aspect_ratio))

        ax1.plot(x[0], y[0], lw=1.8, label=label[0], c=colors[0]) if errs is None else ax1.errorbar(x[0], y[0], errs[0], lw=1.8, label=label[0], c=colors[0])
        ax1.set_xlabel(x_label)
        ax1.set_ylabel(y_label[0])
        
        ax2 = ax1.twinx()
        ax2.plot(x[1], y[1], lw=1.8, label=label[1], c=colors[1]) if errs is None else ax2.errorbar(x[1], y[1], errs[1], lw=1.8, label=label[1], c=colors[1])
        ax2.set_xlabel(x_label)
        ax2.set_ylabel(y_label[1])
    
        if cartesian_axes[1]:
            ax2.axhline(axes_location[1], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
        if cartesian_axes[0]:
            ax2.axvline(axes_location[0], linestyle="--", linewidth=1.0, color="0.6", zorder=0)

        
        plt.setp(ax1.get_xticklabels(), fontsize=14)
        plt.setp(ax2.get_yticklabels(), fontsize=14)
        plt.setp(ax1.get_yticklabels(), fontsize=14)
        
        # Minor ticks
        ax2.xaxis.set_minor_locator(AutoMinorLocator())
        ax2.yaxis.set_minor_locator(AutoMinorLocator())
        # Tick parameters
        ax2.tick_params(which="both", width=1)
        ax2.tick_params(which="major", length=6)
        ax2.tick_params(which="minor", length=3)
        
        h1, lab1 = ax1.get_legend_handles_labels()
        h2, lab2 = ax2.get_legend_handles_labels()
        ax1.legend(h1 + h2, lab1 + lab2, loc='best', frameon=False,handlelength=2.0)

    else:
        fig, ax = plt.subplots(figsize=(6.0, 6.0*aspect_ratio))
        for i, a in enumerate(x):
            ax.plot(x[i], y[i], lw=1.8, label=label[i]) if errs is None else ax.errorbar(x[i], y[i], errs[i], lw=1.8, label=label[i])
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    
        if cartesian_axes[1]:
            ax.axhline(axes_location[1], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
        if cartesian_axes[0]:
            ax.axvline(axes_location[0], linestyle="--", linewidth=1.0, color="0.6", zorder=0)
        
        # Minor ticks
        ax.xaxis.set_minor_locator(AutoMinorLocator())
        ax.yaxis.set_minor_locator(AutoMinorLocator())
        # Tick parameters
        ax.tick_params(which="both", width=1)
        ax.tick_params(which="major", length=6)
        ax.tick_params(which="minor", length=3)
        # Legend
        #loc="upper right",
        ax.legend(loc="best",frameon=False,handlelength=2.0)
        
    # Tight layout for journal export
    fig.tight_layout(pad=0.3)
    
    # Save (recommended formats for journals)
    # fig.savefig("figure.pdf")
    # fig.savefig("figure.eps")
    plt.show()