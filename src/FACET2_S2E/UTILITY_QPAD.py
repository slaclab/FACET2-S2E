
import importlib
from . import UTILITY_QPAD_PICMI as picmi
import numpy as np
from pmd_beamphysics import ParticleGroup
import h5py
from scipy.optimize import fsolve
from scipy.special import erf
from scipy.interpolate import interp1d
from scipy.constants import physical_constants
import scipy.constants as cst
import subprocess, os, sys, yaml

import matplotlib.pyplot as plt
from ipywidgets import VBox, HBox, IntSlider, Text, FloatText, Button, Output, Label

from openpmd_viewer import OpenPMDTimeSeries
from matplotlib_inline.backend_inline import set_matplotlib_formats
import matplotlib
from matplotlib.ticker import FuncFormatter
import glob
import json

class QPAD_sim:

    # ParticleGroup
    P = None

    # list of species (beams, plasma, and neutrals)
    species_list = []

    # list of particles layouts (ppc data)
    layouts = []

    # Particle Diagnostics 
    if_beam = []
    part_diags = []

    # Field Diagnostics
    field_diags = []

    # simulation, solver and grid structs
    simulation, solver, grid = None, None, None

    """
    Constructor

    Parameters
    ----------
    n0: float
        Normalizing Density in units of m^{-3}

    """
    def __init__(self, n0 = 1e17 * 1e6):
        self.n0 = n0
        self.wp = np.sqrt(cst.e**2 * self.n0/(cst.epsilon_0 * cst.m_e))
        self.kp = self.wp/cst.c
        self.P = None
        self.layouts, self.species_list = [],[]
        self.simulation, self.solver, self.grid = None,None,None
        self.if_beam, self.part_diags, self.field_diags = [],[],[]
        importlib.reload(picmi)

    """
    Initialize Grid Paramters

    Parameters
    ----------
    nr, nz: integer
        Number of grid cells along r and z, respectfully. 

    zmin, zmax: float
        Grid bounds along z.

    rmin, rmax: float
        Grid bounds along r. Note: rmin should always be zero (axis).

    n_modes: integer
        Number of azimuthal fourier modes m. The code
        uses (2m +1) grids. (1 zero mode + real and imaginary
        components for higher modes).
        
    """
    def init_grid(self,nr = None, nz = None, z = None, r = None, n_modes = 1, max_iterations = 3):
        self.grid = picmi.CylindricalGrid(
            number_of_cells           = [nr, nz],
            lower_bound               = [r[0] , z[0]],
            upper_bound               = [r[1], z[1]],
            lower_boundary_conditions = ['open', 'open'],
            upper_boundary_conditions = ['open', 'open'],
            n_azimuthal_modes = n_modes,
            moving_window_velocity    = [0,cst.c])

        solver_dict = { picmi.codename + '_maximum_iterations' : max_iterations }
        self.solver = picmi.ElectromagneticSolver( grid = self.grid, **solver_dict)


    """
    Add OpenPMD Beam File

    Parameters
    ----------
    PGroup: Beam data (OpenPMD Particle Group)

    qpad_file_out: QPAD readable beam file
        Name of QPAD output file [String]

    directory: folder to save qpad_file_out
        Path to QPAD Sim Folder [String]
    
    op: function to determine beam centroid, defualt = np.median
        centroid calculator [Python function]

    z_select: selector to determine if using z-coord or t-coord, default = False
        Optional selector [Boolean]

    curr_filt: high current region [I > curr_filt] where beam is centered, default = 0
        Current threshold (Amps)

        
    """
    def add_openpmd_file_bunch(self, PGroup, qpad_file_out, op = np.median, directory = '.', z_select=False, curr_filt = 0):
        assert self.grid is not None, Exception("Must initialize grid before adding OpenPMD bunch")
        
        P = PGroup
        P = P[P.status == 1]
        q_grid_norm = (2 * np.pi * self.grid.dr**2 * self.grid.dz) 
        q_raw_norm = (cst.e * self.n0 )
        scale_q = 1.0/(q_grid_norm * q_raw_norm)
        scale_p = 1/(0.511e6)
        
        dataset = P.copy() # modify a copy of P
        self.dist = np.median(dataset.t * cst.c + dataset.z)
        if(z_select):
            self.shift_z =  op(dataset.z)
            dataset.z = (dataset.z -self.shift_z)
        else:
            dt = op(dataset.t)
            self.shift_z = self.dist - cst.c * dt
            dataset.z = cst.c * (dt - dataset.t) # calculate z and center beam at high current region
        nslices = 200
        q_ch = np.copy(dataset.weight)
        z_ch = np.copy(dataset.z)
        zlin = np.linspace(np.min(z_ch), np.max(z_ch),nslices+1)
        # print('min/max',np.min(z_ch), np.max(z_ch))
        curr = np.zeros(nslices)
        dt = (zlin[1]-zlin[0])/cst.c
        for j in range(nslices):
            z1, z2 = zlin[j], zlin[j+1]
            filt = np.logical_and(z_ch > z1, z_ch < z2)
            curr[j] = np.sum(q_ch[filt])/dt
        zlin = 0.5 * (zlin[1:] + zlin[:-1])
        filt_final = curr > curr_filt
        zmin,zmax = np.min(zlin[filt_final]), np.max(zlin[filt_final])
        
        group = dataset.where(np.logical_and(dataset.z > zmin, dataset.z < zmax))
        dispx = np.median(group.x)
        dispy = np.median(group.y) 
        
        x_prime = group.px/group.pz
        y_prime = group.py/group.pz
        driftx = np.average(x_prime)
        drifty = np.average(y_prime)

        adjustxp = driftx
        adjustyp = drifty
    
        dispx,dispy = np.mean(group.x),np.mean(group.y)

            
        self.filt_charge = np.sum(group.weight)
        
        self.shift_x = dispx 
        self.shift_y = dispy
        self.shift_xp = adjustxp
        self.shift_yp = adjustyp

        

        x, y, z = self.kp * (dataset.x-dispx), self.kp * (dataset.y-dispy), self.kp * dataset.z
        px, py, pz = dataset['px']*scale_p, dataset['py']*scale_p, dataset['pz']*scale_p
        px -= pz * adjustxp
        py -= pz * adjustyp

        # print('done filter')
        q = dataset.weight *scale_q
        hf = h5py.File(directory + '/' + qpad_file_out, 'w')
        hf.create_dataset('x1', data=x)
        hf.create_dataset('x2', data=y)
        hf.create_dataset('x3', data=z)
        hf.create_dataset('p1', data=px)
        hf.create_dataset('p2', data=py)
        hf.create_dataset('p3', data=pz)
        hf.create_dataset('q',data=-np.abs(q))
        hf.close()
        # print('saved hdf5')
        self.species_list.append(picmi.Species( particle_type = 'electron', 
            initial_distribution = picmi.OpenPMDFileDistribution(qpad_file_out)))
        self.if_beam.append(True)
        self.layouts.append(picmi.FileLayout(grid = self.grid))


    


    """
    Add tri-Gaussian electron bunch

    Parameters
    ----------
    charge: float
    Total charge [C]

    bunch_rms_size: vector of length 3 of floats
        RMS bunch size along (x,y,z) [m]

    bunch_centroid_position: vector of length 3 of floats, default = [0, 0, 0]
        Bunch centroid position (x,y,z) [m]
    
    bunch_centroid_velocity: vector of length 3 of floats, default = [0, 0, 19569.47]
        RMS velocity in units of p/mc (unitless)

    bunch_rms_velocity: vector of length 3 of floats, default = [0, 0, 0]
        RMS velocity in units of sigma_p/mc (unitless)

    ppc: list of 3 integers, default = [2, 1, 2]
        ppc along r, phi, and z

    num_theta: 
        integer, default = 8
        ppc along azimuthal direction
    
    
    """
    def add_gaussian_electron_bunch(self, charge, bunch_rms_size, 
        bunch_centroid_position = [0, 0 ,0], bunch_centroid_velocity = [0, 0, 19569.47],
         bunch_rms_velocity = [0, 0 ,0], ppc = [2, 1, 2], num_theta = 8):

        n_physical_particles = abs(int(charge/cst.e))

        dist = picmi.GaussianBunchDistribution(
            n_physical_particles = n_physical_particles,
            rms_bunch_size       = bunch_rms_size,
            rms_velocity         = [cst.c * x for x in bunch_rms_velocity],
            centroid_position    = bunch_centroid_position,
            centroid_velocity    = [cst.c * x for x in bunch_centroid_velocity] )
        
        self.species_list.append(picmi.Species( particle_type = 'electron', initial_distribution = dist))
        layout_dict = { picmi.codename + '_num_theta' : num_theta }
        self.layouts.append(picmi.GriddedLayout(
                    grid = self.grid,
                    n_macroparticle_per_cell = ppc, 
                    **layout_dict))
        self.if_beam.append(True)


    """
    Add uniform pre-ionized plasma

    Parameters
    ----------
    number_density: float
    Plasma electron number density [m^-3]

    ppc: list of 2 integers, default = [4, 1]
        ppc along r and phi

    num_theta: 
        integer, default = 8
        ppc along azimuthal direction
    """
    def add_uniform_plasma(self, number_density = 0, ppc = [4, 1], num_theta = 8):
        if(self.grid is None):
            print("Warning: Initialize grid before adding plasma")
            return
        self.species_list.append(picmi.Species(particle_type = 'electron', 
            initial_distribution = picmi.UniformDistribution(density = number_density) ))

        layout_dict = { picmi.codename + '_num_theta' : num_theta }
        self.layouts.append(picmi.GriddedLayout(
                    grid = self.grid,
                    n_macroparticle_per_cell = ppc, 
                    **layout_dict))
        self.if_beam.append(False)


    """
    Add longitudinal plasma profile

    Parameters
    ----------
    z: array of floats
        Longitudinal position of neutral gas profile [m].

    nz: array of floats
        Number density of neutral gas profile [m^-3].

    n0_factor: float
        Normalizing density factor [m^-3].

    ppc: list of 2 integers, default = [4, 1]
        ppc along r and phi. ppc(1) * ppc(2) = total macroelectrons 
        per cell (default is 4)

    num_theta: 
        integer, default = 8
        ppc along azimuthal direction
        
    """
    def add_longitudinal_plasma_profile(self, z, nz, ppc = [4, 1], num_theta = 8):
        if(self.grid is None):
            print("Warning: Initialize grid before adding neutral gas")
            return
        assert self.grid is not None, Exception("Must initialize grid before adding Plasma")

        self.species_list.append(picmi.Species(particle_type = 'electron', 
            initial_distribution = picmi.PiecewiseDistribution(density = self.n0, piecewise_s = z, piecewise_fs = nz)))

        layout_dict = { picmi.codename + '_num_theta' : num_theta }
        self.layouts.append(picmi.GriddedLayout(
                    grid = self.grid,
                    n_macroparticle_per_cell = ppc, 
                    **layout_dict))
        self.if_beam.append(False)



    """
    Add uniform neutral gas (e.g. Li)

    Parameters
    ----------
    number_density: float
        Number density of gas [m^-3].

    particle_type: string
        A string specifying an atom (e.g. Li, Ar...) as defined in
        the openPMD 2 species type extension, openPMD-standard/EXT_SpeciesType.md

    max_level: integer, optional 
        Specifies maximum ionization level.

    ppc: list of 2 integers, default = [4, 4]
        ppc along r and phi. ppc(1) * ppc(2) = total ionized macroelectrons 
        per cell (default is 16)

    num_theta: 
        integer, default = 8
        ppc along azimuthal direction
        
    """
    def add_uniform_neutral_gas(self, number_density = 0, particle_type = 'Li', max_level = None, ppc = [4, 4], num_theta = 8):
        if(self.grid is None):
            print("Warning: Initialize grid before adding neutral gas")
            return
        assert self.grid is not None, Exception("Must initialize grid before adding Plasma")
        if(max_level is not None):
            neut_dict = { picmi.codename + '_ion_max' : max_level }
        else:
            neut_dict = {}

        self.species_list.append(picmi.Neutral(particle_type = particle_type, 
            initial_distribution = picmi.UniformDistribution(density = number_density), 
            **neut_dict ))

        layout_dict = { picmi.codename + '_num_theta' : num_theta }
        self.layouts.append(picmi.GriddedLayout(
                    grid = self.grid,
                    n_macroparticle_per_cell = ppc, 
                    **layout_dict))
        self.if_beam.append(False)



    """
    Add longitudinal neutral gas (e.g. Li)

    Parameters
    ----------
    z: array of floats
        Longitudinal position of neutral gas profile [m].

    nz: array of floats
        Number density of neutral gas profile [m^-3].

    n0_factor: float
        Normalizing density factor [m^-3].

    particle_type: string
        A string specifying an atom (e.g. Li, Ar...) as defined in
        the openPMD 2 species type extension, openPMD-standard/EXT_SpeciesType.md

    max_level: integer, optional 
        Specifies maximum ionization level.

    ppc: list of 2 integers, default = [4, 4]
        ppc along r and phi. ppc(1) * ppc(2) = total ionized macroelectrons 
        per cell (default is 16)

    num_theta: 
        integer, default = 8
        ppc along azimuthal direction
        
    """
    def add_longitudinal_neutral_gas_profile(self, z, nz,  particle_type = 'Li', max_level = None, ppc = [4, 4], num_theta = 8):
        if(self.grid is None):
            print("Warning: Initialize grid before adding neutral gas")
            return
        assert self.grid is not None, Exception("Must initialize grid before adding Plasma")
        if(max_level is not None):
            neut_dict = { picmi.codename + '_ion_max' : max_level }
        else:
            neut_dict = {}

        self.species_list.append(picmi.Neutral(particle_type = particle_type, 
            initial_distribution = picmi.PiecewiseDistribution(density = self.n0, piecewise_s = z, piecewise_fs = nz), 
            **neut_dict ))

        layout_dict = { picmi.codename + '_num_theta' : num_theta }
        self.layouts.append(picmi.GriddedLayout(
                    grid = self.grid,
                    n_macroparticle_per_cell = ppc, 
                    **layout_dict))
        self.if_beam.append(False)







    """
    Adds Raw Particle Diagnostic for beam dumps

    Parameters
    ----------

    period: integer, default = 1
        Frequency of data dumps (1 dumps every timestep)

    period: integer, default = 1
        Sampling frequency of particles (1 dumps every particle, 2 dumps every other part)
        
    """
    def add_particle_diagnostics(self, period = 1, psample = 1):
        part_diag_dict = { picmi.codename + '_sample' : 1}
        beam_list = []
        for i in range(len(self.species_list)):
            if(self.if_beam[i]):
                beam_list.append(self.species_list[i])
        self.part_diags.append(picmi.ParticleDiagnostic(period = period,
                             species = beam_list,
                              **part_diag_dict))

    """
    Adds Field Diagnostic to data dumps

    Parameters
    ----------
    data_list: list of strings
        Field Data to dump (e.g. ['Er', 'Ephi', 'Ez', 'Br', 'Bphi', 'Bz', 'psi', 'rho'])

    period: integer, default = 1
        Frequency of data dumps (1 dumps every timestep)

    """
    def add_field_diagnostics(self, data_list = [], period = 1):
        self.field_diags.append(picmi.FieldDiagnostic(data_list = data_list,
                                       grid = self.grid,
                                       period = period))


    """
    Constructs simulation input file and runs QPADs

    Parameters
    ----------
    dt: float
        Time step of simulation [s].

    tmax: float
        Maximum time of simulation [s].

    nodes: list of 2 integers, default = [1, 1]
        mpi procs along r and z

    report_timings: boolean, default = False
        Flag to dump simulation timings  
        
    """
    def run_simulation(self,dt, tmax, nodes = [1, 1], sim_dir = '.', report_timings = False):

        sim_dict = { picmi.codename + '_nodes' : nodes, 
                    picmi.codename + '_n0' : self.n0,
                    picmi.codename + '_timings' : report_timings}

        self.simulation = picmi.Simulation(solver = self.solver, verbose = 1,
            time_step_size = dt, max_time = tmax, **sim_dict)

        for i in range(len(self.species_list)):
            self.simulation.add_species(species = self.species_list[i], layout = self.layouts[i])

        for i in range(len(self.field_diags)):
            self.simulation.add_diagnostic(self.field_diags[i])

        for i in range(len(self.part_diags)):
            self.simulation.add_diagnostic(self.part_diags[i])
        
        self.simulation.write_input_file(sim_dir+ '/qpinput.json')
        env = dict(os.environ)
        if('CONDA_PREFIX' in env):
            env['PATH'] =env['CONDA_PREFIX'] + '/bin:'  + env['PATH']
        else:
            env['PATH' ] = os.path.dirname(sys.executable) + ":" + env['PATH']

        procs = np.prod(nodes)
        cmd = ["mpirun", "--oversubscribe", "-np", str(procs), "qpad-pmd.e"]
        # cmd = ["mpirun", "-np", str(procs), "qpad-pmd.e"]

        # using_slurm = bool(os.environ.get("SLURM_JOB_ID"))
        # if using_slurm:
        #     slurm_ntasks = int(os.environ.get("SLURM_NTASKS",str(1)))
        #     if(slurm_ntasks < procs):
        #         print(f"Warning: QPAD MPI processes={procs} are oversubcribed above SLURM_NTASKS={slurm_ntasks} set by environment...")
        #         cmd = ["mpirun", "--oversubscribe", "-np", str(procs), "qpad-pmd.e"]

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            text=True,
            cwd = sim_dir, 
            env=env
        )

        for line in proc.stdout:
            # Remove newline
            line = line.rstrip()
            # Print, overwriting the same line
            sys.stdout.write("\r" + line + " " * 30)
            sys.stdout.flush()

        proc.wait()
        print()  # final newline

    """
    Returns Final ParticleGroup after QPAD Sim 
    Also corrects <x,x'> phase space due to centering
    
    Parameters
    ----------
    simPath: Path to QPAD Simulation (string)

    time_step: Simulation time step of beam dump (integer) 
        
    """
    def getBeamFromQPAD(self, simPath, time_step, tToZ = True):
        eleString = f'{simPath}/Beam1/Raw/raw_' + str(100000000 + time_step)[1:] + '.h5'
        P = ParticleGroup(eleString)
        P = P[P.status == 1]

        if tToZ:
            P.z = -cst.c* P["delta_t"]
            #P.t = 0 * P.t #I haven't decided the best practice for this yet. Technically the beam is not self-consistent without t being set to zero but not doing so is convenient for backwards compatibility
            
        sim_time = self.simulation.time_step_size * time_step
        # account for drifts in beam xprime and yprime
        dx = self.shift_xp * (sim_time/self.kp) + self.shift_x
        dy = self.shift_yp * (sim_time/self.kp) + self.shift_y
        P.x += dx
        P.y += dy
        # time add simulation runtime and linac time (P.t is really just delta_t)
        P.t = P.t  + self.dist/cst.c + sim_time/self.wp

        # shift transverse momenta by initial drifts (shift_xp, shift_yp)
        # assuming vx/c, vy/c << 1
        gamma = np.sqrt(1 + (P['px']/0.511e6)**2 + (P['py']/0.511e6)**2 + (P['pz']/0.511e6)**2)
        px_new = P['px'] +  gamma * self.shift_xp * 0.511e6
        py_new = P['py'] + gamma * self.shift_yp * 0.511e6
        P.px = px_new
        P.py = py_new
        
        return P
        



""" 
Generates the plasma density of the lithium oven/helium as a function of z position.

The position and density at each position is returned as output in the following order:

[z array, Lithium density array, Helium density array]

Args:
    Nz: Number of positions in z.
    Z: [m] Maximum z position to generate, inclusive.
    P: [torr] Buffer gas pressure.
    T_bkgd: [K] Temperature of the background He buffer gas.
    l_He: [m] Length of He density to use from thermodynamics calulation. Interpolation
        is used outside of this region.
    filename_Li: Filename to output the Li density to.
    filename_He: Filename to output the He density to.
"""
def generate_Li_oven_profile(Nz = 1001, Z = 0.6, P = 5.0, T_bkgd = 273.15, l_He = 0.44 ):

    # Calculation variables
    z = np.linspace(0.0, Z, Nz)
    n = np.zeros(Nz, dtype="double")
    center = Z / 2
    kB = physical_constants["Boltzmann constant"][0]


    # Lithium properties
    def Pv(T):
        """Calculates the vapor pressure of the lithium gas as a function of temperature.

        Args:
            T: [K] temperature of the lithium gas.

        Returns:
            Pv: [torr] vapor pressure of the lithium.
        """
        T = T * 1e-3
        return np.exp(-2.0532 * np.log(T) - 19.4268 / T + 9.4993 + 0.753 * T) / 133.0e-6


    def f(T):
        return Pv(T) - P


    T = fsolve(f, 1000.0)[0]
    ne = 9.66e24 * P / T

    # Background lithium density - necessary for find He density
    Pv_bkgd = Pv(T_bkgd)
    n_bkgd = 9.66e24 * Pv_bkgd / T_bkgd

    # Uniform accelerating plasma
    length = 238e-3
    z_start = center - 0.5 * length
    z_end = center + 0.5 * length
    sel = (z > z_start) * (z < z_end)
    n[sel] = 1.0

    # Entrance ramp - error function
    ent_start = center - 400.0e-3
    s_ent = 22.0e-3
    sel = (z >= ent_start) * (z <= z_start)
    n[sel] = 0.5 * (1 + erf((z[sel] - z_start + 100.0e-3) / (np.sqrt(2) * s_ent)))
    n[sel] *= 1.0 / n[sel][-1]  # Make sure curve is continuous

    # Exit ramp - error function
    exit_end = center + 400.0e-3
    s_ext = 22.0e-3
    sel = (z >= z_end) * (z <= exit_end)
    n[sel] = 0.5 * (1 + erf(-(z[sel] - z_end - 100.0e-3) / (np.sqrt(2) * s_ext)))
    n[sel] *= 1.0 / n[sel][0]  # Make sure curve is continuous

    n *= ne
    n += n_bkgd

    # Save the Li plasma density file
    data = np.stack((z, n), axis=1)

    # Calculate He plasma density
    # First create interpolations to go from density to temperature and Li pressure
    T_int = np.linspace(200, 1200, 1001)
    P_int = Pv(T_int)
    n_int = 9.66e24 * P_int / T_int
    T_from_n = interp1d(n_int, T_int)
    P_from_n = interp1d(n_int, P_int)

    # Find the temperature and Li pressure along the oven, then calculate He density
    T_n = T_from_n(n)
    P_n = P_from_n(n)
    n_He = ((P - P_n) * 133.32236842) / (kB * T_n)
    n_He_bkgd = ((P) * 133.32236842) / (kB * T_bkgd)

    # Above meathod breaks down at low Li pressure, use linear interpolation from the ramps
    z_HeStart = center - 0.5 * l_He
    z_HeEnd = center + 0.5 * l_He
    nHe = np.zeros(Nz)
    sel = (z > z_HeStart) * (z < z_HeEnd)
    nHe[sel] = n_He[sel]

    # Extend linearly from the ends
    slope = (nHe[sel][10] - nHe[sel][0]) / (z[sel][10] - z[sel][0])
    selUp = z <= z_HeStart
    nHe[selUp] = slope * (z[selUp] - z[sel][0]) + nHe[sel][0]
    selDown = z >= z_HeEnd
    nHe[selDown] = -slope * (z[selDown] - z[sel][-1]) + nHe[sel][-1]

    # Set to background density
    sel = nHe > n_He_bkgd
    nHe[sel] = n_He_bkgd

    data = np.stack((z, nHe), axis=1)

    return [z, n, nHe]

# aux functions for dictionary lookup
def eq(s, t):
    return s.lower() == t.lower()

def get(table, key):
    try:
        val = table[key]
        return val
    except:
        print(f"Key {str(key)} not found in QPAD settings file")
        return None

def present(table, key):
    return key in table


    
""" 
Runs QPAD simulation with settings specified in defaultsFile

Args:
    tao: Pytao instance
    defaultsFile: Path to yml specifying default parameters
"""
def run_QPAD(tao,
    PGroup,
    defaultsFile=None,
    verbose = False,
    **overrides):

    # read in QPAD settings from file
    if not defaultsFile:
        defaultsFile = f'{tao.filePathGlobal}/qpad/2025-08-20-QPAD_defaults.yml'
        if verbose:
            print(f"No defaults file provided to setLattice(). Using {defaultsFile}")
        
    with open(defaultsFile, 'r') as file:
        defaults = yaml.safe_load(file)
    # print(defaults)

    try: 
        sim_settings = get(defaults, 'simulation')
        grid_settings = get(sim_settings,'grid')
        diag_settings = get(defaults, 'diagnostics')
        plasma_settings = get(defaults, 'plasma')

    except:
        print(f"Missing sections in {defaultsFile}!")




    # read in plasma config
    if(eq(plasma_settings['config'], 'oven')):
        z, nLi, nHe = generate_Li_oven_profile(P = plasma_settings['P_torr'])
        n0 = np.max(nLi)
        zsim = z[-1] - 1e-6


    # initialize QPAD simulation object
    sim = QPAD_sim(n0)
    kp, wp = sim.kp, sim.wp
    sim.init_grid(nr = get(grid_settings, 'r_cells'), 
                    nz = get(grid_settings, 'z_cells'), 
                    r = get(grid_settings, 'r'), 
                    z = get(grid_settings, 'z'),
                    n_modes = get(grid_settings, 'max_mode'))


    # reads patchFile, centers <x> and <x'> and exports to QPAD-formatted file 'qpad_file.h5' in 'directory'
    sim.add_openpmd_file_bunch(PGroup,
        'qpad_file.h5', 
        directory = tao.qpadSimPath, 
        z_select = False,
        curr_filt = 1e3)
    
    # Set up plasma source 
    if(eq(plasma_settings['config'], 'oven')):
        num_theta = 8 * max(1, get(grid_settings, 'max_mode'))
        if(not get(plasma_settings, 'preionized')):
            sim.add_longitudinal_neutral_gas_profile(z,  nLi,
             particle_type = 'Li', max_level = 1, num_theta = num_theta ) # model first level of Lithium

            # sim.add_longitudinal_neutral_gas_profile(z, np.abs(nHe),
            #  particle_type = 'He', max_level = 1, num_theta = num_theta ) # model first level of Helium
        else:
            sim.add_longitudinal_plasma_profile(z,  nLi, num_theta = num_theta ) # pre-ionized plasma
    

    
    dt_qpad = 20
    final_timestep = int(kp * zsim/dt_qpad)

    # add diagnostics
    ndumps = get(diag_settings, 'ndumps') or 1
    field_quants = get(diag_settings,'field_quants') or ['rho', 'Ez']

    # Add diagnostics 
    if(get(diag_settings, 'save_fields')):
        sim.add_field_diagnostics(data_list = field_quants, period =int(final_timestep/ndumps)) # 
    sim.add_particle_diagnostics(period = int(final_timestep/ndumps))
    
        
    
    # Run QPAD Simulation
    print(f"Running QPAD Simulation in {tao.qpadSimPath}")
    sim.run_simulation(dt = dt_qpad/wp, 
                        tmax = zsim/cst.c, 
                        nodes = get(sim_settings, 'nprocs'),
                        sim_dir = tao.qpadSimPath,
                        report_timings = get(sim_settings, 'if_timing'))

    # Export output Beam ParticleGroup
    P = sim.getBeamFromQPAD(tao.qpadSimPath, int(final_timestep/ndumps) * ndumps, tToZ = False)
    zbeam =  int(final_timestep/ndumps) * ndumps * (dt_qpad/kp)
    return P, zbeam

  


""" 
Auxiliary routines for visualizing QPAD simulation

"""  



""" 
Opens Field data located at path using OpenPMD backend libs. Combines modes into single mesh at angle phi.

Args:
    path: String
        Path to Mesh data

    phi: Float
        Angle at which modes are evaluated and combined.

    max_mode: Integer
        number of modes summed. If None, sums over all modes. 

    axis_norm: Float
        normalizing constant for r=0 cells for certain meshes.
"""
def combined_dset_mod(path, iteration, phi = 0, max_mode = None, axis_norm = 1):
    ts_dset = OpenPMDTimeSeries(f'{path}/Re0/')
    dset, dset_info = ts_dset.get_field( iteration=iteration,  field = ts_dset.avail_fields[0], plot=False )
    dset = dset.T
    dset_size = dset.shape
    dset[:,0] *= axis_norm
    merged_dset = np.zeros((dset_size[0], dset_size[1] * 2 - 1))
    merged_dset[:, 0:dset_size[1] - 1] += np.fliplr(dset[:, 1:])
    merged_dset[:, dset_size[1] - 1:] += dset
    
    max_mode = max_mode or int((sum(1 for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))) -1)/2)
    # add m > 0 modes to the merged dataset
    for mode in np.arange(1, max_mode + 1):
        ts_dset = OpenPMDTimeSeries(f'{path}/Re{mode}/')
        dset, dset_info = ts_dset.get_field( iteration=iteration,  field = ts_dset.avail_fields[0], plot=False )
        dset = dset.T
        merged_dset[:, 0:dset_size[1] - 1] += np.fliplr(dset[:, 1:]) * 2 * np.cos(mode * (phi + np.pi))
        merged_dset[:, dset_size[1] - 1:] += dset * 2 * np.cos(mode * phi)

        ts_dset = OpenPMDTimeSeries(f'{path}/Im{mode}/')
        dset, dset_info = ts_dset.get_field( iteration=iteration,  field = ts_dset.avail_fields[0], plot=False )
        dset = dset.T
        merged_dset[:, 0:dset_size[1] - 1] -= np.fliplr(dset[:, 1:]) * 2 * np.sin(mode * (phi + np.pi))
        merged_dset[:, dset_size[1] - 1:] -= dset * 2 * np.sin(mode * phi)
    return merged_dset.T, dset_info



""" 
Returns a matplotlib figure of QPAD Simulation data

Args:
    sim_fold: String
        Path to QPAD simulation folder

    ndump: Integer
        The dump number to visualize. ndump = 0,1,2 ... corresponds to the first, second, third data dump

    quants: List of Strings
        Data to plot. Valid inputs are ['raw', 'rho', 'rho_beam','rho_ion', 'rho_plasma', 'ez', 'bphi' ...]
        'raw' is the 6D phase space of the beam. Everything else is mesh data. 

    plot_type: List of Strings
        Specifies plot type for each quantity. For 'raw', string needs to be two phase space quantities (e.g. 'z,pz')
        'imshow' plots 2d mesh data, 'slice,z,50e-6' plots a 1d slice at z=50e-6.

    xlims: List of float tuples
        Specifies xlims of each subplot. If [None, None], lets matplotlib automatically determine bounds.

    ylims: List of float tuples
        Specifies ylims of each subplot. If [None, None], lets matplotlib automatically determine bounds.

    vlims: List of float tuples
        Specifies vlims of each subplot. If [None, None], lets matplotlib automatically determine bounds.
        Only used for imshow.

    angle: Float
        Azimuthal angle where modes are evaluated and summed.
        E.g., angle = 0 (90) plots the x (y) plane.
    
    ncols: Integer
        Number of columns for matplotlib plot

    figsize: Float tuple
        Size of figure inches.

    cmaps: List of Strings 
        Specifies colormap for imshow/raw plots. Ignored for all other plot types.


"""
def plotQPAD(sim_fold = '',
             ndump = 1,
             quants = ['rho', 'ez'], 
             plot_type = ['imshow','imshow'],
             xlims = [[None,None], [None,None]],
             ylims = [[None,None], [None,None]],
             vlims = [[0,2], [None,None]],
             angle = 0, 
             ncols = 1,
             figsize = (10,7),
             cmaps = ['Blues', 'jet'] ):
    
    
    
    set_matplotlib_formats('retina')

    matplotlib.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 300})
    # read in parameters from qpad simulation
    file =  open(f'{sim_fold}/qpinput.json', 'r')
    sim_params = json.load(file)
    file.close()


    
    # determine reference wp, kp
    n0, dt = sim_params['simulation']['n0'],sim_params['simulation']['dt'] 
    wp = np.sqrt(cst.e**2 * n0*1e6/(cst.epsilon_0 * cst.m_e))
    kp = wp/cst.c

    # calculate normalizing constant for r=0 cells 
    if('neutrals' in sim_params):
        n = np.prod(sim_params['neutrals'][0]['ppc'])
    else:
        n = (sim_params['species'][0])['ppc'][0]
    axis_charge_norm = 3 * n**2/(2 * (1 + 2 * n**2))

    # using openPMD libs to store relevant simulation data/metadata/folders
    ts_plot_array = []
    fold_plot_array = []
    for quant in quants:
        if(quant.lower() == 'raw'):
            ts_plot_array.append([])
            fold_plot_array.append([])
            continue
        key = 'Charge' if quant[:3].lower() == 'rho' else quant 
        key = 'Ion_charge' if 'ion' in quant.lower() else key
        key = 'Ez' if 'ez' in quant.lower() else key 
        if('beam' in quant.lower()):
            matches = glob.glob(f"{sim_fold}/B*/{key}", recursive=True)
        elif('plasma' in quant.lower()):
            matches = glob.glob(f"{sim_fold}/[!B]*/{key}", recursive=True)
        else:
            matches = glob.glob(f"{sim_fold}/**/{key}", recursive=True)
    
        matches = np.sort([m for m in matches if os.path.isdir(m)])[::-1]
        ts_plot = []
        fold_plot = []
        for i in matches:
            ts_plot.append(OpenPMDTimeSeries(f'{i}/Re0/'))
            fold_plot.append(f'{i}/')
        ts_plot_array.append(ts_plot)
        fold_plot_array.append(fold_plot)



    # construct the figure
    # close any open figures
    plt.close('all')
    plt.ioff()

    fig = plt.figure(ndump, figsize=figsize, constrained_layout=True)

    # define tickformatter
    tick_formatter = FuncFormatter(lambda val, pos: f'{val*1e6:0.0f}e-6')

    nrows = int(len(quants)/ncols) + int(len(quants) % ncols != 0)


    for j in range(len(quants)):
        plot_idx = (j%nrows) * ncols + int(j/(nrows))
        ax = plt.subplot(nrows,ncols,plot_idx+1)

        ##### plotting beam phase-space #####
        if(quants[j].lower() == 'raw'):
            matches = glob.glob(f"{sim_fold}/**/Raw/**", recursive=True)
            matches = np.sort([m for m in matches if not os.path.isdir(m)])
            P = ParticleGroup(matches[ndump])
            P.z = -P['t'] * cst.c + dt/kp * int(matches[ndump][-11:-3])
            keys = plot_type[j].split(',')
            keys = [r.strip() for r in keys]
            data1, data2 = P[keys[0]],P[keys[1]]
            if keys[0].lower()not in ['z','r','x','y']:
                data1 *= 1e-9
            if keys[1].lower() not in ['z','r','x','y']:
                data2 *= 1e-9
            H, xedges, yedges = np.histogram2d(data1, data2, bins=1000)
            plot_temp1 = ax.imshow(H.T, interpolation='bilinear', origin='lower',
                extent=[xedges[0], xedges[-1], yedges[0], yedges[-1]], aspect = 'auto', cmap = 'jet',
                norm = matplotlib.colors.SymLogNorm(np.median(H[H>0])))

            if(keys[0].lower() == 'z'):

                ax.set_xlim(xedges[-1], xedges[0])
                ax.set_xlim(xlims[j][-1],xlims[j][0])

            # format x-axis
            if keys[0].lower() in ['z','r','x','y']:
                ax.set_xlabel(r'$' + keys[0] + r'\ (m)$')
                ax.get_xaxis().set_major_formatter( tick_formatter )
            else:
                ax.set_xlabel(r'$' + keys[0] + r' \ (GeV/c)$')

            # format yaxis
            if keys[1].lower() in ['z','r','x','y']:
                ax.set_ylabel(r'$' + keys[1] + r' \ (m)$')
                ax.get_yaxis().set_major_formatter( tick_formatter )
            else:
                ax.set_ylabel(r'$' + keys[1] + r' \ (GeV/c)$')            
            cb1 = fig.colorbar(plot_temp1, ax = ax, pad = 0.05)
            cb1.set_label(r'$ charge \ [a.u.]$')

            # skip rest of loop
            continue
        


        ##### otherwise plotting mesh data #####

        # get list of field mesh data for current subplot
        ts_temp = ts_plot_array[j]

        # get list of folder locations
        fold_temp = fold_plot_array[j]

        # reference first field data
        ts = ts_temp[0]
        iterations = ts.iterations
        tprop = ts.t[ndump]
        field = ts.avail_fields[0]

        # reads in all mesh data into F
        # if plotting charge density, combines multiple beam and plasma mesh data.
        F, info = ts.get_field( iteration=iterations[ndump],  field = ts.avail_fields[0], plot=False )
        axis_norm = axis_charge_norm if 'rho' in quants[j].lower() else 1
        F, info = combined_dset_mod(fold_temp[0], iterations[ndump],phi = angle, axis_norm = axis_norm)
        for k in range(1,len(ts_temp)):
            F_t, info_t = combined_dset_mod(fold_temp[k], iterations[ndump],phi= angle, axis_norm = axis_norm)
            F += F_t

        # units and normalizing constants for e, b, and rho.
        field_units_raw = {'e' : 'GeV/m', 
                           'r':  rf'{n0:.1e} \ cm^{{-3}}'.replace("+",""),
                           'b' : 'T'} 
        mul_norm = { 'e' : cst.m_e * cst.c**2/cst.e * kp *1e-9, 
                     'r' : 1,
                     'b' :cst.m_e * cst.c/cst.e * kp}

        F = F * mul_norm[quants[j].lower()[0]]
        field_units = field_units_raw[quants[j].lower()[0]]
        field_label = r'$' + quants[j] 
        if('rho' in quants[j].lower()):
            F = np.abs(F)
            if('beam' in quants[j].lower()):
                field_label = r'$n_b' 
            elif('ion' in quants[j].lower()):
                field_label = r'$n_i'
            elif('plasma' in quants[j].lower()):
                field_label = r'$n_p' 
            else:
                field_label = r'$n_e' 
                
        quant_label = field_label + r' \ [' + field_units + ']$'
            
        # plot title: propagation distance and angle of visualization
        zprop = cst.c * tprop 
        title = r'$Z_{sim}$' + rf'= {zprop:.3f} m  ($\phi = {angle}^\circ)$'
        plt.suptitle(title)

        axes_labels = [r'$ z \ (m)$' if info.axes[x] =='\\xi' else r'$'+ info.axes[x] + r'\ (m)$' for x in info.axes]
        extent = info.imshow_extent
        extent[0],extent[1] = -extent[0],-extent[1]


        if(plot_type[j] == 'imshow'): # plotting 2D mesh data
            extent[2] = -extent[3]
            plot_temp = ax.imshow(F, extent=extent, origin='lower',
                       interpolation='bilinear', aspect='auto', vmin = vlims[j][0], vmax = vlims[j][1], cmap = cmaps[j])
            ax.get_yaxis().set_major_formatter( tick_formatter )
            cb1 = fig.colorbar(plot_temp, ax = ax)
            cb1.set_label(quant_label)
            ax.set_ylabel(axes_labels[0])
            ax.set_xlabel(axes_labels[1])
            ax.set_xlim(xlims[j])
            ax.set_ylim(ylims[j])
        else:                          # plotting 1D slice data
            keys = plot_type[j].split(',')
            keys = [r.strip() for r in keys]
            label = keys[1] + '=' + keys[2]
            if('r' in keys[1]):
                z= vars(info)[info.axes[1]]
                r = np.linspace(-extent[3],extent[3], F.shape[0])
                ind = np.argmax(r > float(keys[2]))
                lineout = F[ind,:]
                ax.plot(-z,lineout, label = label)
                ax.set_xlabel(axes_labels[1])
                ax.set_xlim(extent[0],extent[1])
            else:
                z= vars(info)[info.axes[1]]
                r = np.linspace(-extent[3],extent[3], F.shape[0])
                ind = np.argmax(-z < float(keys[2]))
                lineout = F[:,ind]
                ax.plot(r,lineout, label = label)
                ax.set_xlabel(axes_labels[0])
            ax.set_xlim(xlims[j])
            ax.set_ylim(ylims[j])
            ax.legend()
            ax.set_ylabel(quant_label)
        
        ax.get_xaxis().set_major_formatter( tick_formatter )
    return fig


""" 
Routine that saves matplotlib figures of quants for all possible QPAD data dumps. 
Also generates an mp4 movie of the results using FFMPEG. 

Args:
    sim_fold: String
        Path to QPAD simulation folder

    ndump: Integer
        The dump number to visualize. ndump = 0,1,2 ... corresponds to the first, second, third data dump

    quants: List of Strings
        Data to plot. Valid inputs are ['raw', 'rho', 'rho_beam','rho_ion', 'rho_plasma', 'ez', 'bphi' ...]
        'raw' is the 6D phase space of the beam. Everything else is mesh data. 

    plot_type: List of Strings
        Specifies plot type for each quantity. For 'raw', string needs to be two phase space quantities (e.g. 'z,pz')
        'imshow' plots 2d mesh data, 'slice,z,50e-6' plots a 1d slice at z=50e-6.

    xlims: List of float tuples
        Specifies xlims of each subplot. If [None, None], lets matplotlib automatically determine bounds.

    ylims: List of float tuples
        Specifies ylims of each subplot. If [None, None], lets matplotlib automatically determine bounds.

    vlims: List of float tuples
        Specifies vlims of each subplot. If [None, None], lets matplotlib automatically determine bounds.
        Only used for imshow.

    angle: Float
        Azimuthal angle where modes are evaluated and summed.
        E.g., angle = 0 (90) plots the x (y) plane.
    
    ncols: Integer
        Number of columns for matplotlib plot

    figsize: Float tuple
        Size of figure inches.

    cmaps: List of Strings 
        Specifies colormap for imshow/raw plots. Ignored for all other plot types.

    save_fold: String
        Path to save all figures. If save_fold does not exist, it is created.


"""
def saveAllQPADFigures(sim_fold = '',
         quants = ['rho', 'ez'], 
         plot_type = ['imshow','imshow'],
         xlims = [[None,None], [None,None]],
         ylims = [[None,None], [None,None]],
         vlims = [[0,2], [None,None]],
         angle = 0, 
         ncols = 1,
         figsize = (10,7),
         cmaps = ['Blues', 'jet'],
         save_fold = './'):

    
    fold = f'{sim_fold}/Beam1/Raw/'
    files = [f for f in os.listdir(fold) if os.path.isfile(os.path.join(fold, f))]

    os.makedirs(save_fold, exist_ok=True)
    plt.ioff()
    for ndump in range(len(files)):
        plt.clf()
        fig = plotQPAD(sim_fold, ndump, quants, plot_type, xlims, ylims, vlims, angle, ncols, figsize, cmaps)
        filename = f"{ndump:06d}.png"
        filepath = os.path.join(save_fold, filename)
        fig.savefig(filepath, format = 'png', dpi = 300)
        plt.close(fig)
    print('Saving all QPAD Figures to ' + save_fold)
    proc = subprocess.Popen(
        ['ffmpeg', "-framerate", str(2), "-y", "-i", "%06d.png", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "output.mp4"],
        cwd = save_fold,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT 
    )

    
""" 
Returns an interactive ui for visualizing QPAD simulation data.

Args:
    sim_fold: String
        Path to QPAD simulation folder

    ndump: Integer
        The dump number to visualize. ndump = 0,1,2 ... corresponds to the first, second, third data dump

    quants: List of Strings
        Data to plot. Valid inputs are ['raw', 'rho', 'rho_beam','rho_ion', 'rho_plasma', 'ez', 'bphi' ...]
        'raw' is the 6D phase space of the beam. Everything else is mesh data. 

    plot_type: List of Strings
        Specifies plot type for each quantity. For 'raw', string needs to be two phase space quantities (e.g. 'z,pz')
        'imshow' plots 2d mesh data, 'slice,z,50e-6' plots a 1d slice at z=50e-6.

    xlims: List of float tuples
        Specifies xlims of each subplot. If [None, None], lets matplotlib automatically determine bounds.

    ylims: List of float tuples
        Specifies ylims of each subplot. If [None, None], lets matplotlib automatically determine bounds.

    vlims: List of float tuples
        Specifies vlims of each subplot. If [None, None], lets matplotlib automatically determine bounds.
        Only used for imshow.

    angle: Float
        Azimuthal angle where modes are evaluated and summed.
        E.g., angle = 0 (90) plots the x (y) plane.
    
    ncols: Integer
        Number of columns for matplotlib plot

    figsize: Float tuple
        Size of figure inches.

    cmaps: List of Strings 
        Specifies colormap for imshow/raw plots. Ignored for all other plot types.
        
    save_fold: String
        Path to save all figures. If save_fold does not exist, it is created. Can be modified in widgets.


"""
def plotInteractiveQPADFigure(sim_fold = '',
         quants = ['rho', 'ez'], 
         plot_type = ['imshow','imshow'],
         xlims = [[None,None], [None,None]],
         ylims = [[None,None], [None,None]],
         vlims = [[0,2], [None,None]],
         angle = 0, 
         ncols = 1,
         figsize = (10,7),
         cmaps = ['Blues', 'jet'],
         save_fold = './' ):
    
    fold = f'{sim_fold}/Beam1/Raw/'
    files = [f for f in os.listdir(fold) if os.path.isfile(os.path.join(fold, f))]
    slider    = IntSlider(value=int(len(files)/2), min=0,
                        max=len(files)-1, step=1, description="ndumps", continuous_update=True)

    btn_minus = Button(description='–', layout={'width': '40px'})
    btn_plus  = Button(description='+', layout={'width': '40px'})

    def decrement(_):
        if slider.value > slider.min:
            slider.value -= 1

    def increment(_):
        if slider.value < slider.max:
            slider.value += 1

    _current = {'fig': None}
    btn_minus.on_click(decrement)
    btn_plus.on_click(increment)
    save_btn = Button(description="save", tooltip="Save plot")
    out     = Output()

    def update(*_):               
        with out:
            out.clear_output(wait=True)
            fig = plotQPAD(sim_fold, slider.value, quants, plot_type, xlims, ylims, vlims, angle, ncols, figsize, cmaps)
            _current['fig'] = fig
            display(fig)

    folder_input = Text(
    value=save_fold,   # default current directory
    placeholder="Enter folder path",
    description="Save Folder:",
    layout={"width": "400px"})

    def _on_save(_):
        fig = _current.get('fig', None)
        if fig is None:
            print("No figure available to save yet.")
            return
            
        folder = folder_input.value.strip()
        if not folder:
            folder = "./"  # default to current dir
        os.makedirs(folder, exist_ok=True)
        
        filename = f"{slider.value:06d}.png"
        filepath = os.path.join(folder, filename)

        fig.savefig(filepath, format = 'png', dpi = 300)
        

    save_btn.on_click(_on_save)

    slider.observe(update, names="value")
    ui = VBox([
        HBox([btn_minus, slider, btn_plus, save_btn]),
        folder_input,
        out,
    ])
    return ui, update


def plotPlasmaProfile(defaultsFile, filepath, figsize = (6,5)):
    set_matplotlib_formats('retina')
    matplotlib.rcParams.update({'figure.dpi': 200, 'savefig.dpi': 300})

    with open(filepath + '/' + defaultsFile, 'r') as file:
        defaults = yaml.safe_load(file)
        
    plasma_settings = defaults['plasma']
    if(plasma_settings['config'].lower() == 'oven'):
            z, nLi, nHe = generate_Li_oven_profile(P = plasma_settings['P_torr'])
            n0 = np.max(nLi)
    plt.close('all')
    plt.ioff()
    
    fig= plt.figure(figsize=figsize, constrained_layout=True)
    ax = plt.subplot(111)
    plt.suptitle('Lithium Oven Density Profile')
    ax.plot(z,nLi/n0, 'k', label = 'Li')
    ax.plot(z,nHe/n0, 'r', label = 'He')
    ax.legend(loc='upper center')
    ax.set_ylabel(r'$n' + rf'\ [{n0*1e-6:.1e} \ cm^{{-3}}'.replace("+","") + ']$')
    ax.set_xlabel(r'$Z_{sim} \ [m]$')
    return fig



        
