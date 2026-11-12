"""
FACET2_S2E: Simulation tools for FACET-II start-to-end beam dynamics

Layout:
    simulators/          the three S2E stages
        impact.py        IMPACT-T photoinjector
        bmad/            Bmad/Tao: initialization, tracking (core.trackBeam is the S2E driver),
                         simulation runs, energy tuning, DAQ-experiment lattices, scans
            lattice/     setLattice and EPICS-unit element control, linac phasing,
                         optics/matrices, final focus solver
        qpad.py, qpad_picmi.py   QPAD plasma stage
    beam/                bunch generation, manipulation, statistics, microbunching (any code)
    plotting/            phase-space, bunch summary, floorplan, styling

Everything listed in __all__ is available directly as FACET2_S2E.<name>.
"""

## Simulation

from .simulators.bmad.core import (
    initializeTao,
    applyBMADCollectiveEffectSettings,
    trackBeam,
    trackBeamHelper,
    getBeamAtElement,
    writeBeam,
    makeBeamActiveBeamFile,
)
from .simulators.bmad.config import (
    loadConfig,
    applyOtherConfig,
    disableAutoQuadEnergyCompensation,
    disableAutoMagnetEnergyCompensation,
)
from .simulators.bmad.runs import (
    set_beam,
    twiss_match_beam,
    set_lattice_twiss,
    l2_virtual_match,
    tune_bc20_sextupole_offsets,
    BC20_TUNE_SEXTUPOLES,
    L2_MATCH_QUADS,
    golden_twiss_targets,
    GOLDEN_TWISS_PR10571,
    run_initialized_sim,
    run_initialized_sim_edit_lattice_energy_for_dipoles,
    run_initialized_sim_edit_bunch_energy,
    edit_energy_based_on_beam_all,
)
from .simulators.bmad.energy import (
    tune_to_P0Cs,
    edit_energy_based_on_beam_inj,
    edit_energy_based_on_beam_L1,
    edit_energy_based_on_beam_L2,
    edit_energy_based_on_beam_L3,
)
from .simulators.bmad.experiment import (
    get_tao_from_experiment,
    ### BMAD to DAQ funcs
    get_l0a_phase,
    get_l0a_ampl,
    get_l0b_phase,
    get_l0b_ampl,
    get_l0a_energy_from_quads,
    ### Lattice edit functions
    edit_tao_based_on_experiment_database,
    edit_energy_tao_based_on_experiment_database,
)
from .simulators.bmad.scans import (
    make_1d_scan,
    make_comparison_dz_2nd_order,
)

## Lattice

from .simulators.bmad.lattice.set_lattice import (
    setLattice,
    getBendkG, getQuadkG, getSextkG, fixMagnetFields,
    setBendkG, setQuadkG, setSextkG,
    setXOffset, setYOffset,
    getKickerkG, setKickerkG,
    getBendGeVc, setBendGeVc,
    setAllWChicaneSextupoles,
    setAllWChicaneSextupolesXOffsets,
    setAllWChicaneSextupolesYOffsets,
)
from .simulators.bmad.lattice.linac import getLinacMatchStrings, setLinacPhase, setLinacGradientAuto, scaleLinacGradientAuto
from .simulators.bmad.lattice.optics import (
    displayMatrix,
    getMatrix,
    getMatrixLEGACY,
    setLatticeAndGetMatrix,
    launchTwissCorrection,
    launchTwissCorrectionObjective,
    get_element_array,
    get_rij,
    get_tijk,
)
from .simulators.bmad.lattice.final_focus import finalFocusSolver

## Beam

from .beam.generation import (
    make_bunch,
    make_simple_bunch,
    make_simple_bunch_flatter,
    make_simple_bunch_standalone,
    make_simple_bunch_theory_from_bunch_sims,
)
from .beam.manipulation import (
    modifyInputBeamSimple,
    #sqrtm_psd,
    #invsqrtm_psd,
    edit_bunch_parameters_from_PG,
    edit_bunch_parameters,
    cut_length,
    ballisticPropagation,
    nudgeMacroparticleWeights,
    getDriverAndWitness,
    centerBeam,
    collimateBeam,
    sortIndices,
    sliceBeam,
    getSingleBeamSlice,
)
from .beam.analysis import (
    calcBMAG,
    smallestInterval,
    smallestIntervalImpliedSigma,
    smallestIntervalImpliedEmittance,
    smallestIntervalImpliedEmittanceModelFunction,
    emittance,
    getBeamSpecs,
    generalizedEmittanceSolver,
    generalizedEmittanceSolverObjective,
)
from .beam.microbunching import (
    addLHmodulation,
    make_modulated_bunch,
    hist,
    get_nbins_for_wavelength,
    get_spectrum,
    print_spec,
    #find_nearest,
    get_spec_band,
    analyze_spec,
    get_microbunching_gain_from_beams,
    get_microbunching_gain,
)

## Plotting

from .plotting.phase_space import plotMod, slicePlotMod
from .plotting.floorplan import floorplanPlot
from .plotting.bunch_summary import (
    print_result,
    print_result_from_tao,
    print_result_from_file,
)
from .plotting.style import (
    enable_plt_styling,
    #normalize_arrays,
    make_a_plot,
)

## External codes

from .simulators.qpad import (
    plotInteractiveQPADFigure,
    saveAllQPADFigures,
    plotPlasmaProfile,
)

__all__ = [
    # Core initialization and tracking
    'initializeTao',
    'applyBMADCollectiveEffectSettings',
    'trackBeam',
    'trackBeamHelper',
    'getBeamAtElement',
    'ballisticPropagation',
    
    # Beam manipulation and analysis
    'nudgeMacroparticleWeights',
    'getDriverAndWitness',
    'writeBeam',
    'makeBeamActiveBeamFile',
    'centerBeam',
    'collimateBeam',
    'sliceBeam',
    'getSingleBeamSlice',
    
    # Statistical analysis
    'smallestInterval',
    'smallestIntervalImpliedSigma',
    'smallestIntervalImpliedEmittance',
    'smallestIntervalImpliedEmittanceModelFunction',
    'emittance',
    'getBeamSpecs',
    
    # Matrix and lattice operations
    'displayMatrix',
    'getMatrix',
    'getMatrixLEGACY',
    'setLatticeAndGetMatrix',
    
    # Beam modulation
    'addLHmodulation',
    'calcBMAG',
    
    # Configuration utilities
    'loadConfig',
    
    # Optimization and correction
    'launchTwissCorrection',
    'launchTwissCorrectionObjective',
    'generalizedEmittanceSolver',
    'generalizedEmittanceSolverObjective',
    
    # Lattice configuration
    'disableAutoQuadEnergyCompensation',
    'disableAutoMagnetEnergyCompensation',
    'applyOtherConfig',
    
    # Helper utilities
    'sortIndices',
    
    # QPAD visualization and analysis
    'plotInteractiveQPADFigure',
    'saveAllQPADFigures',
    'plotPlasmaProfile',
    
    # Plotting utilities
    'plotMod',
    'slicePlotMod',
    'floorplanPlot',
    
    # Linac configuration
    'getLinacMatchStrings',
    'setLinacPhase',
    'setLinacGradientAuto',
    'scaleLinacGradientAuto',
    
    # Lattice element control
    'setLattice',
    'getBendkG', 'getQuadkG', 'getSextkG', 'fixMagnetFields',
    'setBendkG', 'setQuadkG', 'setSextkG',
    'setXOffset', 'setYOffset',
    'getKickerkG', 'setKickerkG',
    'getBendGeVc', 'setBendGeVc',
    
    # Final focus optimization
    'finalFocusSolver',

    # Bmad workflow built around get_tao_from_experiment(): beams, runs, energy tuning, DAQ settings

    ## Bunch support functions

    ### Create a bunch
    'make_bunch',
    'make_simple_bunch',
    'make_simple_bunch_flatter',
    'make_simple_bunch_standalone',
    'make_simple_bunch_theory_from_bunch_sims',
    ### Add microbunching to a bunch
    'make_modulated_bunch',
    ### Modify the bunch as a whole (sizes, means, chirps, correlations),
    'modifyInputBeamSimple',
    #'sqrtm_psd',
    #'invsqrtm_psd',
    'edit_bunch_parameters_from_PG',
    'edit_bunch_parameters',
    ### Cut the bunch to a certain length
    'cut_length',

    ## Initialize and run a simulation

    ### High end
    'get_tao_from_experiment',
    'set_beam',
    'twiss_match_beam',
    'set_lattice_twiss',
    'l2_virtual_match',
    'tune_bc20_sextupole_offsets',
    'BC20_TUNE_SEXTUPOLES',
    'L2_MATCH_QUADS',
    'golden_twiss_targets',
    'GOLDEN_TWISS_PR10571',
    'run_initialized_sim',
    'run_initialized_sim_edit_bunch_energy',
    'run_initialized_sim_edit_lattice_energy_for_dipoles',
    ### Correct the lattice to match the desired Pz
    'tune_to_P0Cs',
    'edit_energy_based_on_beam_inj',
    'edit_energy_based_on_beam_L1',
    'edit_energy_based_on_beam_L2',
    'edit_energy_based_on_beam_L3',
    'edit_energy_based_on_beam_all',

    ## Edit lattice according to experiment

    ### BMAD to DAQ funcs
    'get_l0a_phase',
    'get_l0a_ampl',
    'get_l0b_phase',
    'get_l0b_ampl',
    'get_l0a_energy_from_quads',
    ### Lattice edit functions
    'edit_tao_based_on_experiment_database',
    'edit_energy_tao_based_on_experiment_database',

    ## Scans

    'make_1d_scan',
    'make_comparison_dz_2nd_order',

    ## Simulation parameters functions

    # Get lattice info
    'get_element_array',
    'get_rij',
    'get_tijk',
    ### Sextupole settings
    'setAllWChicaneSextupoles',
    'setAllWChicaneSextupolesXOffsets',
    'setAllWChicaneSextupolesYOffsets',

    ## Plotting functions

    'enable_plt_styling',
    ### Display a bunch
    'print_result',
    'print_result_from_tao',
    'print_result_from_file',

    ## Make a nice plot from arrays
    #'normalize_arrays',
    'make_a_plot',

    ## Display the bunch density distribution \rho(z)
    'hist',

    ## Spectrum functions
    'get_nbins_for_wavelength',
    'get_spectrum',
    'print_spec',
    #'find_nearest',
    'get_spec_band',
    'analyze_spec',
    'get_microbunching_gain_from_beams',
    'get_microbunching_gain',
]

__version__ = '0.1.0'
