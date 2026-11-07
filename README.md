# FACET-II start-to-end (S2E) simulation toolkit

This repository contains utilities, Jupyter notebooks, and configuration files used to perform start-to-end (S2E) simulations of the [FACET-II](https://facet-ii.slac.stanford.edu/) particle accelerator beamline, a US Department of Energy National National User Facility which hosts hundreds of users a year.  The core workflow uses [IMPACT-T](https://github.com/impact-lbl/IMPACT-T) for beam generation and low energy transport, [Bmad, Tao, and PyTao](https://www.classe.cornell.edu/bmad/) for most of the beam transport through the kilometer-long linear accelerator, and, optionally, [QPAD](https://picksc.physics.ucla.edu/qpad.html) for particle-in-cell simulations of the beam in plasma, and [openPMD-beamphysics](https://github.com/ChristopherMayes/openPMD-beamphysics) for handling beam files. It is intended to abstract away unnecessary detail so present or prospective facility users can quickly and easily run the most common types of simulations including parameter scans, constrained optimization of both Twiss and multiparticle tracking objectives, and jitter sensitivity analysis.


## Installation

You need conda and Git LFS (the beam files are stored with LFS; a full pull is larger than 2 GB).

```bash
git clone https://github.com/slaclab/FACET2-S2E.git
cd FACET2-S2E
git lfs pull
conda env create -f bmadQPADCondaEnv.yml      # bmadQPADCondaEnv_dev.yml for development (adds test tools)
conda activate bmad-qpad
pip install -e .
python -m ipykernel install --user --name bmad-qpad --display-name "bmad-qpad"   # optional, for Jupyter
```

The editable install (`-e`) lets the package find the lattice, configs and scratch folder in this repository. See [docs/installation.md](docs/installation.md) for details, partial LFS downloads and notes for S3DF.

## Documentation

| Document | Contents |
|---|---|
| [docs/installation.md](docs/installation.md) | Installation, Git LFS and large files, working on S3DF |
| [docs/bmad_simulations.md](docs/bmad_simulations.md) | Bmad simulations with `get_tao_from_experiment()`: beam preparation, collective effects, energy tuning and bends, all options |
| [docs/daq_settings.md](docs/daq_settings.md) | (Optional) loading magnet and RF settings from a FACET-II DAQ scan, and what is loaded |
| [docs/api_reference.md](docs/api_reference.md) | Main functions with examples, and where each function lives in the package |
| [tests/README.md](tests/README.md) | Test suite: categories, markers, how to run |

The notebooks in the `ARCHIVE` folder are previous investigations and may serve as additional examples.

## Examples

The notebooks in [`examples/`](examples/) demonstrate typical workflows:

* **`Example - Basic introduction.ipynb`** – runs Bmad simulations with a reference lattice and beam file.  It also introduces some basic functionality like reading and setting magnets using control system units, phasing linacs, etc.
* **`Example - IMPACT-T beam generation.ipynb`** – performs a full S2E run that generates the input beam with IMPACT‑T before tracking it through the lattice to the end of the beamline
* **`Example - Final focus tuning.ipynb`** – demonstrates the final focus optics optimizer to pick magnet settings to achieve desired Twiss
* **`Example - Multiparticle tracking optimization.ipynb`** – demonstrates optimization constrained by real-world hardware limits of a multiparticle tracked beam
* **`Example - Solution postprocessing and analysis.ipynb`** – postprocessing and analysis of the beam throughout the lattice
* **`Example - BMAD tutorial.ipynb`** – Bmad simulations with `get_tao_from_experiment()`, optionally with the lattice settings saved in a FACET-II DAQ scan. See [docs/bmad_simulations.md](docs/bmad_simulations.md)
* **`Example - Beam visualization.nb`** – Mathematica notebook for advanced beam visualization and analysis, including 3D animation generation
* **`Example - Optimization progress dashboard.nb`** – Mathematica companion notebook which visualizes optimization progress, e.g. parameter sensitivities and convergence
* **`Example - Jitter study.py`** – Parallel computation of many simulations with parameters subject to jitter, informed by real-world measurements
* **`Example - QPAD jitter simulation.py`** – performs a single S2E jitter simulation using QPAD to model plasma wakefield acceleration in a Lithium Oven plasma source (located at PENT+25 cm).

## Main Repo Features

- `initializeTao()` – set up a Bmad/PyTao instance. Optionally run IMPACT‑T to create a beam or import a reference beam
- `setLattice()` – apply lattice configuration to commonly changed knobs using a dictionary or reference file
- `trackBeam()` – track a beam between arbitrary points in the lattice, applying specialized functions like centering or energy correction at checkpoints
- `get_tao_from_experiment()` – one call that sets up the lattice, prepares the beam, sets the collective effects and tracks, optionally with the magnet and RF settings saved in a FACET-II DAQ scan ([docs/bmad_simulations.md](docs/bmad_simulations.md))

### Other features

- `simulators.bmad.lattice.set_lattice` functions to translate between the language and units of the FACET-II EPICS control system and simulation
- `simulators.bmad.lattice.linac` which conveniently phases and sets the gradients of the linacs
- Plotting tools for displaying beams and the beamline itself
- Twiss optimizers for the final focus and golden lattice matching
- Infrastructure for dealing with two-bunch operation
- Various options of calculating spot sizes and emittances
- Tools to model laser heater interactions
- Mathematica notebooks which track and analyze optimizer progress
- Mathematica notebooks which visualize beam files, including as 3D animations


## Tests

Unit, integration, smoke and system tests are in `tests/` and run through GitHub CI on each commit:

```bash
pytest tests/ -m "not slow"     # fast tests
pytest tests/                   # everything, including full simulator runs
```

More details in [tests/README.md](tests/README.md).

## Repository layout

```
ARCHIVE/                 Historical studies, optimizations, and experimental notebooks
beams/                   Reference beams and scripts to generate them
bmad/                    Bmad 'golden lattice' (https://github.com/slaclab/facet2-lattice)
docs/                    Documentation (see below)
examples/                Example Jupyter notebooks and Python scripts
impact/                  IMPACT‑T configuration files
other_configs/           Atypical configurations including misalignment and steering solutions
qpad/                    QPAD configuration files
setLattice_configs/      Reference configurations
src/FACET2_S2E/          Main package source code, sorted by purpose:
  ├── simulators/        The three start-to-end stages
  │   ├── impact.py      IMPACT-T photoinjector
  │   ├── bmad/          Bmad/Tao: initialization and tracking (trackBeam drives the full S2E),
  │   │   │              simulation runs, energy tuning, lattices from DAQ data, scans
  │   │   └── lattice/   setLattice and EPICS-unit element control, linac phasing,
  │   │                  optics/matrices, final focus solver
  │   └── qpad.py        QPAD plasma stage (and qpad_picmi.py)
  ├── beam/              Bunch generation, manipulation, statistics, microbunching
  └── plotting/          Phase-space, bunch summary and floorplan plots; styling
tests/                   Automated test suite (unit, integration, system tests)
  ├── unit/              Unit tests for core functions
  ├── integration/       Integration tests calling real functions
  └── system/            End-to-end simulator tests
bmadCondaEnv.yml         Conda environment specification for Bmad and Impact-T
bmadQPADCondaEnv.yml     Conda environment specification for Bmad, Impact-T, and QPAD
bmadQPADCondaEnv_dev.yml Development environment with testing dependencies
pyproject.toml           Package configuration and dependencies
```

## Acknowledgements

FACET-II DAQ datasets are read with [facet2-daq](https://github.com/slaclab/facet2-daq),
which is based on Robert Ariniello's
[standaloneFACETScripts](https://github.com/rariniello/standaloneFACETScripts).

## Support

For support, contact @majernik-slac-stanford-edu
