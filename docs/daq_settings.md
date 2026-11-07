# Loading settings from a FACET-II DAQ scan

The magnet and RF settings saved in a FACET-II DAQ scan can be loaded into the Bmad lattice. This is optional: without a scan, the simulations use the lattice from `setLattice_configs/` (see [bmad_simulations.md](bmad_simulations.md)).

## Where the DAQ data is

The DAQ data is only accessible from inside SLAC; its location is given in the internal FACET-II documentation. Tell the package where it is in one of two ways:

- set the `FACET2_DAQ_ROOT` environment variable to the root folder of the DAQ data, e.g. once for the conda environment:
  ```bash
  conda env config vars set FACET2_DAQ_ROOT=<DAQ data root>
  conda activate <your environment>     # re-activate to apply
  ```
- or pass `daq_root="<DAQ data root>"` to `get_tao_from_experiment()`.

The DAQ data may not be visible from every node (see [installation.md](installation.md#on-s3df)).

## Choosing a scan

A scan is identified by three strings:

| Argument | Example | Meaning |
|---|---|---|
| `experiment` | `"BEAMPHYS"` | Experiment name |
| `scan_number` | `"14438"` | DAQ scan number |
| `date` | `"/2026/20260121"` | `/YYYY/YYYYMMDD` of the scan |

The file that gets read is
`<DAQ data root>/<experiment><date>/<experiment>_<scan_number>/<experiment>_<scan_number>.mat`.

## With `get_tao_from_experiment()`

Pass the three strings to `get_tao_from_experiment()`; everything else works as described in [bmad_simulations.md](bmad_simulations.md). There, the beam energy is matched to the lattice automatically, and the correctors are off unless `correctors_coef` is set.

## With `initializeTao()` / `trackBeam()`

The DAQ values can also be applied on top of the standard `initializeTao()` / `trackBeam()` setup: the lattice from `defaults.yml`, CSR only in the bunch compressors, and the beam loaded through `activeBeamFile.h5`.

```python
import os
import FACET2_S2E as qs
from facet2_daq import DATASET

REPO = "/path/to/FACET2-S2E"
DAQ_ROOT = os.environ["FACET2_DAQ_ROOT"]   # see "Where the DAQ data is" above

tao = qs.initializeTao(
    filePath=REPO,
    inputBeamFilePathSuffix="/beams/2024-12-11_Impact_OneBunch/2024-12-11_oneBunch.h5",
    numMacroParticles=50000,
    csrTF=True,
)

ds = DATASET("", "BEAMPHYS", "14438", pathfull=f"{DAQ_ROOT}/BEAMPHYS/2026/20260121")

# Everything (RF, quads, sextupoles, correctors):
qs.edit_tao_based_on_experiment_database(tao, ds)            # correctors_coef=0 to skip the correctors
# ...or only the RF (energy profile), leaving the magnets untouched:
# qs.edit_energy_tao_based_on_experiment_database(tao, ds)

qs.trackBeam(tao, filepath=REPO, trackStart="L0AFEND", trackEnd="END",
             centerBC14=True, assertBC14Energy=True,
             centerBC20=True, assertBC20Energy=True)

P = qs.getBeamAtElement(tao, "END")    # openPMD ParticleGroup
```

Here the beam keeps the energy stored in its file. The DAQ RF settings can shift the design energy at L0AFEND, so the beam and the lattice may start at slightly different energies.

The corrector defaults differ on purpose: `edit_tao_based_on_experiment_database()` loads everything, correctors included, while `get_tao_from_experiment()` leaves the correctors off because simulations usually don't want them.

## What is loaded

Each value is the mean over all shots in the scan. If a magnet or phase was the scanned variable, the lattice gets its average value over all steps.

| Group | Lattice elements | DAQ source |
|---|---|---|
| Injector RF | `L0AF` phase and voltage, `L0BF` phase and voltage | `KLYS_LI10_31_SFB_PDES` (−20° offset), `KLYS_LI10_31_ADES`, `KLYS_LI10_41_SFB_PDES`, `KLYS_LI10_41_ADES`. Both voltages are then scaled together so the design energy at BX10661 is 125 MeV. |
| L1 / L2 / L3 phases | All cavities of each linac get one phase | `KLYS_LI11_11_SSSB_PDES`, `LI14_SBST_1_PHAS`, `LI19_SBST_1_PHAS` |
| L1 / L2 / L3 amplitudes | Not loaded | The gradients are set so the energies are 335 / 4500 / 10000 MeV. |
| Quadrupoles | Sector 10 (11 quads), sector 11 (11 quads), Q19851 and Q19871, and sector 20 (W-chicane Q1–Q6 including the boost supplies, final focus, spectrometer) | `*_BDES`, `*_BCON`, `*_BACT` |
| Sextupoles | S1EL, S2EL, S3EL, S3ER, S2ER, S1ER strengths | `LI20_LGPS_*_BACT` |
| Sextupole movers | X/Y offsets of S1EL, S2EL, S2ER, S1ER (S3 stays at 0) | `SIOC_SYS1_ML00_AO*` (mm) |
| Correctors | Sector 10 from XC10721 (or from the start with `correctors_from_beg`) and sector 11, scaled by `correctors_coef` | `*COR_*_BDES`, `*COR_*_BCON` |
| Dipole energies | DL10, BC11, BC20 bends (BC14 stays at 4500 MeV). Used only with `tune_dipoles=True`. | `BEND_*`, `LI20_LGPS_*` |

**Not loaded:**
- L1/L2/L3 amplitudes;
- phases of individual klystrons (one phase per linac is used);
- quadrupoles in sectors 12–18 and most of sector 19 (these are not saved in the DAQ);
- bend fields (setting `B_FIELD` in Bmad moves every downstream element);
- laser heater, XTCAV and kickers;
- the beam distribution.

These keep their values from `defaults.yml` and the `lattice` file.

The DAQ files are read with `DATASET` from [facet2-daq](https://github.com/slaclab/facet2-daq), which is installed with the conda environment. It is based on Robert Ariniello's [standaloneFACETScripts](https://github.com/rariniello/standaloneFACETScripts).
