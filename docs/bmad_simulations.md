# Bmad simulations with `get_tao_from_experiment()`

`get_tao_from_experiment()` sets up and runs a Bmad simulation of the FACET-II linac in one call: it builds the lattice, prepares the beam, sets the collective effects and tracks the beam.

It can also load the magnet and RF settings saved in a FACET-II DAQ scan. This is optional: if you don't need the DAQ data, leave `experiment` and `scan_number` empty, and the lattice is `setLattice_configs/defaults.yml` plus the file given in `lattice`. Loading DAQ settings is described in [daq_settings.md](daq_settings.md).

Examples of every option are in [`examples/Example - BMAD tutorial.ipynb`](<../examples/Example - BMAD tutorial.ipynb>).

## Example

```python
import FACET2_S2E as qs

REPO = "/path/to/FACET2-S2E"

tao = qs.get_tao_from_experiment(
    filepath=REPO,                       # optional with "pip install -e ."
    start="L0AFEND", finish="END",
    file_ext=REPO + "/beams/2024-12-11_Impact_OneBunch/2024-12-11_oneBunch",   # no ".h5"
    N_to_use_from_file=5e4,
    lattice="setLattice_configs/2024-12-09_oneBunch_CSR-on_optimized.yml",
    csrTF=True, lscTF=True,
    sr_wakes_on=True, lr_wakes_on=True,
    # experiment="BEAMPHYS", scan_number="14438", date="/2026/20260121",   # to load the settings of a DAQ scan
)

P = qs.getBeamAtElement(tao, "END", tToZ=False)
qs.print_result_from_tao(tao, location="END", couples=[['x', 'xp'], ['y', 'yp'], ['z', 'pz']], z_from_t=True)
```

The function does the following, in order:

1. Initializes Tao, applies `defaults.yml` plus `lattice`, and sets the collective effects.
2. If a scan is given, loads its settings (see [daq_settings.md](daq_settings.md)).
3. Prepares the beam and writes it to `temp_beam/temp.h5` and `temp_beam/temp_e.h5`:
   - sets the beam energy to the design energy at `start` (see `energy`);
   - optionally resamples, rescales or re-centers it (`N_to_use_from_file`, `moments`, `means`);
   - sets all z to 0 and centers t.
4. Optionally runs the beam-based energy feedback (`energy_edit_on_beam`) and the dipole treatment (`tune_dipoles`).
5. Tracks from `start` to `finish` (if `run=True`). The beam at `start` and `finish` is saved to `temp_beam/<ELEMENT>temp.h5`.

## Main options

| Option | Default | Meaning |
|---|---|---|
| `filepath` | `None` | Path to this repository. If `None`, it is found from the installed package (requires `pip install -e .`). |
| `experiment`, `scan_number`, `date` | `""` | DAQ scan to load (optional, see [daq_settings.md](daq_settings.md)). |
| `daq_root` | `None` | Root folder of the DAQ data. If `None`, the `FACET2_DAQ_ROOT` environment variable is used. |
| `lattice` | `2024-12-09_oneBunch_CSR-on_optimized.yml` | Settings applied on top of `defaults.yml`. |
| `file_ext` | `""` | Input beam, without `.h5`. If `""`, a Gaussian beam is generated from `moments`. |
| `N_to_use_from_file` | `None` | Randomly sample this many particles from the file. |
| `gaussFromExternal`, `N_in_simple_bunch` | `False`, `5e4` | Replace the file beam with a Gaussian beam of the same RMS sizes. |
| `energy` | `None` | `None`: design energy at `start`. `-1`: keep the file energy. A positive number: that energy in MeV. |
| `moments`, `means`, `charge` | — | Rescale or re-center the beam (see the docstring). |
| `csrTF`, `lscTF` | `False` | CSR and space charge. When on, they are applied to **all** elements. |
| `csr_method`, `lsc_method`, `n_bin`, `grid_size` | `1_dim`, `slice`, 32, `[32,32,32]` | Collective-effect calculation settings. |
| `n_rf_steps` | `100` | RF steps per accelerating structure. `None` keeps the lattice's 1000 (5000 in L0 and LI11): the same beam, but ~8x slower. |
| `sr_wakes_on`, `lr_wakes_on` | **`False`** | Short- and long-range wakes. **Off by default.** |
| `edit_only_energy_from_exp` | `False` | Load only the RF settings from the DAQ, not the magnets. |
| `correctors_coef` | **`0`** | Scale factor for the DAQ corrector values. `0` turns them off; `-1/10` uses the experimental values. |
| `correctors_from_beg` | `False` | Load the sector-10 correctors from the start of the injector instead of from the dogleg. |
| `desired_P0Cs_MeV` | `[None]*4` | Design energies other than 125 / 335 / 4500 / 10000 MeV, keeping the bend geometry of the nominal energies. |
| `edited_bunch_energy_at_checkpoints_MeV` | `[None]*4` | Force the beam energy at BX0FBEG, BC11CBEG, ENDL2F and ENDL3F_2. |
| `energy_edit_on_beam`, `desired_beam_energies_for_the_feedback` | `False` | Beam-based energy feedback: scale the cavities until the tracked beam reaches the target energies. |
| `tune_dipoles`, `tune_dipoles_to_125_335_4500_10000_MeV` | `True`, `False` | Fix the bend fields to those of the DAQ energies (or of the nominal energies). |
| `run`, `locationsToSave` | `True`, `[start, finish]` | Track now, and where to save the beam. |

## Running an already initialized lattice

With `run=False`, you can modify `tao` first and then track with:

- `qs.run_initialized_sim(tao, start, finish)` (add `treat_dipoles_TF=True` to fix the bend fields to the nominal-energy values first)
- `qs.run_initialized_sim_edit_lattice_energy_for_dipoles(tao, start, finish, desired_P0Cs_MeV=...)`
- `qs.run_initialized_sim_edit_bunch_energy(tao, start, finish, edited_bunch_energy_at_checkpoints_MeV=...)`
- `qs.set_beam(tao, file)` to load another beam.

`initializeTao(autoLoadActiveFile=False, loadCustomLatticeTF=True, latticeFile=...)` sets up only the lattice and the collective effects, as `get_tao_from_experiment()` does internally; the beam is then loaded with `set_beam()`. With the default `autoLoadActiveFile=True`, `initializeTao()` also loads the beam, for use with `trackBeam()`.

## Things to know

- **Scratch files are overwritten.** To save disk space, intermediate beams are written to `<repo>/temp_beam/` under fixed names (`temp.h5`, `temp_e.h5`, `<ELEMENT>temp.h5`) and overwritten on every run. Copy any result you want to keep, and don't run several simulations from the same repository at the same time.
- **Repository path.** With an editable install (`pip install -e .`), `filepath` can be omitted; in a notebook, `str(Path(qs.__file__).resolve().parents[2])` gives the same path. With a regular `pip install .`, always pass it.
- **CSR and space charge apply to every element**, not only the bunch compressors. 3D methods (`steady_state_3d`, `fft_3d`) are slow: test with few particles first and run long simulations through Slurm, not on a login node. `steady_state_3d` needs Bmad 20260828 or newer (see [installation.md](installation.md)).
