# Installation

## Requirements

- **conda** (for example [Miniforge](https://github.com/conda-forge/miniforge)).
- **Git LFS.** The beam files are stored with Git LFS. A full `git lfs pull` is larger than 2 GB, so the target folder needs more than 2 GB of free space (or pull only the beams you need, see step 1).

## Steps

1. **Clone the repository.**
   ```bash
   git clone https://github.com/slaclab/FACET2-S2E.git
   cd FACET2-S2E
   git lfs pull
   ```
   To skip the large files at first and download only the beams you need:
   ```bash
   GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/slaclab/FACET2-S2E.git
   cd FACET2-S2E
   git lfs pull --include="beams/2024-12-11_Impact_OneBunch/*"
   ```
   Without LFS the notebooks fail to load the example beams (see [Large files](#large-files)).

2. **Create the conda environment.** Use `bmadQPADCondaEnv.yml` to use the package, or `bmadQPADCondaEnv_dev.yml` (adds the test dependencies) to develop it:
   ```bash
   conda env create -f bmadQPADCondaEnv.yml
   conda activate bmad-qpad
   ```
   The `prefix:` line in the yml files points to another user's directory. If conda tries to use it, give the location explicitly:
   ```bash
   conda env create -f bmadQPADCondaEnv.yml -p /path/to/envs/bmad-qpad
   conda activate /path/to/envs/bmad-qpad
   ```
   An environment created before the DAQ reader became a separate package lacks
   [facet2-daq](https://github.com/slaclab/facet2-daq), and `import FACET2_S2E` then fails
   with `ModuleNotFoundError: No module named 'facet2_daq'`. Add it to that environment with:
   ```bash
   pip install git+https://github.com/slaclab/facet2-daq.git@v0.1.0
   ```

3. **Install the package in editable mode.**
   ```bash
   pip install -e .
   ```
   With an editable install the package finds the lattice (`bmad/`), the configs (`setLattice_configs/`) and the scratch folder (`temp_beam/`) inside this repository, so the `filePath` / `filepath` arguments can be omitted. With a regular `pip install .`, always pass the repository path.

4. **(Optional) Register the Jupyter kernel.**
   ```bash
   python -m ipykernel install --user --name bmad-qpad --display-name "bmad-qpad"
   ```

## On S3DF

- Home quotas are small. Install conda and the environments in a group directory.
- The FACET-II DAQ data is **not** mounted on the login nodes. To load DAQ scans, work on a node where it is visible, e.g. a compute node through Slurm or S3DF OnDemand, and tell the package where the data is (see [daq_settings.md](daq_settings.md#where-the-daq-data-is)). Everything else works on any node.
- Run long simulations through Slurm, not on a login node.

## Large files

Git LFS is required because beam files can be tens of megabytes. Without LFS you may see errors such as:

```
OSError: Unable to synchronously open file (file signature not found)
```

If you cannot use LFS, download the `.h5` beam files from another source and place them in the corresponding folders under `beams/`.
