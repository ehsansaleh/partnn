# Learning Neural Representations for Particles

This repository contains code supporting three papers:

1. **Compact Aerosol Size-Composition Representations Preserving CCN, Optical, and Ice Nucleation Properties**
2. **Reconstructing the Aerosol State from Partial Observations with Generative Modeling**
3. **AeroSEL: A Linear Set Embedding for Aerosol Populations**

The code trains and evaluates neural representations of aerosol size-composition states, including histogram-based VAEs, conditional VAEs, and particle/set-based embeddings.

## Repository Overview

The `partnn` package contains the reusable model, preprocessing, metric, evaluation, visualization, and I/O code. The two main experiment entry points are:

- `partnn/vaehist.py` for histogram-based VAE and conditional VAE experiments.
- `partnn/vaepart.py` for the particle/set-based AeroSEL experiments.

Experiment configurations live under `configs`:

- `configs/02_adhoc` contains the histogram VAE and conditional VAE configurations for the first two papers.
- `configs/03_prtcls` contains the particle/set-based configurations for the AeroSEL paper.

The notebooks are mostly for data preparation, interactive inspection, and figure generation:

- `notebooks/n02_histdata.ipynb` builds the histogram data used by the histogram and conditional experiments.
- `notebooks/n20_vaehist.ipynb` is the notebook version of the histogram VAE workflow.
- `notebooks/n08_plotting.ipynb` collects trained outputs and generates the paper-style diagnostic figures.
- `notebooks/08_plotting/37_aerompl.yml` defines the plotting experiment groups used by the figure notebook.

Install the Python dependencies with:

```bash
pip install -r requirements.txt
```

The code expects repository-local `data`, `results`, and `storage` directories. The main aerosol histogram input referenced by the configs is `data/02_masshist/03_bwchisamp.nc`. Generated result files are not tracked in git.

## Compact Aerosol Size-Composition Representations Preserving CCN, Optical, and Ice Nucleation Properties

This paper uses the histogram VAE workflow in `partnn/vaehist.py`. The main configurations are:

- `configs/02_adhoc/11_mlphist.yml` for the MLP histogram VAE experiments.
- `configs/02_adhoc/12_cnnhist.yml` for the CNN histogram VAE experiments.
- `configs/02_adhoc/10_utils`, `11_utils`, and `12_utils` for shared preprocessing, neural network, metric, evaluation, and plotting fragments.

A useful first check is to run the driver in dry-run mode with one of the main configs:

```bash
python -m partnn.vaehist -c 02_adhoc/11_mlphist -d cpu --dry-run
```

For full experiments, use the same entry point without `--dry-run`, usually on a CUDA device. The plotting notebook reads the resulting `20_vaehist` outputs through the `trad.hist` group in `notebooks/08_plotting/37_aerompl.yml`.

## Reconstructing the Aerosol State from Partial Observations with Generative Modeling

This paper uses the conditional VAE workflow, also through `partnn/vaehist.py`. The main public starting configurations are:

- `configs/02_adhoc/18_mlpcnd.yml` for conditional mass-quantile experiments and ablations.
- `configs/02_adhoc/21_mlpcnd.yml` for low-dimensional ACSM/SMPS-style measurement conditioning.
- `configs/02_adhoc/23_mlpcnd.yml` for higher-dimensional measurement conditioning.

The same driver pattern applies:

```bash
python -m partnn.vaehist -c 02_adhoc/21_mlpcnd -d cpu --dry-run
```

For full runs, choose the relevant conditional config and run on the intended compute device. The figure workflow is in `notebooks/n08_plotting.ipynb`; the corresponding plotting groups in `notebooks/08_plotting/37_aerompl.yml` are the `cond.*` groups, especially `cond.cont.acsmsmps`, `cond.cont.trilbl`, and the `cond.mqnt.*` ablation groups.

## AeroSEL: A Linear Set Embedding for Aerosol Populations

This paper uses the particle/set-based workflow in `partnn/vaepart.py`. The main configuration is:

- `configs/03_prtcls/02_mlpprt.yml` for the MLP particle VAE experiments.
- `configs/03_prtcls/12_utils` for shared particle preprocessing, network, metric, visualization, and evaluation fragments.

A dry-run sanity check is:

```bash
python -m partnn.vaepart -c 03_prtcls/02_mlpprt -d cpu --dry-run
```

Full AeroSEL runs use the same entry point without `--dry-run`. The plotting notebook reads the VAE particle outputs through the `trad.part.gen` group in `notebooks/08_plotting/37_aerompl.yml`; those plots are written under `notebooks/08_plotting/supplement/10_vaepart`.

## Generating Figures

The main paper-style figures are generated from `notebooks/n08_plotting.ipynb`. That notebook uses `notebooks/08_plotting/37_aerompl.yml` to map experiment groups to result files and figure directories.

At a high level:

1. Run the relevant training/evaluation configs so the expected HDF result files exist under `results` or the notebook-specific result directory.
2. Open `notebooks/n08_plotting.ipynb`.
3. Select the appropriate experiment group in the data-collection cell: `trad.hist` for the first paper, `cond.*` for the second paper, and `trad.part.gen` for the AeroSEL paper.
4. Run the corresponding figure sections.

The notebook is intentionally shared across papers, so not every cell is relevant to every paper.
