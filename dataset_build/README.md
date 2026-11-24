# Building the particle dataset

These scripts rebuild the particle dataset used to train the models of the
AeroSEL/AeroMELD paper and of the WRF advection paper (see the top-level
README) from the raw PartMC-MOSAIC scenario library in the Illinois Data Bank
(doi:10.13012/B2IDB-2774261_V1, CC0). They use the `partnn` package and the
data family `05_bwchi` of `notebooks/21_data/01_datasets.yml`, and do not
change any other code in this repository.

The dataset has 25,000 populations (1000 scenarios, 25 times each), each
resampled to 1000 particles with 15 chemical species, with their diagnostics:
number and speciated mass size distributions, the CCN spectrum, the frozen
fraction, and the scattering and absorption coefficients.

Requirements: the packages of `requirements.txt`, a CUDA GPU for the
diagnostics, `pigz`, and about 300 GB of disk for the raw scenarios.

## Steps

```bash
RAW=/path/to/raw/01_bwchisamp
DATA=/path/to/data

# 1. Download the raw scenarios (281 GB as 1000 per-scenario tarballs).
bash dataset_build/fetch_idb.sh $RAW

# 2. Resample the particles (CPU; about 15 minutes with 24 readers).
python dataset_build/build_prtcls_h5.py --raw-dir $RAW --out $DATA/02_bwchisamp_00.h5 \
    --stage prt --n-readers 24 --device cpu

# 3. The diagnostics on 19 size bins up to 100 um (GPU; about 10 minutes).
python dataset_build/build_prtcls_h5.py --raw-dir $RAW --out $DATA/02_bwchisamp_00.h5 \
    --stage diag --opt-bs 8 --device cuda:0

# 4. The PM10 dataset: every particle of 10 um or more (118 particles in 117
#    populations) replaced by a random particle under 10 um from the same
#    PartMC population (CPU).
python dataset_build/clean_coarse.py $DATA/02_bwchisamp_00.h5 $DATA/04_bwchisamp_pm10.h5 \
    --raw-dir $RAW --n-readers 12

# 5. The diagnostics of the PM10 dataset on 15 size bins up to 10 um (GPU).
python dataset_build/build_prtcls_h5.py --raw-dir $RAW --out $DATA/04_bwchisamp_pm10.h5 \
    --stage diag --opt-bs 8 --n-bins 15 --diam-high 1e-5 --device cuda:0

# 6. The optics with the geometric cross-section of the whole particle (the
#    dataset's own optics use that of the shell only), which replace the
#    dataset's optics in the WRF advection paper's training targets. The CCN
#    spectrum is also written at 287 K (as stored, a check) and at 290 K.
python dataset_build/recompute_diagnostics.py $DATA/04_bwchisamp_pm10.h5 $DATA/diag_fixed.npz
```

`h5_digest.py` writes a one-line-per-dataset digest (shape, dtype, sha1, sum,
min, max) of an h5 file and compares two digests, to check a rebuild against
the original without moving the files:

```bash
python dataset_build/h5_digest.py $DATA/04_bwchisamp_pm10.h5 > pm10.digest
python dataset_build/h5_digest.py --compare pm10.digest other.digest
```

`build_prtcls_h5.py` is a script version of the data notebook of this
repository's development version (data family `05_bwchi`). Its resampling
random number generator is used in the same order as the notebook's, so the
particles are reproduced exactly.
