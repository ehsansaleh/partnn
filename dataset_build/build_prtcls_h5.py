#!/usr/bin/env python
"""
Rebuild `data/06_prtcls/02_bwchisamp_00.h5` (the AeroMELD training data) from
the raw PartMC-MOSAIC scenarios in the Illinois Data Bank deposit
(doi:10.13012/B2IDB-2774261_V1).

This is a script transcription of the data notebook `n21_data.ipynb`
(cells 5, 8, 10, 11, 13, 16, 18, 20, 22) for data family `05_bwchi`, run as
a single rank (`my_rank, n_workers = 0, 1`, as in the notebook's cell 2). The
only differences from the notebook are:

  * The input is `<raw-dir>/scenario_N.tar.gz` (as written by `fetch_idb.sh`).
    Worker processes stream each tarball through pigz and open its netCDF
    files in memory, so nothing is unpacked to disk.
  * `--n-snr` limits the build to the first N scenarios (n_grid = [N, 1, 1])
    for testing.
  * The two passes are split into `--stage prt` (CPU: resample the particles)
    and `--stage diag` (GPU: AeroMeasures diagnostics; resumable).

The resampling RNG (`np.random.RandomState(12345)`) is consumed in the same
scenario/time order as the notebook, so the particle data should match the
original file exactly when all 1000 scenarios are built.

Usage (from anywhere; the script chdirs into `<code-dir>/notebooks`):
    python build_prtcls_h5.py [--code-dir .../partnn] --raw-dir .../raw/01_bwchisamp \
        --out .../data/02_bwchisamp_00.h5 \
        [--n-snr 10] [--stage all|prt|diag] [--n-readers 8] [--device cuda:0]
"""
import os
import sys
import time
import shutil
import argparse
import subprocess
from os.path import exists
from textwrap import dedent
from itertools import product
from concurrent.futures import ProcessPoolExecutor

my_parser = argparse.ArgumentParser()
my_parser.add_argument('--code-dir', type=str,
                       default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
my_parser.add_argument('--raw-dir', type=str, required=True)
my_parser.add_argument('--out', type=str, required=True)
my_parser.add_argument('--n-snr', type=int, default=None)
my_parser.add_argument('--stage', type=str, default='all', choices=['all', 'prt', 'diag'])
my_parser.add_argument('--n-readers', type=int, default=8)
my_parser.add_argument('--device', type=str, default=None)
# The yml's optics batch (`aeroopt/eval/bs: 64` populations) needs more than the
# 40 GB of an NVIDIA A100. Populations are independent, so this only changes memory use.
my_parser.add_argument('--opt-bs', type=int, default=None)
# Diameter histogram overrides for the diag stage (clean_coarse.py builds a PM10
# dataset with --n-bins 15 --diam-high 1e-5, the same 0.25-decade bins up to 10 um).
my_parser.add_argument('--n-bins', type=int, default=None)
my_parser.add_argument('--diam-high', type=float, default=None)
args = my_parser.parse_args()

code_dir = os.path.abspath(args.code_dir)
raw_dir = os.path.abspath(args.raw_dir)
h5_pathout = os.path.abspath(args.out)
assert os.path.isfile(f'{code_dir}/notebooks/21_data/01_datasets.yml'), code_dir

# The notebook loads its config relative to `notebooks/`.
os.chdir(f'{code_dir}/notebooks')
sys.path.insert(0, code_dir)

import h5py
import torch
import numpy as np
from netCDF4 import Dataset
from ruamel import yaml as ruyaml
from collections import OrderedDict as odict
from partnn.aerometrics import AeroMeasures
from partnn.io_utils import eval_formula, augmnt_yamlhdf
from partnn.io_utils import parse_refs, get_subdict, hie2deep

fam_data = '05_bwchi'
my_rank, n_workers = 0, 1


# ---------------------------------------------------------------------------
# n21 cell 5: verbatim
# ---------------------------------------------------------------------------
def get_partdata(nc_fp, nc_path, snridx_nc, chem_species_keep, load_arrays=True, lib='numpy'):
    if snridx_nc in (None, 'all'):
        i_setstrt, i_setend = 0, None
    else:
        raise NotImplementedError('only the bwchi (snridx_nc="all") path is transcribed')

    n_chem_og, = nc_fp.variables['aero_species'].shape
    chem_species_ogstr = nc_fp.variables['aero_species'].names
    assert isinstance(chem_species_ogstr, str)

    chem_species_og = chem_species_ogstr.split(',')
    assert len(chem_species_og) == n_chem_og

    assert isinstance(chem_species_keep, str)
    if chem_species_keep == 'all':
        n_chem = n_chem_og
        chem_species = chem_species_og
        i_chemkeep = slice(None, None, None)
    else:
        chem_species = chem_species_keep.split(',')
        n_chem = len(chem_species)
        for chm in chem_species:
            assert chm in chem_species_og, dedent(f'''
                The "{chm}" chemical specie could not be found in the input
                file's chemical species:
                    available chemical species: "{chem_species_ogstr}"''')
        i_chemkeep = [chem_species_og.index(chm) for chm in chem_species]

    chem_species_str = ','.join(chem_species)
    n_part = nc_fp.variables['aero_particle_mass'].shape[1]

    if load_arrays:
        m_chmprt = nc_fp.variables['aero_particle_mass'][i_chemkeep, i_setstrt:i_setend]
        assert m_chmprt.shape == (n_chem, n_part)
        assert m_chmprt.dtype == np.float64, m_chmprt.dtype
        m_prtchm = m_chmprt.T
        assert m_prtchm.shape == (n_part, n_chem)

        rho_chm = nc_fp.variables['aero_density'][i_chemkeep].astype(np.float64)
        assert rho_chm.shape == (n_chem,)

        n_prt = nc_fp.variables['aero_num_conc'][i_setstrt:i_setend]
        assert n_prt.shape == (n_part,)
        assert n_prt.dtype == np.float64, n_prt.dtype
        assert lib == 'numpy'
    else:
        m_prtchm, n_prt, rho_chm = None, None, None

    out_dict = dict(m_prtchmog=m_prtchm, n_prtog=n_prt, rho_chm=rho_chm,
        n_partog=n_part, n_chem=n_chem, chem_species_str=chem_species_str)

    return out_dict

def rsmpl_partdata(m_prtchmog, n_prtog, n_partog, n_chem, n_part, np_random):
    assert m_prtchmog.shape == (n_partog, n_chem)
    assert n_prtog.shape == (n_partog,)

    if n_part < n_partog:
        i_selpart = np.argsort(np_random.rand(n_partog))[:n_part]
        assert i_selpart.shape == (n_part,)
        w_rptprts = 1
    elif n_part == n_partog:
        i_selpart = slice(None)
        w_rptprts = 1
    elif n_part > n_partog:
        i_selpart = (np.arange(n_part) % n_partog).astype(int)
        n_repeats = 1 + (n_part - 1 - i_selpart) // n_partog
        w_rptprts = 1.0 / n_repeats
    else:
        raise ValueError(f'this should be unreachable!')

    m_prtchm = m_prtchmog[i_selpart]
    assert m_prtchm.shape == (n_part, n_chem)
    n_prt_ = n_prtog[i_selpart]
    assert n_prt_.shape == (n_part,)
    w_impsamp = n_partog / n_part
    n_prt = n_prt_ * (w_rptprts * w_impsamp)
    assert n_prt.shape == (n_part,)

    return dict(m_prtchm=m_prtchm, n_prt=n_prt)


# ---------------------------------------------------------------------------
# Reading a scenario straight out of its tarball (no unpacking to disk)
# ---------------------------------------------------------------------------
def read_scenario(raw_dir, scnr_idx, n_t, chem_species_keep):
    """
    Streams `<raw_dir>/scenario_N.tar.gz` and applies `get_partdata` to each
    of its `n_t` netCDF files in memory. Runs in a worker process.
    """
    import tarfile
    tgz = f'{raw_dir}/scenario_{scnr_idx}.tar.gz'
    nc_names = {f'scenario_{scnr_idx}/scenario_{scnr_idx}_0001_000000{t_idx+1:02d}.nc': t_idx
        for t_idx in range(n_t)}
    gz_cmd = ['pigz', '-dc', tgz] if shutil.which('pigz') else ['gzip', '-dc', tgz]
    proc = subprocess.Popen(gz_cmd, stdout=subprocess.PIPE)
    part_datas = dict()
    with tarfile.open(fileobj=proc.stdout, mode='r|') as tf:
        for member in tf:
            if member.name not in nc_names:
                continue
            t_idx = nc_names[member.name]
            buf = tf.extractfile(member).read()
            with Dataset(member.name, "r", memory=buf) as nc_fp:
                part_data = get_partdata(nc_fp, f'{tgz}:{member.name}', 'all',
                    chem_species_keep, load_arrays=True, lib='numpy')
            for key in ('m_prtchmog', 'n_prtog', 'rho_chm'):
                assert not np.ma.is_masked(part_data[key]), f'{tgz}:{member.name} {key}'
                part_data[key] = np.ma.getdata(part_data[key])
            part_datas[t_idx] = part_data
    assert proc.wait() == 0, f'decompressing {tgz} failed'
    assert sorted(part_datas) == list(range(n_t)), dedent(f'''
        The BlueWater chi-sampling netCDF data files are missing:
            tgz = {tgz}
            missing t_idx = {sorted(set(range(n_t)) - set(part_datas))}''')
    return [part_datas[t_idx] for t_idx in range(n_t)]


# ---------------------------------------------------------------------------
# n21 cells 8 and 10: the data family config
# ---------------------------------------------------------------------------
with open('./21_data/01_datasets.yml', "r") as fp:
    info_fams1 = dict(ruyaml.safe_load(fp))
info_fams2 = parse_refs(info_fams1, trnsfrmtn='hie',
    pathsep=' -> ', cfg_path='21_data/01_datasets.yml')
# `inp/nc_pathfn` stays an unevaluated string; `read_scenario` replaces `get_bwncpath`.
tnsltns = {'all': 'all'}
info_fams = {key: eval_formula(val, tnsltns, catch=True) for key, val in info_fams2.items()}

device_name = args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu')
tch_device = torch.device(device_name)
tch_dtype = torch.float64
tch_cdtype = torch.complex128

info_fam = get_subdict(info_fams, fam_data)
n_t_all = info_fam['n_t']
n_grid = info_fam['n_grid']
data_tree = info_fam['inp/data_tree']
n_part = info_fam['n_part']
chem_species_keep = info_fam['chem_species']
subtitle = info_fam['subtitle']
if args.n_snr is not None:
    assert 0 < args.n_snr <= n_grid[0]
    n_grid = [args.n_snr, 1, 1]
n_grid = tuple(n_grid)
os.makedirs(os.path.dirname(h5_pathout), exist_ok=True)
rng_pathout = f'{h5_pathout}.rngstate.npz'

n_grid_x, n_grid_y, n_grid_z = n_grid
snr_idxs = list(range(n_grid_x * n_grid_y * n_grid_z))
t_idxs = list(range(n_t_all))
my_snridxs = np.array_split(snr_idxs, n_workers)[my_rank].tolist()
my_snridxsset = set(my_snridxs)
n_snr, n_mysnr, n_t = len(snr_idxs), len(my_snridxs), len(t_idxs)
n_mypop = n_mysnr * n_t

prod_itrtr = product(range(n_grid_x), range(n_grid_y), range(n_grid_z))
my_targets_ = [(scnr_idx, i_x, i_y, i_z)
    for scnr_idx, (i_x, i_y, i_z) in enumerate(prod_itrtr)
    if scnr_idx in my_snridxsset]
my_targets = [(i_snridx, scnr_idx, i_x, i_y, i_z)
    for i_snridx, (scnr_idx, i_x, i_y, i_z) in enumerate(my_targets_)]
print(f'family={fam_data} n_grid={n_grid} n_t={n_t} n_part={n_part} '
      f'n_pop={n_mypop} device={device_name}', flush=True)
print(f'out={h5_pathout}', flush=True)


# ---------------------------------------------------------------------------
# Stage "prt": n21 cells 11, 13, and 16 (netCDF -> resampled particles)
# ---------------------------------------------------------------------------
def build_prt():
    np_random = np.random.RandomState(seed=12345)

    # Resuming: the RNG state is saved after each finished scenario so that a
    # restarted build consumes the same random numbers as an uninterrupted one.
    n_snrdone = 0
    if exists(rng_pathout) and exists(h5_pathout):
        rng_npz = np.load(rng_pathout, allow_pickle=True)
        n_snrdone = int(rng_npz['n_snrdone'])
        np_random.set_state(tuple(rng_npz['state']))
        print(f'resuming after {n_snrdone} finished scenarios', flush=True)
    elif exists(h5_pathout):
        raise RuntimeError(f'{h5_pathout} exists without {rng_pathout}; move it away first')
    if n_snrdone == n_mysnr:
        print('stage prt: already done', flush=True)
        return

    # Scenarios are read in parallel, but resampled strictly in order so the
    # RNG stream matches the notebook.
    start_time = time.perf_counter()
    pool = ProcessPoolExecutor(max_workers=args.n_readers)
    todo = my_targets[n_snrdone:]
    futures = odict()
    def submit_upto(i_todo):
        for _, scnr_idx, _, _, _ in todo[len(futures) + n_popped[0]:i_todo + 1]:
            futures[scnr_idx] = pool.submit(read_scenario, raw_dir, scnr_idx, n_t, chem_species_keep)
    n_popped = [0]
    n_ahead = args.n_readers + 2
    submit_upto(n_ahead)

    # Getting the number of chemicals from the first scenario (n21 cell 11)
    part_info = futures[todo[0][1]].result()[0]
    n_chem = part_info['n_chem']
    chem_species_str = part_info['chem_species_str']

    # Initializing the output HDF file (n21 cell 13)
    h5fp_out = h5py.File(h5_pathout, "a")
    dims_prtdict = {
        'n_snr': n_mysnr,
        'n_snr_all': n_snr,
        'n_part': n_part,
        'n_t': n_t,
        'n_pop': n_mypop,
        'n_chem': n_chem,
        'n_grid_x': n_grid_x,
        'n_grid_y': n_grid_y,
        'n_grid_z': n_grid_z,
        'one': 1,
        'chem_species': chem_species_str}

    attrs_prtdictraw = get_subdict(info_fams, 'attrs/prt')
    attrs_prtdict = dict()
    for key in list(attrs_prtdictraw):
        if key.endswith('/val') or (key in ('subtitle')):
            val = eval_formula(attrs_prtdictraw[key],
                {**dims_prtdict, 'n_mysnr': n_mysnr, 'n_mypop': n_mypop,
                'subtitle': subtitle}, catch=True)
        else:
            val = attrs_prtdictraw[key]
        attrs_prtdict[key] = val

    augmnt_yamlhdf(dims_prtdict, h5fp_out, 'dims')
    augmnt_yamlhdf(attrs_prtdict, h5fp_out, 'attrs')

    for key in ['m_prtchm', 'n_prt', 'w_prt', 'n_tot', 'i_snr', 'i_x', 'i_y', 'i_z', 'i_t']:
        key_attrs = get_subdict(attrs_prtdict, key)
        if key not in h5fp_out:
            key_chunks = tuple(key_attrs['chunks/val'])
            assert all(isinstance(v, int) for v in key_chunks)
            key_shape = key_attrs['shape/val']
            assert all(isinstance(v, int) for v in key_shape)
            # Small test builds (--n-snr < 4) have fewer than 100 populations.
            key_chunks = tuple(min(c, s) for c, s in zip(key_chunks, key_shape))
            h5fp_out.create_dataset(key,
                shape=key_shape, chunks=key_chunks,
                dtype=key_attrs['dtype'], fillvalue=-1)

    # The main netCDF to HDF transport (n21 cell 16)
    for i_todo, (i_snridx, scnr_idx, i_x, i_y, i_z) in enumerate(todo):
        submit_upto(i_todo + n_ahead)
        part_datas = futures.pop(scnr_idx).result()
        n_popped[0] += 1
        for t_idx in t_idxs:
            i_pop = i_snridx * n_t + t_idx
            part_data = part_datas[t_idx]

            n_partog = part_data['n_partog']
            assert part_data['n_chem'] == n_chem
            assert chem_species_str == part_data['chem_species_str']
            m_prtchmog = part_data['m_prtchmog']
            assert m_prtchmog.shape == (n_partog, n_chem)
            n_prtog = part_data['n_prtog']
            assert n_prtog.shape == (n_partog,)

            rsmpld_data = rsmpl_partdata(m_prtchmog, n_prtog, n_partog, n_chem, n_part, np_random)
            m_prtchm = rsmpld_data['m_prtchm']
            assert m_prtchm.shape == (n_part, n_chem)
            n_prt = rsmpld_data['n_prt'].reshape(n_part, 1)
            w_prt = n_prt / n_prt.sum(axis=-2, keepdims=True)
            n_tot = n_prt.sum(axis=-2, keepdims=True)
            assert n_tot.shape == (1, 1)

            h5fp_out['m_prtchm'][i_pop, :, :] = m_prtchm
            h5fp_out['n_prt'][i_pop, :, :] = n_prt
            h5fp_out['w_prt'][i_pop, :, :] = w_prt
            h5fp_out['n_tot'][i_pop, :, :] = n_tot
            h5fp_out['i_snr'][i_pop] = scnr_idx
            h5fp_out['i_x'][i_pop] = i_x
            h5fp_out['i_y'][i_pop] = i_y
            h5fp_out['i_z'][i_pop] = i_z
            h5fp_out['i_t'][i_pop] = t_idx

        del part_datas
        h5fp_out.flush()
        np.savez(rng_pathout, n_snrdone=i_snridx + 1,
            state=np.array(np_random.get_state(), dtype=object))
        if (i_snridx + 1) % 10 == 0 or (i_snridx + 1) == n_mysnr:
            el = time.perf_counter() - start_time
            print(f'  prt: {i_snridx + 1}/{n_mysnr} scenarios, {el:.0f} s', flush=True)

    pool.shutdown()
    h5fp_out.close()
    print(f'stage prt: done in {time.perf_counter() - start_time:.1f} s', flush=True)


# ---------------------------------------------------------------------------
# Stage "diag": n21 cells 20 and 22 (AeroMeasures diagnostics)
# ---------------------------------------------------------------------------
def build_diag():
    h5fp_out = h5py.File(h5_pathout, "a")
    assert (h5fp_out['i_snr'][:] >= 0).all(), 'stage prt is not finished'
    dims_disk = ruyaml.safe_load(h5fp_out['dims'][0].decode('utf-8'))
    n_chem = dims_disk['n_chem']
    chem_species_lst = dims_disk['chem_species'].split(',')
    assert dims_disk['n_part'] == n_part

    diag_hparams1 = get_subdict(info_fam, 'diag')
    n_mbdiag = diag_hparams1.pop('n_mb')
    if args.opt_bs is not None:
        assert 'hparams/aeroopt/eval/bs' in diag_hparams1
        diag_hparams1['hparams/aeroopt/eval/bs'] = args.opt_bs
    for key, val in [('n_bins', args.n_bins), ('diam_high', args.diam_high)]:
        if val is not None:
            assert f'hparams/aerohst/{key}' in diag_hparams1
            diag_hparams1[f'hparams/aerohst/{key}'] = val
    tnsltn_vars = dict(n_chem=n_chem, n_part=n_part, chem_species=chem_species_lst)
    diag_hparams2 = {key: eval_formula(val, tnsltn_vars, catch=True)
        for key, val in diag_hparams1.items()}
    diag_calcltr = AeroMeasures(diag_hparams2, tch_device, tch_dtype, tch_cdtype)

    diag_shapevars1 = diag_calcltr.get_shapevars()
    diag_shapevars2 = {key: val for key, val in diag_shapevars1.items() if val is not None}

    attrs_dictraw = get_subdict(info_fams, 'attrs')
    get_subdict(attrs_dictraw, 'prt', pop=True)
    attrs_dict = dict()
    for key in list(attrs_dictraw):
        if key.endswith('/val'):
            val = eval_formula(attrs_dictraw[key],
                {**diag_shapevars1, 'n_mysnr': n_mysnr, 'n_mypop': n_mypop,
                'n_chem': n_chem, 'n_t': n_t}, catch=True)
        else:
            val = attrs_dictraw[key]
        attrs_dict[key] = val

    diag_types = diag_calcltr.pp_config['hparams/output/type']
    diag_keys = diag_calcltr.pp_config['hparams/measures']
    augmnt_yamlhdf(diag_shapevars2, h5fp_out, 'dims', do_raise=True)
    pp_config = diag_calcltr.pp_config
    for diag_type in diag_types:
        attrs_diagdicthie = get_subdict(attrs_dict, diag_type)
        augmnt_yamlhdf(attrs_diagdicthie, h5fp_out, 'attrs', do_raise=True)

        attrs_diagdict = hie2deep(attrs_diagdicthie, maxdepth=1)
        for key, key_attrs in attrs_diagdict.items():
            if key not in h5fp_out:
                key_chunks = tuple(key_attrs['chunks/val'])
                assert all(isinstance(v, int) for v in key_chunks)
                key_shape = key_attrs['shape/val']
                assert all(isinstance(v, int) for v in key_shape)
                key_chunks = tuple(min(c, s) for c, s in zip(key_chunks, key_shape))
                h5fp_out.create_dataset(key,
                    shape=key_shape, chunks=key_chunks,
                    dtype=key_attrs['dtype'], fillvalue=-12345)

            if key == 'diam_low':
                h5fp_out[key][:] = pp_config['hparams/aerohst/diam_low']
            elif key == 'diam_high':
                h5fp_out[key][:] = pp_config['hparams/aerohst/diam_high']
            elif key == 'diam_logbase':
                h5fp_out[key][:] = pp_config['hparams/aerohst/diam_logbase']
            elif key == 'chem_rho':
                h5fp_out[key][:] = pp_config['hparams/aerocst/rho_chm'].detach().cpu().numpy()
            elif key == 'chem_kappa':
                h5fp_out[key][:] = pp_config['hparams/aerocst/kappa_chm'].detach().cpu().numpy()
            elif key == 'wave_len':
                h5fp_out[key][:] = pp_config['hparams/aerocst/len_wv'].detach().cpu().numpy()
            elif key == 'chem_refr_real':
                h5fp_out[key][:] = pp_config['hparams/aerocst/refr_wvchm'].detach().cpu().numpy().real
            elif key == 'chem_refr_imag':
                h5fp_out[key][:] = pp_config['hparams/aerocst/refr_wvchm'].detach().cpu().numpy().imag

    # The diagnostics calculation (n21 cell 22)
    start_time = time.perf_counter()
    m_popprtchmh5 = h5fp_out['m_prtchm']
    assert m_popprtchmh5.shape == (n_mypop, n_part, n_chem)
    n_popprth5 = h5fp_out['n_prt']
    assert n_popprth5.shape == (n_mypop, n_part, 1)

    n_mbs = int(np.ceil(n_mypop / n_mbdiag))
    for ii_mb in range(n_mbs):
        i1_pop = ii_mb * n_mbdiag
        i2_pop = min(i1_pop + n_mbdiag, n_mypop)
        n_mb = i2_pop - i1_pop

        if all((h5fp_out[key][i1_pop: i2_pop] != -12345).all() for key in diag_keys):
            continue

        with torch.no_grad():
            m_mbprtchmnp = m_popprtchmh5[i1_pop:i2_pop][None]
            m_mbprtchm = torch.from_numpy(m_mbprtchmnp).to(device=tch_device, dtype=tch_dtype)
            assert m_mbprtchm.shape == (1, n_mb, n_part, n_chem)
            n_mbprtnp = n_popprth5[i1_pop:i2_pop][None]
            n_mbprt = torch.from_numpy(n_mbprtnp).to(device=tch_device, dtype=tch_dtype)
            assert n_mbprt.shape == (1, n_mb, n_part, 1)

            inputs_mb = {'m_prtchm': m_mbprtchm, 'n_prt': n_mbprt}
            diag_data, _ = diag_calcltr.forward(inputs_mb)

        for key, val in diag_data.items():
            h5fp_out[key][i1_pop: i2_pop, ...] = val.squeeze(dim=0).detach().cpu().numpy()
        h5fp_out.flush()
        el = time.perf_counter() - start_time
        mem = torch.cuda.max_memory_allocated() / 1e9 if tch_device.type == 'cuda' else 0
        print(f'  diag: minibatch {ii_mb + 1}/{n_mbs}, {el:.0f} s, peak {mem:.1f} GB', flush=True)

    h5fp_out.close()
    print(f'stage diag: done in {time.perf_counter() - start_time:.1f} s', flush=True)


if args.stage in ('all', 'prt'):
    build_prt()
if args.stage in ('all', 'diag'):
    build_diag()
