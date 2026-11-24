#!/usr/bin/env python
"""
Make a PM10 version of the AeroMELD particle dataset: copy the particle data
and replace every resampled particle with a wet diameter of 10 um or more by
a random particle under 10 um from the same PartMC population.

The 1000 particles of each population are a uniform random subsample of the
~180,000 PartMC particles, all with the same number weight (n_partog / 1000
times the PartMC weight). Particles over 10 um are very rare in the PartMC
populations, but a population that happens to draw one gets a coarse dust or
water particle that carries 50-99.99% of its mass and changes the scattering
and frozen fraction by large factors, and some of these particles (all over
20 um) also broke the Mie solver. There are 118 such particles in 117 of the
25000 populations. The replacement keeps the number weights (they are equal
within a population), so n_prt, w_prt, and n_tot are unchanged.

Only the particle stage is written here (m_prtchm, n_prt, w_prt, n_tot, the
indices, and the particle-stage dims and attrs). The diagnostics are then
recomputed for all populations on 15 bins up to 10 um with
    build_prtcls_h5.py --stage diag --n-bins 15 --diam-high 1e-5 --out OUT
The report (OUT.coarse.npz) lists the replaced particles and, for every time
of the scenarios that were read, the number and mass fractions of the PartMC
particles over 10 um.

Usage (from anywhere, CPU only):
    python clean_coarse.py SRC.h5 OUT.h5 --raw-dir .../raw/01_bwchisamp [--n-readers 16]
"""
import os
import shutil
import tarfile
import argparse
import subprocess
from concurrent.futures import ProcessPoolExecutor
import h5py
import numpy as np
import yaml
from netCDF4 import Dataset

my_parser = argparse.ArgumentParser()
my_parser.add_argument('src')
my_parser.add_argument('out')
my_parser.add_argument('--raw-dir', required=True)
my_parser.add_argument('--dmax', type=float, default=1e-5, help='largest kept wet diameter (m)')
my_parser.add_argument('--n-readers', type=int, default=16)
my_parser.add_argument('--seed', type=int, default=20261016)
my_parser.add_argument('--chunk', type=int, default=1000)
args = my_parser.parse_args()

PRT_KEYS = ['m_prtchm', 'n_prt', 'w_prt', 'n_tot', 'i_snr', 'i_x', 'i_y', 'i_z', 'i_t']
PRT_DIMS = ['n_snr', 'n_snr_all', 'n_part', 'n_t', 'n_pop', 'n_chem',
            'n_grid_x', 'n_grid_y', 'n_grid_z', 'one', 'chem_species']


def diameters(m_prtchm, rho):
    """Wet volume-equivalent diameters (m) of particles (..., n_chem), as in AeroMeasures."""
    return np.cbrt(6 * (m_prtchm / rho).sum(axis=-1) / np.pi)


def read_scenario(raw_dir, scnr_idx, n_t, species, rho, dmax, t_keep):
    """
    Streams one scenario tarball. Returns, for every time, the number of PartMC
    particles, the number and mass fractions of those at or above `dmax`, and,
    for the times in `t_keep` only, the particles themselves (m_prtchm, n_prt).
    """
    tgz = f'{raw_dir}/scenario_{scnr_idx}.tar.gz'
    names = {f'scenario_{scnr_idx}/scenario_{scnr_idx}_0001_000000{t + 1:02d}.nc': t for t in range(n_t)}
    gz_cmd = ['pigz', '-dc', tgz] if shutil.which('pigz') else ['gzip', '-dc', tgz]
    proc = subprocess.Popen(gz_cmd, stdout=subprocess.PIPE)
    stats, parts, rho_og = dict(), dict(), None
    with tarfile.open(fileobj=proc.stdout, mode='r|') as tf:
        for member in tf:
            if member.name not in names:
                continue
            t = names[member.name]
            buf = tf.extractfile(member).read()
            with Dataset(member.name, 'r', memory=buf) as nc:
                sp_og = nc.variables['aero_species'].names.split(',')
                i_keep = [sp_og.index(s) for s in species]
                m = np.ma.getdata(nc.variables['aero_particle_mass'][i_keep, :]).T.astype(np.float64)
                n = np.ma.getdata(nc.variables['aero_num_conc'][:]).astype(np.float64)
                rho_og = np.ma.getdata(nc.variables['aero_density'][i_keep]).astype(np.float64)
            coarse = diameters(m, rho) >= dmax
            mass = m.sum(axis=-1) * n
            stats[t] = (len(n), int(coarse.sum()), n[coarse].sum() / n.sum(), mass[coarse].sum() / mass.sum())
            if t in t_keep:
                parts[t] = (m, n)
    proc.stdout.close()
    assert proc.wait() == 0, f'decompressing {tgz} failed'
    assert sorted(stats) == list(range(n_t)), f'{tgz}: missing times {sorted(set(range(n_t)) - set(stats))}'
    return stats, parts, rho_og


with h5py.File(args.src, 'r') as src:
    rho = src['chem_rho'][:]
    dims = yaml.safe_load(src['dims'][0].decode())
    attrs = yaml.safe_load(src['attrs'][0].decode())
    species = dims['chem_species'].split(',')
    n_pop, n_part = src['m_prtchm'].shape[:2]
    n_t = dims['n_t']
    i_snr, i_t = src['i_snr'][:], src['i_t'][:]

    # The populations with a particle at or above dmax
    big = np.zeros((n_pop, n_part), dtype=bool)
    for i0 in range(0, n_pop, args.chunk):
        big[i0:i0 + args.chunk] = diameters(src['m_prtchm'][i0:i0 + args.chunk], rho) >= args.dmax
    pops = np.nonzero(big.any(axis=1))[0]
    print(f'{big.sum()} particles >= {args.dmax * 1e6:g} um in {len(pops)} populations '
          f'of {len(np.unique(i_snr[pops]))} scenarios', flush=True)

    # Copying the particle stage, with the particle-stage dims and attrs only, so
    # that the diag stage can add its own (and a different number of bins).
    assert not os.path.exists(args.out), f'{args.out} exists'
    with h5py.File(args.out, 'w') as out:
        for key in PRT_KEYS:
            d = src[key]
            out.create_dataset(key, shape=d.shape, dtype=d.dtype, chunks=d.chunks, fillvalue=d.fillvalue)
            for i0 in range(0, d.shape[0], args.chunk):
                out[key][i0:i0 + args.chunk] = d[i0:i0 + args.chunk]
        dims_prt = {k: dims[k] for k in PRT_DIMS}
        attrs_prt = {k: v for k, v in attrs.items() if k.split('/')[0] in PRT_KEYS + ['title', 'subtitle']}
        for key, val in [('dims', dims_prt), ('attrs', attrs_prt)]:
            out.create_dataset(key, shape=1, dtype=h5py.string_dtype(), data=[yaml.safe_dump(val)])

# Reading the scenarios of those populations
snr_pops = dict()
for i in pops:
    snr_pops.setdefault(int(i_snr[i]), []).append(int(i))
rng = np.random.RandomState(args.seed)
rep = {k: [] for k in ['pop', 'prt', 'd_old', 'd_new', 'mfrac_old']}
stats = {k: [] for k in ['snr', 't', 'n_partog', 'n_coarse', 'nfrac_coarse', 'mfrac_coarse']}
with h5py.File(args.out, 'r+') as out, ProcessPoolExecutor(args.n_readers) as pool:
    futures = {s: pool.submit(read_scenario, args.raw_dir, s, n_t, species, rho, args.dmax,
                              {int(i_t[i]) for i in ii}) for s, ii in snr_pops.items()}
    for n_done, (s, ii) in enumerate(snr_pops.items()):
        snr_stats, parts, rho_og = futures.pop(s).result()
        assert np.allclose(rho_og, rho), (s, rho_og, rho)
        for t, st in snr_stats.items():
            for k, v in zip(['snr', 't', 'n_partog', 'n_coarse', 'nfrac_coarse', 'mfrac_coarse'], (s, t, *st)):
                stats[k].append(v)
        for i_pop in ii:
            m_og, n_og = parts[int(i_t[i_pop])]
            n_partog = len(n_og)
            d_og = diameters(m_og, rho)
            m, n_prt = out['m_prtchm'][i_pop], out['n_prt'][i_pop]
            # Locating the resampled particles in the PartMC population
            og_rows = {r.tobytes(): j for j, r in enumerate(m_og)}
            sel = np.array([og_rows.get(r.tobytes(), -1) for r in m])
            assert (sel >= 0).all(), f'population {i_pop}: resampled particles not in the PartMC data'
            assert np.array_equal(n_prt[:, 0], n_og[sel] * (1 * (n_partog / n_part))), i_pop
            cand = np.setdiff1d(np.nonzero(d_og < args.dmax)[0], sel)
            jbig = np.nonzero(big[i_pop])[0]
            new = rng.choice(cand, len(jbig), replace=False)
            mfrac = (m[jbig].sum(-1) * n_prt[jbig, 0]).sum() / (m.sum(-1) * n_prt[:, 0]).sum()
            for j, k in zip(jbig, new):
                rep['pop'].append(i_pop), rep['prt'].append(j), rep['mfrac_old'].append(mfrac)
                rep['d_old'].append(diameters(m[j], rho)), rep['d_new'].append(d_og[k])
                m[j] = m_og[k]
                # The same weight formula as rsmpl_partdata (the PartMC weights are equal in practice)
                n_prt[j, 0] = n_og[k] * (1 * (n_partog / n_part))
            out['m_prtchm'][i_pop] = m
            out['n_prt'][i_pop] = n_prt
            out['w_prt'][i_pop] = n_prt / n_prt.sum(axis=-2, keepdims=True)
            out['n_tot'][i_pop] = n_prt.sum(axis=-2, keepdims=True)
        if (n_done + 1) % 10 == 0 or n_done + 1 == len(snr_pops):
            print(f'  {n_done + 1}/{len(snr_pops)} scenarios', flush=True)

    # Checks: every particle is now under dmax, and only the flagged particles changed
    with h5py.File(args.src, 'r') as src:
        n_changed, ntot_relchg = 0, 0.0
        for i0 in range(0, n_pop, args.chunk):
            a, b = src['m_prtchm'][i0:i0 + args.chunk], out['m_prtchm'][i0:i0 + args.chunk]
            assert (diameters(b, rho) < args.dmax).all()
            changed = (a != b).any(axis=-1)
            assert (changed == big[i0:i0 + args.chunk]).all()
            n_changed += changed.sum()
            na, nb = src['n_tot'][i0:i0 + args.chunk], out['n_tot'][i0:i0 + args.chunk]
            ntot_relchg = max(ntot_relchg, float(np.abs(nb / na - 1).max()))
    print(f'replaced {n_changed} particles; all particles now < {args.dmax * 1e6:g} um; '
          f'max relative change of n_tot {ntot_relchg:.3g}')

rep = {k: np.array(v) for k, v in rep.items()}
stats = {k: np.array(v) for k, v in stats.items()}
np.savez(f'{args.out}.coarse.npz', pops=pops, dmax=args.dmax, **rep, **{f'snr_{k}': v for k, v in stats.items()})
q = lambda x: np.quantile(x, [0, 0.5, 0.9, 1])
print('quantiles 0/50/90/100 %:')
print('  replaced-particle diameters (um), old:', q(rep['d_old'] * 1e6), ' new:', q(rep['d_new'] * 1e6))
print('  their mass fraction of the resampled population:', q(rep['mfrac_old']))
print(f'  in the PartMC populations of the {len(snr_pops)} scenarios (all times):')
print('    particles >= dmax:', q(stats['n_coarse']), ' of', q(stats['n_partog']))
print('    their number fraction:', q(stats['nfrac_coarse']), ' mass fraction:', q(stats['mfrac_coarse']))
print('wrote', args.out, f'and {args.out}.coarse.npz')
