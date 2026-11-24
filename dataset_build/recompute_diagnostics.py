#!/usr/bin/env python
"""
Corrected diagnostics for the training dataset:

  optics  partnn's get_bulkoptical with the geometric cross-section of the
          WHOLE particle, pi/4 d^2 (the dataset used pi/4 d_shell^2, the shell
          volume without BC and dust), n_max 220 and every other setting as the
          dataset build (21_data/01_datasets.yml); also the unmodified function on
          the first populations, which must reproduce the stored values.
  CCN     partnn's per-particle kappa-Koehler (get_apprxccncdf, 'prt') at
          287 K (as stored; a check) and at 290 K (the paper's PartMC side).

partnn is not modified: the corrected optics function is a copy patched in
memory. Output npz: qs_pop, qa_pop (n_pop, 8), ccn_cdf_287, ccn_cdf_290
(n_pop, 100), and the checks.

Usage (GPU node, from anywhere):
    python recompute_diagnostics.py DATA.h5 OUT.npz [--bs 32] [--n-check 256]
"""
import argparse
import inspect
import os
import sys
import time

import h5py
import numpy as np
import torch

ap = argparse.ArgumentParser()
ap.add_argument('data')
ap.add_argument('out')
ap.add_argument('--bs', type=int, default=32)
ap.add_argument('--n-check', type=int, default=256)
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import partnn.aerometrics as AM  # noqa: E402

src = inspect.getsource(AM.get_bulkoptical)
OLD = 'crossec_prt = d_prtshll.square() * (math.pi / 4)'
assert src.count(OLD) == 1, 'partnn get_bulkoptical changed; check before using this script'
ns = dict(vars(AM))
exec(src.replace(OLD, 'crossec_prt = d_prt.square() * (math.pi / 4)').replace('def get_bulkoptical', 'def fixed'), ns)
fixed = ns['fixed']

dev = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
f = h5py.File(args.data, 'r')
ML = ['SO4', 'NO3', 'Cl', 'NH4', 'ARO1', 'ARO2', 'ALK1', 'OLE1', 'API1', 'Na', 'OIN', 'OC', 'BC', 'MOC', 'H2O']
rho = torch.tensor(f['chem_rho'][()], device=dev)
kap = torch.tensor(f['chem_kappa'][()], device=dev)
refr = torch.tensor(f['chem_refr_real'][()] + 1j * f['chem_refr_imag'][()], device=dev)      # (wave, chem)
wl = torch.tensor(f['wave_len'][()], device=dev)
n_pop = f['n_prt'].shape[0]
qs, qa = np.zeros((n_pop, 8)), np.zeros((n_pop, 8))
c287, c290 = np.zeros((n_pop, 100)), np.zeros((n_pop, 100))
chk_opt, chk_ccn = [], []
t0 = time.time()
for i0 in range(0, n_pop, args.bs):
    i1 = min(i0 + args.bs, n_pop)
    nb = i1 - i0
    m = torch.tensor(f['m_prtchm'][i0:i1], device=dev)[None]                       # (1, nb, 1000, 15)
    n = torch.tensor(f['n_prt'][i0:i1], device=dev)[None]                          # (1, nb, 1000, 1)
    with torch.no_grad():
        oa = (m, n, rho, refr, wl, 220, complex(1.5, 0), ML, 1, nb, 15, 1000, 8, dev, torch.float64,
              torch.complex128)
        r = fixed(*oa, input_type='prt')
        qs[i0:i1] = r['qs_pop'].reshape(nb, 8).cpu().numpy()
        qa[i0:i1] = r['qa_pop'].reshape(nb, 8).cpu().numpy()
        for T, out in ((287.0, c287), (290.0, c290)):
            c = AM.get_apprxccncdf(m, n, 1e-3, 1e-1, 100, T, rho, kap, ML, 0.0, 150, 1e-12, 1, nb, 15, 1000,
                                   dev, torch.float64, input_type='prt', strict=False)
            out[i0:i1] = c['ccn_cdf'].reshape(nb, 100).cpu().numpy()
        if i0 < args.n_check:
            r0 = AM.get_bulkoptical(*oa, input_type='prt')
            s = f['qs_pop'][i0:i1].reshape(nb, 8)
            chk_opt.append(np.abs(r0['qs_pop'].reshape(nb, 8).cpu().numpy() / s - 1).max())
    if i0 < args.n_check:
        st = f['ccn_cdf'][i0:i1].reshape(nb, 100)
        chk_ccn.append(np.abs(c287[i0:i1] - st).max())
    if (i0 // args.bs) % 50 == 0:
        print('%d / %d populations, %.0f s' % (i1, n_pop, time.time() - t0), flush=True)
print('checks on the first %d populations: unmodified optics vs stored max rel diff %.2e; CCN 287 K vs stored '
      'max abs diff %.2e' % (args.n_check, max(chk_opt), max(chk_ccn)))
st_s, st_a = f['qs_pop'][:].reshape(n_pop, 8), f['qa_pop'][:].reshape(n_pop, 8)
np.savez(args.out, qs_pop=qs, qa_pop=qa, ccn_cdf_287=c287, ccn_cdf_290=c290, check_optics=np.array(chk_opt),
         check_ccn=np.array(chk_ccn), ratio_scat=qs / st_s, ratio_abs=qa / st_a, i_snr=f['i_snr'][:])
for name, r in (('scattering', qs / st_s), ('absorption', qa / st_a)):
    print('%s corrected / stored over %d populations: median %s, p90 %s, p99 %s'
          % (name, n_pop, np.round(np.median(r, 0), 3), np.round(np.percentile(r, 90, 0), 3),
             np.round(np.percentile(r, 99, 0), 3)))
print('CCN/CN 290 K - 287 K: mean %.4f, max %.4f' % ((c290 - c287).mean(), np.abs(c290 - c287).max()))
print('wrote', args.out, '%.0f s' % (time.time() - t0))
