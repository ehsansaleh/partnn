#!/usr/bin/env python
"""
A small text digest of an h5 file, for checking a copy or a rebuild of a large
file without moving it: one line per dataset with its shape, dtype, the sha1
of its bytes, and, for numeric data, the sum, min and max. Two digests are
compared with --compare: identical bytes, or the largest difference of the
sums, mins and maxes relative to the largest absolute value.

Usage:
    python h5_digest.py FILE.h5 > FILE.digest
    python h5_digest.py --compare A.digest B.digest
"""
import argparse
import hashlib
import sys
import h5py
import numpy as np

my_parser = argparse.ArgumentParser()
my_parser.add_argument('paths', nargs='+')
my_parser.add_argument('--compare', action='store_true')
args = my_parser.parse_args()


def digest(path):
    with h5py.File(path, 'r') as f:
        names = []
        f.visititems(lambda name, obj: names.append(name) if isinstance(obj, h5py.Dataset) else None)
        for name in sorted(names):
            d = f[name]
            sha, s, lo, hi = hashlib.sha1(), 0.0, np.inf, -np.inf
            numeric = d.dtype.kind in 'biuf'
            step = max(1, d.shape[0] // max(1, d.nbytes // 2**28)) if d.ndim else None
            for i0 in (range(0, d.shape[0], step) if d.ndim and d.shape[0] else [None]):
                x = d[()] if i0 is None else d[i0:i0 + step]
                x = np.asarray(x)
                if x.dtype.kind == 'O':
                    x = np.array([v if isinstance(v, bytes) else str(v).encode() for v in x.ravel()])
                sha.update(np.ascontiguousarray(x).tobytes())
                if numeric and x.size:
                    s += float(np.sum(x, dtype=np.float64))
                    lo, hi = min(lo, float(x.min())), max(hi, float(x.max()))
            stats = f' {s!r} {lo!r} {hi!r}' if numeric else ''
            print(f'{name} {d.shape} {d.dtype} {sha.hexdigest()}{stats}'.replace(', ', ','), flush=True)


def compare(a, b):
    def read(p):
        out = dict()
        for line in open(p):
            name, shape, dtype, sha, *stats = line.split()
            out[name] = (shape, dtype, sha, [float(v) for v in stats])
        return out
    A, B = read(a), read(b)
    for name in sorted(set(A) | set(B)):
        if name not in A or name not in B:
            print(f'{name}: only in {a if name in A else b}')
            continue
        (sa, ta, ha, xa), (sb, tb, hb, xb) = A[name], B[name]
        if (sa, ta) != (sb, tb):
            print(f'{name}: shape/dtype {sa} {ta} vs {sb} {tb}')
        elif ha == hb:
            print(f'{name}: identical')
        elif xa:
            scale = max(abs(xa[1]), abs(xa[2]), 1e-300)
            rel = max(abs(u - v) for u, v in zip(xa, xb)) / scale
            print(f'{name}: bytes differ; sum/min/max differ by {rel:.1e} of max |value|')
        else:
            print(f'{name}: bytes differ')


if args.compare:
    compare(*args.paths)
else:
    for p in args.paths:
        digest(p)
