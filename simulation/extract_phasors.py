import os
import sys
import glob
import time

import numpy as np
import pandas as pd

UNDEF = 1e29
DEFAULT_OUT = "phasors.npz"

def extract(path):
    d = pd.read_csv(path, header=None).to_numpy()
    d = d[d[:, 5] < UNDEF]
    if len(d) == 0:
        raise ValueError("no valid value")
    t, nid, x, y, z, T = d.T
    tu = np.unique(t)
    f = 1.0 / (16.0 * np.median(np.diff(tu)))
    ids = np.unique(nid)
    it = np.searchsorted(tu, t)
    ii = np.searchsorted(ids, nid)
    M = np.full((len(tu), len(ids)), np.nan)
    M[it, ii] = T
    if np.isnan(M).any():
        raise ValueError("incomplete time-node matrix")
    XYZ = np.zeros((len(ids), 3))
    XYZ[ii] = np.c_[x, y, z]

    A = np.c_[np.ones_like(tu), tu, tu**2, tu**3,
              np.cos(2*np.pi*f*tu), np.sin(2*np.pi*f*tu)]
    c, *_ = np.linalg.lstsq(A, M, rcond=None)
    return f, ids, XYZ, c[4] - 1j*c[5], M.mean(axis=0)

def main():
    if len(sys.argv) < 2:
        print("usage: python extract_phasors.py <csv_directory> [output.npz]")
        return
    dirpath = sys.argv[1]
    outpath = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_OUT
    files = sorted(glob.glob(os.path.join(dirpath, "*.csv")))
    print(f"{len(files)} files in {dirpath}\n")
    store, t0 = {}, time.time()
    for path in files:
        key = os.path.basename(path)[:-4]
        try:
            f, ids, XYZ, ph, dc = extract(path)
        except Exception as e:
            print(f"{key:12s} SKIPPED: {e}")
            continue
        store[f"{key}_f"] = f
        store[f"{key}_ids"] = ids.astype(np.int32)
        store[f"{key}_xyz"] = XYZ.astype(np.float32)
        store[f"{key}_ph"] = ph.astype(np.complex64)
        store[f"{key}_dc"] = dc.astype(np.float32)
        print(f"{key:12s} f={f:8.5f} Hz  {len(ids):5d} nodes  "
              f"|A|max {np.abs(ph).max()*1e3:8.2f} mK")
    np.savez_compressed(outpath, **store)
    mb = os.path.getsize(outpath) / 1e6
    print(f"\nwritten {outpath}  ({mb:.1f} MB, {time.time()-t0:.0f} s)")
    print("This is the file to share, not the CSVs.")

if __name__ == "__main__":
    main()
