import os
import sys
import numpy as np
import pandas as pd

from make_campaign import BATCHES, BANDS, WORKROOT, win, band

UNDEF = 1e29

NODE_FACTOR = 1.18

def load(path):
    d = pd.read_csv(path, header=None).to_numpy()
    t, nid, x, y, z, T = d.T
    ok = T < UNDEF

    missing = np.setdiff1d(np.unique(nid[~ok]), np.unique(nid[ok]))
    zp = np.array([])
    xyzp = np.zeros((0, 3))
    if len(missing):
        first = np.isin(nid, missing)
        _, idx = np.unique(nid[first], return_index=True)
        zp = z[first][idx]
        xyzp = np.c_[x[first][idx], y[first][idx], zp]
    tu = np.unique(t)
    f = 1.0 / (16.0 * np.median(np.diff(tu)))
    ids = np.unique(nid[ok])
    sub = d[ok]
    it = np.searchsorted(tu, sub[:, 0])
    ii = np.searchsorted(ids, sub[:, 1])
    M = np.full((len(tu), len(ids)), np.nan)
    M[it, ii] = sub[:, 5]
    XYZ = np.zeros((len(ids), 3))
    XYZ[ii] = sub[:, 2:5]
    A = np.c_[np.ones_like(tu), tu, tu**2, tu**3,
              np.cos(2*np.pi*f*tu), np.sin(2*np.pi*f*tu)]
    c, *_ = np.linalg.lstsq(A, M, rcond=None)
    return dict(f=f, ids=ids, xyz=XYZ, ph=c[4] - 1j*c[5],
                n_tot=len(np.unique(nid)), n_ok=len(ids), z_missing=zp,
                xyz_missing=xyzp,
                tmin=sub[:, 5].min(), tmax=sub[:, 5].max())

def read_meta(path):

    vals = {}
    with open(path) as fh:
        lines = [l.strip() for l in fh if l.strip()]
    for i in range(0, len(lines) - 1, 2):
        names = [n.strip() for n in lines[i].split(",")]
        nums = [float(v) for v in lines[i+1].split(",") if v.strip()]
        vals.update(dict(zip(names, nums)))
    return vals

def check_meta(c, freq, meta):

    expected = dict(TH=c["th"], LXP=c["lxp"], HD=c["hd"], HDEPTH=c["hdepth"],
                  HY0=c["hy0"], HY1=c["hy1"], SPOTY=c["spoty"],
                  DEFECT=c["defect"], ESZ=c["esz"], FSZ=c["fsz"], FREQ=freq)
    return [f"{k} = {meta[k]:g} instead of {v:g}, the run was executed with a "
            f".inp different from the one generated now"
            for k, v in expected.items()
            if k in meta and abs(meta[k] - v) > 1e-6 * max(abs(v), 1.0)]

def check_one(c, tag, freq, dirpath):
    runid = f"{c['id']}_{tag}"
    path = os.path.join(dirpath, runid + ".csv")
    if not os.path.exists(path):
        return runid, [], None, [], "not produced yet"

    try:
        r = load(path)
    except Exception as e:
        return runid, [f"unreadable: {type(e).__name__}: {e}"], None, [], None
    bad, warn = [], []

    mpath = os.path.join(dirpath, runid + ".meta")
    if os.path.exists(mpath):
        try:
            bad += check_meta(c, freq, read_meta(mpath))
        except Exception as e:
            warn.append(f".meta unreadable: {e}")
    else:
        warn.append("no .meta: the run comes from an earlier version of RUN_1F, "
                    "so its parameters cannot be verified")

    if abs(r["f"] - freq) / freq > 0.01:
        bad.append(f"frequency inferred as {r['f']:.4f} instead of {freq}")

    mf = np.abs(r["xyz"][:, 2] - c["th"]/1000.0) < 1e-6
    mb = np.abs(r["xyz"][:, 2]) < 1e-6
    nexp = NODE_FACTOR * c["lxp"] * (c.get("lyp") or 30) / (c["fsz"] ** 2)
    for name, m in (("excited", mf), ("rear", mb)):
        if m.sum() < 0.7 * nexp:
            bad.append(f"{name} face too coarse: {int(m.sum())} nodes, "
                       f"about {nexp:.0f} expected with FSZ = {c['fsz']} mm")

    if mf.sum() and mb.sum():
        sq = max(mf.sum(), mb.sum()) / min(mf.sum(), mb.sum())
        if sq > 1.15:
            bad.append(f"unbalanced faces: {int(mf.sum())} against "
                       f"{int(mb.sum())} nodes, ratio {sq:.2f}")

    if r["tmax"] - r["tmin"] < 1e-6:
        bad.append("uniform temperature, the load never arrived")

    if mf.sum():
        w = np.abs(r["ph"][mf]) ** 4
        yc = (r["xyz"][mf, 1] * w).sum() / w.sum() * 1e3
        xc = (r["xyz"][mf, 0] * w).sum() / w.sum() * 1e3
        xs = c.get("spotx", 0) or 0
        if abs(yc - c["spoty"]) > 2.0 or abs(xc - xs) > 2.0:
            bad.append(f"spot at ({xc:+.1f},{yc:+.1f}) "
                       f"instead of ({xs:+.1f},{c['spoty']})")

    frac = r["n_ok"] / max(r["n_tot"], 1)
    if frac < 0.80:
        zp = r["z_missing"]
        zf = c["th"] / 1000.0
        nb = int((np.abs(zp) < 1e-6).sum())
        nf = int((np.abs(zp - zf) < 1e-6).sum())
        alt = len(zp) - nb - nf
        warn.append(f"{len(zp)} nodes without a result: {nf} on the excited "
                    f"face, {nb} on the rear one, {alt} elsewhere")
        if alt:
            p = r["xyz_missing"] * 1e3
            warn.append(f"   coordinates of the missing nodes [mm]: "
                        f"x[{p[:,0].min():.3g},{p[:,0].max():.3g}] "
                        f"y[{p[:,1].min():.3g},{p[:,1].max():.3g}] "
                        f"z[{p[:,2].min():.3g},{p[:,2].max():.3g}]")
            uz, cz = np.unique(np.round(p[:, 2], 4), return_counts=True)
            top = np.argsort(-cz)[:5]
            warn.append("   most frequent z levels: " + ", ".join(
                f"{uz[i]:g} mm ({cz[i]})" for i in top))
    return runid, bad, r, warn, None

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BATCHES:
        print("usage: python check_campaign.py <batch> [csv_directory]")
        print("batches:", ", ".join(BATCHES))
        return
    batch = sys.argv[1]
    dirpath = sys.argv[2] if len(sys.argv) > 2 else win(WORKROOT, batch)
    cases = BATCHES[batch]
    if not os.path.isdir(dirpath):
        print(f"the directory {dirpath} does not exist.")
        print("The batch has not been launched yet, or the working directory "
              "is another one:\n  python check_campaign.py "
              f"{batch} <csv_directory>")
        return
    print(f"batch {batch}, directory: {dirpath}\n")

    store, nfail, pending = {}, 0, []
    for c in cases:
        for tag, freq in band(c):
            runid, bad, r, warn, skip = check_one(c, tag, freq, dirpath)
            if skip:
                pending.append(runid)
                continue
            if r is not None:
                store[runid] = r
            status = "FAIL" if bad else ("WARN" if warn else "PASS")
            nfail += bool(bad)
            extra = ""
            if r is not None:
                mf = np.abs(r["xyz"][:, 2] - c["th"]/1000.0) < 1e-6
                mb = np.abs(r["xyz"][:, 2]) < 1e-6
                extra = (f" | {int(mf.sum())}+{int(mb.sum())} nodes | "
                         f"|A|max {np.abs(r['ph']).max()*1e3:7.1f} mK")
            print(f"{status}  {runid:10s}{extra}")
            for b in bad:
                print(f"        -> {b}")
            for w in warn:
                print(f"        .  {w}")

    pairs = [(f"{c['id']}_{tag}", f"{c['twin']}_{tag}")
             for c in cases if c["twin"] for tag, _ in band(c)]
    done = [(a, b) for a, b in pairs if a in store and b in store]
    if done:
        print("\n--- meshes shared between defective and sound ---")
        for a, b in done:
            same = np.array_equal(store[a]["ids"], store[b]["ids"])
            d = np.abs(store[a]["xyz"] - store[b]["xyz"]).max() if same else np.nan
            ok = same and d == 0
            print(f"{'PASS' if ok else 'FAIL'}  {a} / {b}"
                  f"   identical nodes={same}  coordinate gap={d:.1e}")
            nfail += not ok

    if pending:
        print(f"\n{len(pending)} runs not produced yet: "
              f"{', '.join(pending[:6])}{' ...' if len(pending) > 6 else ''}")
        print("The batch is still running. Run this again when it ends.")
    print(f"\n{'ALL PASS' if nfail == 0 else str(nfail) + ' CHECKS FAILED'}"
          f" over {len(store)} runs read, {len(done)} pairs verified")

if __name__ == "__main__":
    main()
