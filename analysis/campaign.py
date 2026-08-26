import glob
import os

import numpy as np
from scipy.interpolate import LinearNDInterpolator

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.abspath(os.path.join(HERE, "..", "data"))

K_STEEL = 50.0
RHO_STEEL = 7850.0
CP_STEEL = 490.0
ALPHA = K_STEEL / (RHO_STEEL * CP_STEEL)
CV_STEEL = RHO_STEEL * CP_STEEL

SPOT_RADIUS = 4.0
PLATE_WIDTH = 30.0

NETD = 18e-3
FPS = 240.0
DURATION = 25.0
BINNING = 4

def _c(twin, th=5, lxp=80, hd=2.0, hdepth=2.50, hy0=5, hy1=15, spoty=9,
       esz=1.00, note=""):
    return dict(twin=twin, th=th, lxp=lxp, hd=hd, hdepth=hdepth, hy0=hy0,
                hy1=hy1, spoty=spoty, esz=esz, note=note)

CASES = {

    "B1": _c("B0", hdepth=1.25, note="shallow defect"),
    "C1": _c("A0", note="reference configuration"),
    "D1": _c("D0", hdepth=3.75, note="deep defect"),

    "E1": _c("E0", hy1=13, note="channel truncated"),

    "F1": _c("F0", hd=0.5, note="diameter 0.5 mm"),
    "J1": _c("J0", hd=1.0, note="diameter 1.0 mm"),
    "K1": _c("K0", hd=3.0, note="diameter 3.0 mm"),

    "I1": _c("I0", hy0=-5, hy1=5, spoty=0, note="symmetric"),

    "SC1": _c("SC0", note="reference, tones matched to G1"),
    "QC1": _c("QC0", lxp=40, esz=0.80, note="grid check at the highest tone"),

    "G1": _c("G0", th=10, hd=4.0, hdepth=5.00, esz=1.40, note="10 mm plate"),

    "S16": _c("R16", lxp=40, esz=1.60), "S11": _c("R11", lxp=40, esz=1.13),
    "S08": _c("R08", lxp=40, esz=0.80),
    "L12": _c("M12", esz=1.25), "L10": _c("M10", esz=1.00),

    "P16": _c("Q16", lxp=40, esz=1.60), "P11": _c("Q11", lxp=40, esz=1.13),
    "P08": _c("Q08", lxp=40, esz=0.80),

    "X11": _c("Y11", lxp=20, esz=1.13), "X08": _c("Y08", lxp=20, esz=0.80),
    "X06": _c("Y06", lxp=20, esz=0.60),

    "V1": _c("V0", hd=1.40, hdepth=3.10, note="blind, deep and thin"),
    "W1": _c("W0", hd=0.70, hdepth=1.90, note="blind, shallow and thin"),

    "N1": _c("N0", hd=0.75, note="diameter 0.75 mm"),
    "U1": _c("U0", hd=1.50, note="diameter 1.5 mm"),
    "Z1": _c("Z0", hd=2.50, note="diameter 2.5 mm"),
}

HELD_OUT = (("E1", "shortened channel"),
            ("I1", "channel moved"),
            ("V1", "unseen pair, deep and thin"),
            ("W1", "unseen pair, shallow and thin"))

DEPTH_SERIES = ("B1", "C1", "D1")
DIAMETER_SERIES = ("F1", "J1", "C1", "K1")

DIAMETER_EXTRA = ("N1", "U1", "Z1")

def diameter_series():

    reg = registry()
    extra = [c for c in DIAMETER_EXTRA if reg.get(c)]
    cases = list(DIAMETER_SERIES) + extra
    return sorted(cases, key=lambda c: CASES[c]["hd"])
CONV_LEVELS = {
    "conv": ("S16", "S11", "S08"),
    "conv2": ("P16", "P11", "P08"),
    "conv3": ("X11", "X08", "X06"),
}
BRIDGE_LEVELS = ("L12", "L10")

_STORES = {}
_REGISTRY = None

def stores():

    if not _STORES:
        for path in sorted(glob.glob(os.path.join(DATA, "*.npz"))):
            name = os.path.basename(path)[:-4]
            if name == "campagna":
                continue
            _STORES[name] = np.load(path)
    return _STORES

def registry():

    global _REGISTRY
    if _REGISTRY is not None:
        return _REGISTRY
    reg = {}
    for batch, s in stores().items():
        for key in s.files:
            if not key.endswith("_f"):
                continue
            run = key[:-2]
            if "_" not in run:
                continue
            cid, tone = run.rsplit("_", 1)
            if cid not in CASES and cid not in {c["twin"] for c in CASES.values()}:
                continue
            entry = reg.setdefault(cid, {})
            if tone in entry:
                other = entry[tone][0]
                a, b = stores()[other], s
                same = (np.array_equal(a[f"{run}_ids"], b[f"{run}_ids"])
                        and np.abs(a[f"{run}_xyz"] - b[f"{run}_xyz"]).max() == 0)
                if not same:
                    raise ValueError(
                        f"{run} appears in {other} and {batch} with different "
                        f"meshes; one of the two batches is stale")
                continue
            entry[tone] = (batch, float(s[key]))
    _REGISTRY = reg
    return reg

def tones_of(cid):

    r = registry().get(cid, {})
    return sorted(((t, f) for t, (_, f) in r.items()), key=lambda x: x[1])

def common_tones(*cids):

    sets = [set(registry().get(c, {})) for c in cids]
    if not sets:
        return []
    keep = set.intersection(*sets)
    ref = registry()[cids[0]]
    return sorted(((t, ref[t][1]) for t in keep), key=lambda x: x[1])

def pair_by_frequency(cid_a, cid_b, rtol=1e-3):

    out = []
    for ta, fa in tones_of(cid_a):
        for tb, fb in tones_of(cid_b):
            if abs(fa - fb) <= rtol * fa:
                out.append((ta, tb, fa))
                break
    return sorted(out, key=lambda x: x[2])

def pair_by_group(cid_a, cid_b, key=lambda f, th: mu(f) / th, rtol=2e-2):

    out = []
    for ta, fa in tones_of(cid_a):
        ga = key(fa, CASES[cid_a]["th"])
        for tb, fb in tones_of(cid_b):
            gb = key(fb, CASES[cid_b]["th"])
            if abs(ga - gb) <= rtol * ga:
                out.append((ta, tb, ga))
                break
    return sorted(out, key=lambda x: x[2])

def mu(f, alpha=ALPHA):

    return np.sqrt(2.0 * alpha / (2.0 * np.pi * f)) * 1e3

def noise_floor():

    n = FPS * DURATION * BINNING * BINNING
    return NETD / np.sqrt(n / 2.0) * 1e3

def face_masks(xyz, th_mm):

    z = xyz[:, 2]
    back = np.abs(z) < 1e-6
    front = np.abs(z - th_mm / 1000.0) < 1e-6
    return front, back

def differential(cid, tone):

    c = CASES[cid]
    twin = c["twin"]
    reg = registry()
    if tone not in reg.get(cid, {}) or tone not in reg.get(twin, {}):
        raise KeyError(f"{cid}/{twin} at tone {tone} not available")
    ba, fa = reg[cid][tone]
    bb, fb = reg[twin][tone]
    a, b = stores()[ba], stores()[bb]
    ka, kb = f"{cid}_{tone}", f"{twin}_{tone}"
    if not np.array_equal(a[ka + "_ids"], b[kb + "_ids"]):
        raise ValueError(f"{ka} and {kb} do not share the node numbering")
    if np.abs(a[ka + "_xyz"] - b[kb + "_xyz"]).max() != 0.0:
        raise ValueError(f"{ka} and {kb} do not share the node coordinates")
    if abs(fa - fb) / fa > 1e-3:
        raise ValueError(f"{ka} and {kb} were run at different frequencies")
    return fa, a[ka + "_xyz"], a[ka + "_ph"] - b[kb + "_ph"]

def phase_contrast(cid, tone):

    c = CASES[cid]
    reg = registry()
    ba, _ = reg[cid][tone]
    bb, _ = reg[c["twin"]][tone]
    a, b = stores()[ba], stores()[bb]
    return np.degrees(np.angle(a[f"{cid}_{tone}_ph"]
                               / b[f"{c['twin']}_{tone}_ph"]))

def _sample(xyz, values, mask, point_mm, kind="linear"):

    p = xyz[mask][:, :2] * 1e3
    v = values[mask]
    if kind == "linear":
        re = LinearNDInterpolator(p, v.real)(point_mm)
        im = LinearNDInterpolator(p, v.imag)(point_mm)
        if not (np.isfinite(re) and np.isfinite(im)):
            raise ValueError(f"{point_mm} outside the convex hull of the face")
        return complex(re, im)
    j = int(np.argmin(((p - np.asarray(point_mm, float)) ** 2).sum(axis=1)))
    return complex(v[j].real, v[j].imag)

_INTERP = {}

def interpolators(cid, tone, kind="linear"):

    key = (cid, tone, kind)
    if key in _INTERP:
        return _INTERP[key]
    c = CASES[cid]
    f, xyz, d = differential(cid, tone)
    front, back = face_masks(xyz, c["th"])
    packs = []
    for mask in (front, back):
        p = xyz[mask][:, :2] * 1e3
        v = d[mask]
        if kind == "linear":
            packs.append(("linear", LinearNDInterpolator(p, v.real),
                          LinearNDInterpolator(p, v.imag)))
        else:
            packs.append(("nearest", p, v))
    _INTERP[key] = (f, packs, int(front.sum()), int(back.sum()))
    return _INTERP[key]

def _eval(pack, point_mm):
    if pack[0] == "linear":
        re, im = pack[1](point_mm), pack[2](point_mm)
        if not (np.isfinite(re) and np.isfinite(im)):
            raise ValueError(f"{point_mm} outside the convex hull of the face")
        return complex(re, im)
    p, v = pack[1], pack[2]
    j = int(np.argmin(((p - np.asarray(point_mm, float)) ** 2).sum(axis=1)))
    return complex(v[j].real, v[j].imag)

def read(cid, tone, point=None, kind="linear"):

    c = CASES[cid]
    f, packs, nf, nb = interpolators(cid, tone, kind)
    p = point if point is not None else (0.0, float(c["spoty"]))
    dr = _eval(packs[0], p)
    dt = _eval(packs[1], p)
    return dict(f=f, mu=mu(f), d_refl=dr, d_tran=dt, rho=dr / dt,
                n_front=nf, n_back=nb)

def series(cid, kind="linear", tones=None):

    out = []
    for tone, _ in tones_of(cid):
        if tones is not None and tone not in tones:
            continue
        out.append((tone, read(cid, tone, kind=kind)))
    return out

def fwhm_along_x(xyz, field, mask, y_mm, half=12.0, n=1201):

    p = xyz[mask][:, :2] * 1e3
    itp = LinearNDInterpolator(p, field[mask])
    x = np.linspace(-half, half, n)
    v = itp(np.c_[x, np.full_like(x, y_mm)])
    ok = np.isfinite(v)
    x, v = x[ok], v[ok]
    edge = np.abs(x) > 2.0 * half / 3.0
    v = v - (np.median(v[edge]) if edge.any() else 0.0)
    ipk = int(np.argmax(v))
    if v[ipk] <= 0:
        return np.nan, np.nan
    lvl = v[ipk] / 2.0

    def cross(rng):
        for i in rng:
            a, b = v[i], v[i + 1]
            if (a - lvl) * (b - lvl) <= 0 and a != b:
                return x[i] + (lvl - a) / (b - a) * (x[i + 1] - x[i])
        return np.nan

    return (float(cross(range(ipk, len(v) - 1)) - cross(range(ipk - 1, -1, -1))),
            float(v[ipk]))

def spot_to_edge(cid):

    return PLATE_WIDTH / 2.0 - abs(CASES[cid]["spoty"])

def verify(verbose=True):

    bad = []
    reg = registry()
    for cid, c in CASES.items():
        for tone, f in tones_of(cid):
            try:
                _, xyz, _ = differential(cid, tone)
            except (KeyError, ValueError) as e:
                bad.append(f"{cid}_{tone}: {e}")
                continue
            front, back = face_masks(xyz, c["th"])
            nf, nb = int(front.sum()), int(back.sum())
            if nf == 0 or nb == 0:
                bad.append(f"{cid}_{tone}: one of the two faces is empty")
                continue
            if max(nf, nb) / min(nf, nb) > 1.15:
                bad.append(f"{cid}_{tone}: faces unbalanced, {nf} against {nb}")

            expected = 1.18 * c["lxp"] * PLATE_WIDTH / c["esz"] ** 2
            if min(nf, nb) < 0.7 * expected:
                bad.append(f"{cid}_{tone}: face too coarse, {min(nf, nb)} nodes "
                           f"against about {expected:.0f} expected")

            s = stores()[reg[cid][tone][0]]
            w = np.abs(s[f"{cid}_{tone}_ph"][front]) ** 4
            yc = (xyz[front, 1] * w).sum() / w.sum() * 1e3
            xc = (xyz[front, 0] * w).sum() / w.sum() * 1e3
            if abs(yc - c["spoty"]) > 2.0 or abs(xc) > 2.0:
                bad.append(f"{cid}_{tone}: spot at ({xc:+.1f},{yc:+.1f}) "
                           f"instead of (0,{c['spoty']})")
    if verbose:
        n = sum(len(tones_of(c)) for c in CASES)
        print(f"verification: {n} defect and twin pairs checked over "
              f"{len(stores())} batches")
        for b in bad:
            print("  FAIL " + b)
        print("  all invariants hold" if not bad else f"  {len(bad)} FAILURES")
    return len(bad)

if __name__ == "__main__":
    print(f"batches: {', '.join(sorted(stores()))}")
    print(f"alpha = {ALPHA:.4e} m2/s, phasor noise floor "
          f"{noise_floor():.4f} mK\n")
    for cid in sorted(CASES):
        t = tones_of(cid)
        if t:
            print(f"  {cid:5s} ({CASES[cid]['note'] or 'convergence level':32s}) "
                  f"{len(t):2d} tones  "
                  f"{t[0][1]:.4f} to {t[-1][1]:.4f} Hz")
    print()
    raise SystemExit(verify())
