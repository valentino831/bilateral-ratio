import sys

import numpy as np
from scipy.stats import f as fdist

import campaign as cp

QUANTITIES = ("d_refl", "d_tran", "rho")

def gather(batch, quantity="rho", kind="linear"):

    levels = cp.CONV_LEVELS[batch]
    tones = cp.common_tones(*levels)
    out = []
    for tone, _ in tones:
        xs, ys = [], []
        for cid in levels:
            r = cp.read(cid, tone, kind=kind, corrected=False)
            val = {"rho": abs(r["rho"]),
                   "d_refl": abs(r["d_refl"]) * 1e3,
                   "d_tran": abs(r["d_tran"]) * 1e3}[quantity]
            xs.append(cp.CASES[cid]["esz"] / r["mu"])
            ys.append(val)
        if len(xs) >= 3:
            out.append((tone, np.array(xs), np.array(ys)))
    return out

def _fit_at(p, x, y):

    A = np.c_[np.ones_like(x), x ** p]
    scale = abs(y[np.argmin(x)])
    coef, *_ = np.linalg.lstsq(A, y / scale, rcond=None)
    resid = (A @ coef) - y / scale
    return coef[0] * scale, coef[1] * scale, float((resid ** 2).sum())

def _rss(p, data):
    rss = n = npar = 0
    for _, x, y in data:
        rss += _fit_at(p, x, y)[2]
        n += len(x)
        npar += 2
    return rss, n, npar

def pooled_order(data, pmin=0.02, pmax=12.0, npts=2400, alpha=0.05):

    grid = np.linspace(pmin, pmax, npts)
    rss = np.array([_rss(p, data)[0] for p in grid])
    j = int(np.argmin(rss))
    p_hat = float(grid[j])
    _, n, npar = _rss(p_hat, data)
    nu = n - npar - 1
    if nu <= 0:
        return dict(p=p_hat, lo=np.nan, hi=np.nan, dof=nu, n=n)
    thr = rss[j] * (1.0 + fdist.ppf(1 - alpha, 1, nu) / nu)
    inside = grid[rss <= thr]

    at_min_edge = bool(j == 0 or j == len(grid) - 1)
    return dict(p=p_hat, lo=float(inside.min()), hi=float(inside.max()),
                dof=int(nu), n=int(n), identified=not at_min_edge,
                rss_ratio_edges=(float(rss[0] / rss[j]),
                                 float(rss[-1] / rss[j])),
                at_edge=bool(at_min_edge
                             or inside.min() <= grid[0] + 1e-9
                             or inside.max() >= grid[-1] - 1e-9))

def extrapolate(data, p):

    out = {}
    for tone, x, y in data:
        psi_inf, C, _ = _fit_at(p, x, y)
        finest = float(y[np.argmin(x)])
        out[tone] = dict(psi_inf=float(psi_inf), C=float(C), finest=finest,
                         rel_err=float(abs(finest - psi_inf) / abs(psi_inf)),
                         h_over_mu=float(x.min()),
                         x=[float(v) for v in x], y=[float(v) for v in y])
    return out

def coarse_to_fine(data):

    ch = []
    for _, x, y in data:
        o = np.argsort(x)
        ch.append(abs(y[o][-1] - y[o][0]) / abs(y[o][0]))
    return dict(median=float(np.median(ch)), max=float(np.max(ch)))

def fit(batch, quantity="rho", kind="linear"):

    data = gather(batch, quantity, kind)
    if not data:
        return None
    o = pooled_order(data)
    ex = extrapolate(data, o["p"])
    errs = np.array([v["rel_err"] for v in ex.values()])
    o.update(change=coarse_to_fine(data),
             tones=len(data), quantity=quantity, kind=kind, batch=batch,
             err_median=float(np.median(errs)), err_max=float(errs.max()),
             err_min=float(errs.min()), extrap=ex,
             x_min=float(min(v["h_over_mu"] for v in ex.values())),
             x_max=float(max(max(v["x"]) for v in ex.values())))
    return o

def predict_out_of_sample(fit_batch, test_batch, kind="linear"):

    p = pooled_order(gather(fit_batch, "rho", kind))["p"]
    levels = cp.CONV_LEVELS[test_batch]
    coarse, finest = levels[:-1], levels[-1]
    out = []
    for tone, _ in cp.common_tones(*levels):
        xs, ys = [], []
        for cid in coarse:
            r = cp.read(cid, tone, kind=kind, corrected=False)
            xs.append(cp.CASES[cid]["esz"] / r["mu"])
            ys.append(abs(r["rho"]))
        xs, ys = np.array(xs), np.array(ys)

        A = np.c_[np.ones_like(xs), xs ** p]
        coef, *_ = np.linalg.lstsq(A, ys, rcond=None)
        rf = cp.read(finest, tone, kind=kind, corrected=False)
        xf = cp.CASES[finest]["esz"] / rf["mu"]
        pred = float(coef[0] + coef[1] * xf ** p)
        got = float(abs(rf["rho"]))
        out.append(dict(tone=tone, f=rf["f"], h_over_mu=float(xf),
                        predicted=pred, computed=got,
                        rel_err=abs(pred - got) / got))
    return dict(order_used=p, points=out,
                worst=max(o["rel_err"] for o in out) if out else np.nan)

def bridge(kind="linear"):

    p = pooled_order(gather("conv2", "rho", kind))["p"]
    coarse, fine = cp.BRIDGE_LEVELS
    out = []
    for tone, _ in cp.common_tones(coarse, fine):
        a = cp.read(coarse, tone, kind=kind, corrected=False)
        b = cp.read(fine, tone, kind=kind, corrected=False)
        xa = cp.CASES[coarse]["esz"] / a["mu"]
        xb = cp.CASES[fine]["esz"] / b["mu"]
        rel = lambda u, v: abs(abs(u) - abs(v)) / abs(v)
        move = rel(a["rho"], b["rho"])

        move_log = abs(np.log(a["rho"] / b["rho"]))

        resid = move * xb ** p / (xa ** p - xb ** p)
        resid_log = move_log * xb ** p / (xa ** p - xb ** p)
        out.append(dict(tone=tone, f=b["f"],
                        move_rho=float(move), move_log=float(move_log),
                        move_d_refl=float(rel(a["d_refl"], b["d_refl"])),
                        move_d_tran=float(rel(a["d_tran"], b["d_tran"])),
                        residual=float(resid), residual_log=float(resid_log)))
    return dict(order_used=p, points=out,
                residual_max=max(o["residual"] for o in out),
                residual_log_min=min(o["residual_log"] for o in out),
                residual_log_max=max(o["residual_log"] for o in out))

def off_centre_spread(batch, frac=0.30, step=1.0, kind="linear"):

    levels = cp.CONV_LEVELS[batch]
    moves = []
    for tone, _ in cp.common_tones(*levels):
        _, xyz, d = cp.differential(levels[-1], tone)
        c = cp.CASES[levels[-1]]
        _, back = cp.face_masks(xyz, c["th"])
        peak = np.abs(d[back]).max()
        packs = {cid: cp.interpolators(cid, tone, kind)[1] for cid in levels}
        ys = np.arange(c["spoty"] - 4, c["spoty"] + 5.01, step)
        xs = np.arange(-6, 6.01, step)
        for x in xs:
            for y in ys:
                try:
                    if abs(cp._eval(packs[levels[-1]][1], (x, y))) < frac * peak:
                        continue
                    v = [abs(cp._eval(packs[cid][0], (x, y))
                             / cp._eval(packs[cid][1], (x, y)))
                         for cid in levels]
                except ValueError:
                    continue
                if all(np.isfinite(v)):
                    moves.append(abs(v[0] - v[-1]) / abs(v[-1]))
    m = np.array(moves)
    return dict(n=int(m.size), median=float(np.median(m)),
                p90=float(np.percentile(m, 90)))

def main():
    batch = sys.argv[1] if len(sys.argv) > 1 else "conv2"
    print(f"batch {batch}\n")
    for q in QUANTITIES:
        for kind in ("linear", "nearest"):
            o = fit(batch, q, kind)
            if o is None:
                print(f"{q:8s} {kind:8s}  no data")
                continue
            print(f"{q:8s} {kind:8s}  {o['n']:3d} values, {o['tones']} tones, "
                  f"{o['dof']:3d} d.o.f.   p = {o['p']:5.2f}  "
                  f"95 % [{o['lo']:.2f}, {o['hi']:.2f}]"
                  f"{'  (at the search edge)' if o.get('at_edge') else '':22s}"
                  f"  residual {100*o['err_median']:6.2f} %")
        print()
    o = fit(batch, "rho")
    print(f"|rho| tone by tone, common order p = {o['p']:.2f}")
    for tone, v in sorted(o["extrap"].items(), key=lambda kv: kv[1]["h_over_mu"]):
        print(f"  {tone:6s} h/mu = {v['h_over_mu']:.3f}   finest {v['finest']:8.4f}"
              f"   extrapolated {v['psi_inf']:8.4f}   error {100*v['rel_err']:5.2f} %")

if __name__ == "__main__":
    main()
