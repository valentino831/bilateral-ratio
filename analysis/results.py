import json
import os
import sys

import numpy as np

import campaign as cp
import convergence as cv

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.abspath(os.path.join(HERE, ".."))

CUT_Y = 12.0
FRAC_PEAK = 0.30

def geometry():

    c = cp.CASES["C1"]
    return dict(length=float(c["lxp"]), width=cp.PLATE_WIDTH,
                thickness=float(c["th"]), diameter=float(c["hd"]),
                depth=float(c["hdepth"]), spot_y=float(c["spoty"]),
                spot_r=cp.SPOT_RADIUS, chan_y0=float(c["hy0"]),
                chan_y1=float(c["hy1"]),
                clearance=float(c["hdepth"] - c["hd"] / 2))

def campaign_summary():
    reg = cp.registry()
    runs = sum(len(v) for v in reg.values())
    return dict(n_runs=runs, n_batches=len(cp.stores()),
                n_cases=len([c for c in cp.CASES if cp.tones_of(c)]),
                alpha=cp.ALPHA, noise_mK=cp.noise_floor(),
                spot_radius=cp.SPOT_RADIUS, plate_width=cp.PLATE_WIDTH,
                netd_mK=cp.NETD * 1e3, fps=cp.FPS, duration=cp.DURATION,
                binning=cp.BINNING)

def reference_band():

    rows = []
    for cid in ("C1", "SC1"):
        for tone, f in cp.tones_of(cid):
            r = cp.read(cid, tone)
            pc = cp.phase_contrast(cid, tone)
            _, xyz, d = cp.differential(cid, tone)
            front, back = cp.face_masks(xyz, cp.CASES[cid]["th"])
            wr, pr = cp.fwhm_along_x(xyz, np.abs(pc), front, CUT_Y)
            wt, pt = cp.fwhm_along_x(xyz, np.abs(pc), back, CUT_Y)

            ctr = (0.0, float(cp.CASES[cid]["spoty"]))
            cr = cp._sample(xyz, pc.astype(complex), front, ctr).real
            ct = cp._sample(xyz, pc.astype(complex), back, ctr).real
            rows.append(dict(case=cid, tone=tone, f=f, mu=r["mu"],
                             mu_over_L=r["mu"] / cp.CASES[cid]["th"],
                             width_refl=wr, width_tran=wt,
                             peak_refl=pr, peak_tran=pt,
                             dphi_refl=float(cr), dphi_tran=float(ct),
                             d_refl_mK=abs(r["d_refl"]) * 1e3,
                             d_tran_mK=abs(r["d_tran"]) * 1e3,
                             abs_rho=abs(r["rho"]),
                             arg_rho=float(np.degrees(np.angle(r["rho"])))))
    rows.sort(key=lambda r: r["f"])
    g = lambda k: np.array([r[k] for r in rows], float)

    def crossings(key):

        w = g(key)
        w = w[np.isfinite(w)]
        return int((np.diff(np.sign(w - cp.CASES["C1"]["hd"])) != 0).sum())

    nan_r = int(np.isnan(g("width_refl")).sum())
    nan_t = int(np.isnan(g("width_tran")).sum())
    nanmin = lambda k: float(np.nanmin(g(k)))
    nanmax = lambda k: float(np.nanmax(g(k)))
    dr_, dt_ = g("dphi_refl"), g("dphi_tran")
    i_zero = int(np.argmin(np.abs(dr_)))

    i_pk = int(np.argmax(np.abs(dt_)))
    i_pk_r = int(np.argmax(np.abs(dr_)))
    return dict(rows=rows, n_tones=len(rows),
                dphi_tran_all_negative=bool((dt_ < 0).all()),
                dphi_tran_min=float(dt_.min()), dphi_tran_max=float(dt_.max()),
                dphi_refl_sign_change=bool(dr_.max() > 0 > dr_.min()),
                dphi_refl_abs_min=float(abs(dr_[i_zero])),
                f_dphi_refl_zero=float(g("f")[i_zero]),
                dphi_refl_min=float(dr_.min()), dphi_refl_max=float(dr_.max()),
                f_dphi_peak=float(g("f")[i_pk]),
                mul_dphi_peak=float(g("mu_over_L")[i_pk]),
                dphi_tran_end=float(abs(dt_[-1])),
                dphi_peak_same_face=bool(i_pk == i_pk_r),
                width_refl_undefined=nan_r, width_tran_undefined=nan_t,
                f_min=rows[0]["f"], f_max=rows[-1]["f"],
                f_span=rows[-1]["f"] / rows[0]["f"],
                mu_over_L_max=max(g("mu_over_L")),
                mu_over_L_min=min(g("mu_over_L")),
                true_width=cp.CASES["C1"]["hd"],
                width_refl_min=nanmin("width_refl"),
                width_refl_max=nanmax("width_refl"),
                width_tran_min=nanmin("width_tran"),
                width_tran_max=nanmax("width_tran"),
                width_refl_span=nanmax("width_refl") / nanmin("width_refl"),
                width_refl_cross=crossings("width_refl"),
                width_tran_cross=crossings("width_tran"),
                width_tran_span=nanmax("width_tran") / nanmin("width_tran"),
                peak_refl_min=nanmin("peak_refl"),
                peak_refl_max=nanmax("peak_refl"),
                peak_tran_min=nanmin("peak_tran"),
                peak_tran_max=nanmax("peak_tran"),
                rho_min=float(np.nanmin(g("abs_rho"))),
                mu_over_L_at_rho_min=rows[int(g("abs_rho").argmin())]["mu_over_L"],
                f_at_rho_min=rows[int(g("abs_rho").argmin())]["f"])

def convergence_block():
    out = {"fits": {}}
    for q in cv.QUANTITIES:
        for kind in ("linear", "nearest"):
            o = cv.fit("conv2", q, kind)
            o.pop("extrap") if kind == "nearest" else None
            out["fits"][f"{q}_{kind}"] = o
    lin = out["fits"]["rho_linear"]
    nn = out["fits"]["rho_nearest"]
    out["overlap"] = [max(lin["lo"], nn["lo"]), min(lin["hi"], nn["hi"])]
    out["oos"] = cv.predict_out_of_sample("conv2", "conv3")
    out["bridge"] = cv.bridge()
    out["off_centre"] = cv.off_centre_spread("conv2", FRAC_PEAK)

    short = []
    for a, b in (("S11", "X11"), ("S08", "X08")):
        for ta, tb, f in cp.pair_by_frequency(a, b):
            u = abs(cp.read(a, ta)["rho"])
            v = abs(cp.read(b, tb)["rho"])
            short.append(dict(f=f, h=cp.CASES[a]["esz"],
                              full=u, short=v, rel=abs(u - v) / u))
    out["shortening"] = short
    out["shortening_coarse"] = max(s["rel"] for s in short
                                   if s["h"] == max(x["h"] for x in short))
    out["shortening_fine"] = max(s["rel"] for s in short
                                 if s["h"] == min(x["h"] for x in short))

    r = out["bridge"]["residual_log_max"]
    out["declared_uncertainty"] = float(np.ceil(r * 200.0) / 200.0)

    a = abs(cp.read("SC1", "T156")["rho"])
    b = abs(cp.read("QC1", "T156")["rho"])
    out["high_tone_check"] = dict(f=cp.read("QC1", "T156")["f"],
                                  mu=cp.read("QC1", "T156")["mu"],
                                  nominal=a, fine=b, rel=abs(a - b) / b)
    return out

def depth_block():

    cases = cp.depth_series()
    zs = np.array([cp.CASES[c]["hdepth"] for c in cases])
    rows = []
    for tone, f in cp.common_tones(*cases):
        v = np.array([abs(cp.read(c, tone)["rho"]) for c in cases])
        ph = np.array([np.degrees(np.angle(cp.read(c, tone)["rho"]))
                       for c in cases])

        slope = (np.log(v[0]) - np.log(v[-1])) / (zs[-1] - zs[0])
        rows.append(dict(tone=tone, f=f, mu_over_L=cp.mu(f) / cp.CASES["C1"]["th"],
                         rho=[float(u) for u in v], arg=[float(u) for u in ph],
                         slope=float(slope), factor=float(v[0] / v[-1])))
    s = np.array([r["slope"] for r in rows])
    return dict(z=[float(u) for u in zs], rows=rows,
                dz=float(zs[-1] - zs[0]),
                slope_min=float(s.min()), slope_max=float(s.max()),
                f_at_slope_max=rows[int(s.argmax())]["f"],
                factor_max=max(r["factor"] for r in rows))

def diameter_block():

    cases = list(cp.diameter_series())
    ds = np.array([cp.CASES[c]["hd"] for c in cases])

    clear = np.array([cp.CASES[c]["hdepth"] - cp.CASES[c]["hd"] / 2 for c in cases])
    ratio = ds / clear
    rows = []
    for tone, f in cp.common_tones(*cases):
        v = np.array([abs(cp.read(c, tone)["rho"]) for c in cases])
        x, y = np.log(ds), np.log(v)

        A = np.c_[np.ones_like(x), x]
        b1, *_ = np.linalg.lstsq(A, y, rcond=None)
        r2 = 1 - ((y - A @ b1) ** 2).sum() / ((y - y.mean()) ** 2).sum()

        deg = 2
        b2 = np.polyfit(x, y, deg)
        xref = np.log(cp.CASES["C1"]["hd"])
        dref = float(np.polyval(np.polyder(b2), xref))
        local = [float((y[i + 1] - y[i]) / (x[i + 1] - x[i]))
                 for i in range(len(x) - 1)]
        rows.append(dict(tone=tone, f=f, rho=[float(u) for u in v],
                         loglin_slope=float(b1[1]), loglin_r2=float(r2),
                         quad=float(b2[-3]), d_ln_rho_d_ln_d=dref,
                         local=local))

    node = []
    for tone, f in cp.common_tones(*cases):
        y = np.array([np.log(abs(cp.read(c, tone)["rho"])) for c in cases])
        xx = np.log(ds)
        node.extend(np.abs(np.polyval(np.polyfit(xx, y, 2), xx) - y))
    lo = np.array([r["local"][0] for r in rows])
    hi = np.array([r["local"][-1] for r in rows])

    small_a, small_b = cases[0], cases[2]
    small_move = []
    for tone, f in cp.common_tones(small_a, small_b):
        a = abs(cp.read(small_a, tone)["rho"])
        b = abs(cp.read(small_b, tone)["rho"])
        small_move.append(abs(b - a) / a)
    return dict(cases=cases, d=[float(u) for u in ds],
                node_rms=float(np.sqrt(np.mean(np.square(node)))),
                node_max=float(np.max(node)),
                clearance=[float(u) for u in clear],
                ratio=[float(u) for u in ratio], rows=rows,
                r2_max=max(r["loglin_r2"] for r in rows),
                slope_small_min=float(lo.min()), slope_small_max=float(lo.max()),
                slope_large_min=float(hi.min()), slope_large_max=float(hi.max()),
                slope_growth=float(hi.mean() / lo.mean()),
                small_move_min=float(min(small_move)),
                small_move_max=float(max(small_move)),
                d_small=float(ds[0]), d_next=float(ds[1]),
                ratio_small=float(ratio[0]), ratio_next=float(ratio[1]),
                ratio_large=float(ratio[-1]))

def extent_block():

    rows = []
    for tone, f in cp.common_tones("C1", "E1"):
        ra, rb = cp.read("C1", tone)["rho"], cp.read("E1", tone)["rho"]
        a, b = abs(ra), abs(rb)
        rows.append(dict(tone=tone, f=f, full=a, truncated=b,
                         rel=abs(b - a) / a,
                         dphase=float(abs(np.degrees(np.angle(rb / ra)))),
                         dlog=float(abs(np.log(rb / ra)))))
    length_full = cp.CASES["C1"]["hy1"] - cp.CASES["C1"]["hy0"]
    length_trunc = cp.CASES["E1"]["hy1"] - cp.CASES["E1"]["hy0"]
    return dict(rows=rows, rel_max=max(r["rel"] for r in rows),
                dphase_max=max(r["dphase"] for r in rows),
                dlog_max=max(r["dlog"] for r in rows),
                length_full=float(length_full),
                length_truncated=float(length_trunc),
                length_ratio=float(length_full / length_trunc))

def identifiability_block(depth, diam):

    tones = [t for t, _ in cp.common_tones(*cp.depth_series(), *cp.diameter_series())]
    dz_rows = {r["tone"]: r for r in depth["rows"]}
    dd_rows = {r["tone"]: r for r in diam["rows"]}
    tones.sort(key=lambda t: dz_rows[t]["f"])

    raw = np.array([np.angle(cp.read(cp.depth_series()[-1], t)["rho"]
                             / cp.read(cp.depth_series()[0], t)["rho"])
                    for t in tones])
    dphi_z = np.unwrap(raw) / depth["dz"]

    rows = []
    for t, pz in zip(tones, dphi_z):
        f = dz_rows[t]["f"]
        a = -dz_rows[t]["slope"]
        b = dd_rows[t]["d_ln_rho_d_ln_d"]
        pd_ = cp.read("K1", t)["rho"] / cp.read("J1", t)["rho"]
        dph_dd = np.angle(pd_) / np.log(cp.CASES["K1"]["hd"] / cp.CASES["J1"]["hd"])
        rows.append(dict(tone=t, f=f, dlnA_dz=float(a), dlnA_dlnd=float(b),
                         dph_dz=float(pz), dph_dlnd=float(dph_dd),
                         tau=float(-b / a), tau_phase=float(-dph_dd / pz)))
    tau = np.array([r["tau"] for r in rows])
    tauf = np.array([r["tau_phase"] for r in rows])

    def cond(sel):

        J = []
        for r in sel:
            J.append([r["dlnA_dz"], r["dlnA_dlnd"]])
            J.append([r["dph_dz"], r["dph_dlnd"]])
        s_ = np.linalg.svd(np.asarray(J, float), compute_uv=False)
        return float(s_[0] / s_[-1]) if s_[-1] > 0 else np.inf

    ends = [rows[0], rows[-1]]
    conds = dict(one=cond(rows[:1]), ends=cond(ends), all=cond(rows))
    worst = max(rows, key=lambda r: cond([r]))
    best = min(rows, key=lambda r: cond([r]))
    return dict(rows=rows, tones=tones, n_tones=len(rows),
                f_span=rows[-1]["f"] / rows[0]["f"],
                tau_min=float(tau.min()), tau_max=float(tau.max()),
                tau_spread=float((tau.max() - tau.min()) / tau.mean()),
                tau_ref=float(np.median(tau)),
                tauf_min=float(tauf.min()), tauf_max=float(tauf.max()),
                gap_min=float(np.abs(tau - tauf).min()),
                gap_max=float(np.abs(tau - tauf).max()),
                cond_one_best=cond([best]), cond_one_worst=cond([worst]),
                f_one_best=best["f"], f_one_worst=worst["f"],
                conds=conds)

def _separable_surface(depth_cases, diam_cases, tones, d_ref, deg_d=None,
                       deg_z=None):

    def logs(cases, tone):
        ref = cp.read("C1", tone)["rho"]
        return np.array([np.log(cp.read(c, tone)["rho"] / ref) for c in cases])

    z_ref = cp.CASES["C1"]["hdepth"]
    zs = np.array([cp.CASES[c]["hdepth"] for c in depth_cases], float) - z_ref
    ds = np.log(np.array([cp.CASES[c]["hd"] for c in diam_cases], float)
                / d_ref)
    if deg_z is None:
        deg_z = 2 if len(zs) >= 3 else 1
    if deg_d is None:
        deg_d = 2 if len(ds) >= 3 else 1

    def through_origin(x, y, deg):

        a = np.stack([x ** k for k in range(1, deg + 1)], axis=1)
        c, *_ = np.linalg.lstsq(a, y, rcond=None)
        return c

    cz = {t: through_origin(zs, logs(depth_cases, t), deg_z) for t in tones}
    cd = {t: through_origin(ds, logs(diam_cases, t), deg_d) for t in tones}

    def poly(c, x):
        out = np.zeros_like(np.asarray(x, dtype=complex))
        for k, ck in enumerate(c, 1):
            out = out + ck * np.asarray(x) ** k
        return out

    def predict(tone, z, d):
        return (poly(cz[tone], np.asarray(z) - z_ref)
                + poly(cd[tone], np.log(np.asarray(d) / d_ref)))

    return predict

def measured(cid, tone):

    return np.log(cp.read(cid, tone)["rho"] / cp.read("C1", tone)["rho"])

def estimate(target, depth_cases, diam_cases, tones, unc, d_ref):

    predict = _separable_surface(depth_cases, diam_cases, tones, d_ref)
    meas = {t: measured(target, t) for t in tones}
    zs = [cp.CASES[c]["hdepth"] for c in cp.depth_series()]
    ds = [cp.CASES[c]["hd"] for c in cp.diameter_series()]
    zg = np.linspace(min(zs), max(zs), 1401)
    dg = np.exp(np.linspace(np.log(min(ds)), np.log(max(ds)), 561))
    Z, Dg = np.meshgrid(zg, dg, indexing="ij")
    crit = np.zeros_like(Z)
    for t in tones:
        crit += np.abs(predict(t, Z, Dg) - meas[t]) ** 2
    res = np.sqrt(crit / len(tones))
    i, j = np.unravel_index(int(np.argmin(res)), res.shape)
    ok = res <= unc
    return dict(target=target, n_tones=len(tones),
                z_true=float(cp.CASES[target]["hdepth"]),
                d_true=float(cp.CASES[target]["hd"]),
                z_hat=float(zg[i]), d_hat=float(dg[j]),
                residual=float(res[i, j]),
                z_lo=float(Z[ok].min()) if ok.any() else None,
                z_hi=float(Z[ok].max()) if ok.any() else None,
                d_lo=float(Dg[ok].min()) if ok.any() else None,
                d_hi=float(Dg[ok].max()) if ok.any() else None)

def estimation_block(unc):

    tones = [t for t, _ in cp.common_tones(*cp.depth_series(), *cp.diameter_series())]
    f_of = {round(cp.read("C1", t)["f"], 4): t for t in tones}
    d_ref = cp.CASES["C1"]["hd"]
    dep, dia = cp.depth_series(), list(cp.diameter_series())

    def shared(case):

        return [f_of[round(f, 4)] for _, f in cp.tones_of(case)
                if round(f, 4) in f_of]

    def surface_error(cid, cz, cd, tt):

        pr = _separable_surface(cz, cd, tt, d_ref)
        z, d = cp.CASES[cid]["hdepth"], cp.CASES[cid]["hd"]
        e = [abs(pr(t, z, d) - measured(cid, t)) for t in tt]
        return float(np.sqrt(np.mean(np.square(e))))

    rows = []
    r = estimate("C1", dep, dia, tones, unc, d_ref)
    r["mode"] = "reference, all cases"
    r["surface_rms"] = surface_error("C1", dep, dia, tones)
    rows.append(r)
    dia_loo = [c for c in dia if c != "J1"]
    r = estimate("J1", dep, dia_loo, tones, unc, d_ref)
    r["mode"] = "diameter, case left out"
    r["surface_rms"] = surface_error("J1", dep, dia_loo, tones)
    rows.append(r)
    for cid, label in cp.HELD_OUT:
        if cid not in cp.CASES or not shared(cid):
            continue
        r = estimate(cid, dep, dia, shared(cid), unc, d_ref)
        r["mode"] = label
        r["surface_rms"] = surface_error(cid, dep, dia, shared(cid))
        rows.append(r)
    return dict(rows=rows, n_tones=len(tones))

def surrogate_probe(unc):

    tones = [t for t, _ in cp.common_tones(*cp.depth_series(), *cp.diameter_series())]
    dep, dia = cp.depth_series(), list(cp.diameter_series())
    d_ref = cp.CASES["C1"]["hd"]
    ds = np.array([cp.CASES[c]["hd"] for c in dia])
    zs = np.array([cp.CASES[c]["hdepth"] for c in dep]) - cp.CASES["C1"]["hdepth"]
    off_cases = [c for c, _ in cp.HELD_OUT
                 if {t for t, _ in cp.tones_of(c)} >= set(tones)]

    def node_rms(cases, x, deg):

        e = []
        for t in tones:
            y = np.array([measured(c, t) for c in cases])
            a = np.stack([x ** k for k in range(1, deg + 1)], axis=1)
            c, *_ = np.linalg.lstsq(a, y, rcond=None)
            e.extend(np.abs(a @ c - y))
        return float(np.sqrt(np.mean(np.square(e))))

    def off_error(deg_z, deg_d):
        pr = _separable_surface(dep, dia, tones, d_ref, deg_d=deg_d,
                                deg_z=deg_z)
        return {cid: float(np.sqrt(np.mean(
            [abs(pr(t, cp.CASES[cid]["hdepth"], cp.CASES[cid]["hd"])
                 - measured(cid, t)) ** 2 for t in tones])))
                for cid in off_cases}

    def loo(cases, x, deg):

        e = []
        for k in range(len(cases)):
            if cases[k] == "C1":
                continue
            keep = [i for i in range(len(cases)) if i != k]
            for t in tones:
                y = np.array([measured(cases[i], t) for i in keep])
                a = np.stack([x[keep] ** q for q in range(1, deg + 1)], axis=1)
                c, *_ = np.linalg.lstsq(a, y, rcond=None)
                pred = sum(c[q - 1] * x[k] ** q for q in range(1, deg + 1))
                e.append(abs(pred - measured(cases[k], t)))
        return float(np.sqrt(np.mean(np.square(e))))

    degs_z = [d for d in (2, 3, 4) if d <= len(dep) - 1]
    degs_d = [d for d in (2, 3, 4) if d <= len(dia) - 1]
    grid = []
    for dz in degs_z:
        for dd in degs_d:
            grid.append(dict(deg_z=dz, deg_d=dd, off=off_error(dz, dd)))
    best = min(grid, key=lambda g: max(g["off"].values()))
    base = [g for g in grid if g["deg_z"] == 2 and g["deg_d"] == 2][0]
    lz = np.log(ds / d_ref)
    return dict(n_diameters=len(dia), n_depths=len(dep),
                d_values=[float(u) for u in sorted(ds)],
                cases=off_cases, grid=grid, base=base, best=best,
                floor=float(min(max(g["off"].values()) for g in grid)),
                node_z=[dict(deg=d, rms=node_rms(dep, zs, d), loo=loo(dep, zs, d))
                        for d in degs_z],
                node_d=[dict(deg=d, rms=node_rms(dia, lz, d), loo=loo(dia, lz, d))
                        for d in degs_d],
                loo_best_z=min(degs_z, key=lambda d: loo(dep, zs, d)),
                loo_best_d=min(degs_d, key=lambda d: loo(dia, lz, d)))

def profiled_residual(depth, diam, tones, unc):

    predict = _separable_surface(cp.depth_series(), list(cp.diameter_series()),
                                 tones if isinstance(tones, list)
                                 else sorted({t for v in tones.values()
                                              for t in v}),
                                 cp.CASES["C1"]["hd"])
    zs = np.array([cp.CASES[c]["hdepth"] for c in cp.depth_series()])
    ds = np.array([cp.CASES[c]["hd"] for c in cp.diameter_series()])
    zg = np.linspace(zs.min(), zs.max(), 1200)
    dg = np.exp(np.linspace(np.log(ds.min()), np.log(ds.max()), 400))

    out = {}
    for label, sel in tones.items():
        prof, zhat = np.empty_like(dg), np.empty_like(dg)
        for i, d in enumerate(dg):
            c = np.zeros_like(zg)
            for tone in sel:
                c += np.abs(predict(tone, zg, d) - measured("C1", tone)) ** 2
            j = int(np.argmin(c))
            prof[i] = np.sqrt(c[j] / len(sel))
            zhat[i] = zg[j]
        ok = prof <= unc

        out[label] = dict(d=[float(u) for u in dg],
                          residual=[float(u) for u in prof],
                          z_hat=[float(u) for u in zhat],
                          d_lo=float(dg[ok].min()) if ok.any() else None,
                          d_hi=float(dg[ok].max()) if ok.any() else None,
                          z_lo=float(zhat[ok].min()) if ok.any() else None,
                          z_hi=float(zhat[ok].max()) if ok.any() else None,
                          n_tones=len(sel))
    return out

def similarity_block():
    rows = []
    for t5, t10, _ in cp.pair_by_group("SC1", "G1"):
        a = cp.read("SC1", t5)
        b = cp.read("G1", t10)
        L5 = cp.CASES["SC1"]["th"]
        L10 = cp.CASES["G1"]["th"]
        rows.append(dict(
            f5=a["f"], f10=b["f"],
            mu_over_L=a["mu"] / L5, mu_over_L_10=b["mu"] / L10,
            rho5=abs(a["rho"]), rho10=abs(b["rho"]),
            arg5=float(np.degrees(np.angle(a["rho"]))),
            arg10=float(np.degrees(np.angle(b["rho"]))),
            rel=abs(abs(a["rho"]) - abs(b["rho"])) / abs(b["rho"]),
            dphase=float(abs(np.degrees(np.angle(a["rho"] / b["rho"])))),
            lat5=cp.spot_to_edge("SC1") / a["mu"],
            lat10=cp.spot_to_edge("G1") / b["mu"]))
    rows.sort(key=lambda r: r["mu_over_L"])
    return dict(rows=rows,
                rel_min=min(r["rel"] for r in rows),
                rel_max=max(r["rel"] for r in rows),
                dphase_max=max(r["dphase"] for r in rows),
                lat10_at_worst=rows[int(np.argmax([r["rel"] for r in rows]))]["lat10"],
                lat5_at_worst=rows[int(np.argmax([r["rel"] for r in rows]))]["lat5"],
                mu_over_L_at_worst=rows[int(np.argmax([r["rel"] for r in rows]))]["mu_over_L"])

def budget_block(depth, conv):

    S = depth["slope_max"]
    eps = [0.05, 0.10, 0.25]
    sig = cp.noise_floor()

    def best(cid):
        vals = [(abs(cp.read(cid, t)["d_tran"]) * 1e3, f)
                for t, f in cp.tones_of(cid)]
        return max(vals)

    ref, f_ref = best("C1")
    smallest, f_small = best("F1")
    worst_small = min(abs(cp.read("F1", t)["d_tran"]) * 1e3
                      for t, _ in cp.tones_of("F1"))
    rel_noise = sig / ref
    return dict(S=S, f_at_S=depth["f_at_slope_max"],
                eps=eps,
                dz_um=[float(1e3 * np.log(1 + e) / S) for e in eps],
                sigma_mK=sig, ref_d_tran_mK=ref,
                rel_noise=float(rel_noise),
                dz_noise_um=float(1e3 * np.log(1 + rel_noise) / S),
                f_ref=f_ref, f_small=f_small,
                smallest_d_tran_mK=smallest,
                worst_small_d_tran_mK=worst_small,
                snr_smallest=float(smallest / sig),
                snr_smallest_worst=float(worst_small / sig),
                d_smallest=cp.CASES["F1"]["hd"],
                dz_numerical_um=float(1e3 * np.log(1 + conv["declared_uncertainty"]) / S))

def timestep_block():

    pairs = (("C1", 32, "A64", 64), ("A64", 64, "A28", 128),
             ("C1", 32, "A28", 128))

    band = []
    for tone, f in cp.tones_of("A64"):
        raw = np.log(cp.read("C1", tone, corrected=False)["rho"]
                     / cp.read("A64", tone, corrected=False)["rho"])
        cor = np.log(cp.read("C1", tone)["rho"] / cp.read("A64", tone)["rho"])
        band.append(dict(tone=tone, f=f,
                         raw_rel=float(np.exp(raw.real) - 1),
                         raw_deg=float(np.degrees(raw.imag)),
                         cor_rel=float(np.exp(cor.real) - 1),
                         cor_deg=float(np.degrees(cor.imag))))
    band.sort(key=lambda x: x["f"])

    levels = []
    for a, na, b, nb in pairs:
        for tone in ("F02", "T180"):
            if not (cp.registry().get(a, {}).get(tone)
                    and cp.registry().get(b, {}).get(tone)):
                continue
            raw = np.log(cp.read(a, tone, corrected=False)["rho"]
                         / cp.read(b, tone, corrected=False)["rho"])
            cor = np.log(cp.read(a, tone)["rho"] / cp.read(b, tone)["rho"])
            levels.append(dict(tone=tone, f=cp.read(a, tone)["f"],
                               na=na, nb=nb,
                               raw_rel=float(np.exp(raw.real) - 1),
                               raw_deg=float(np.degrees(raw.imag)),
                               cor_rel=float(np.exp(cor.real) - 1),
                               cor_deg=float(np.degrees(cor.imag))))

    deep = []
    for tone in ("F02", "T180"):
        raw = np.log(cp.read("D1", tone, corrected=False)["rho"]
                     / cp.read("C64", tone, corrected=False)["rho"])
        cor = np.log(cp.read("D1", tone)["rho"] / cp.read("C64", tone)["rho"])
        deep.append(dict(tone=tone, f=cp.read("D1", tone)["f"],
                         raw_rel=float(np.exp(raw.real) - 1),
                         raw_deg=float(np.degrees(raw.imag)),
                         cor_rel=float(np.exp(cor.real) - 1),
                         cor_deg=float(np.degrees(cor.imag))))

    gap = max(abs(b["cor_rel"]) for b in band)
    residual = gap / (1.0 - 0.25)

    tones = [t for t, _ in cp.common_tones(*cp.depth_series(),
                                           *cp.diameter_series())]
    predict = _separable_surface(cp.depth_series(), cp.diameter_series(),
                                 tones, cp.CASES["C1"]["hd"])
    zg = np.linspace(0.8, 5.0, 4201)
    sel = [t for t, _ in cp.tones_of("C64")]

    def invert_z(get):
        crit = np.zeros_like(zg)
        for t in sel:
            crit += np.abs(predict(t, zg, cp.CASES["D1"]["hd"]) - get(t)) ** 2
        j = int(np.argmin(crit))
        return float(zg[j]), float(np.sqrt(crit[j] / len(sel)))

    z32, res32 = invert_z(lambda t: measured("D1", t))
    z64, res64 = invert_z(lambda t: np.log(cp.read("C64", t)["rho"]
                                           / cp.read("A64", t)["rho"]))
    z32r, res32r = invert_z(
        lambda t: np.log(cp.read("D1", t, corrected=False)["rho"]
                         / cp.read("C1", t, corrected=False)["rho"]))
    z64r, res64r = invert_z(
        lambda t: np.log(cp.read("C64", t, corrected=False)["rho"]
                         / cp.read("A64", t, corrected=False)["rho"]))
    return dict(band=band, levels=levels, deep=deep,
                raw_rel_max=max(abs(b["raw_rel"]) for b in band),
                raw_deg_max=max(abs(b["raw_deg"]) for b in band),
                cor_rel_max=gap,
                cor_deg_max=max(abs(b["cor_deg"]) for b in band),
                deep_raw_deg_max=max(abs(b["raw_deg"]) for b in deep),
                deep_cor_rel_max=max(abs(b["cor_rel"]) for b in deep),
                residual=residual,
                z_shift=abs(z64 - z32), res32=res32, res64=res64,
                z_shift_raw=abs(z64r - z32r), res32_raw=res32r,
                res64_raw=res64r)

def misalignment_block(unc):

    tones = [t for t, _ in cp.common_tones(*cp.depth_series(),
                                           *cp.diameter_series())]
    predict = _separable_surface(cp.depth_series(), cp.diameter_series(),
                                 tones, cp.CASES["C1"]["hd"])
    zg = np.linspace(0.8, 5.0, 2101)
    dg = np.exp(np.linspace(np.log(0.3), np.log(5.0), 601))
    Z, D = np.meshgrid(zg, dg, indexing="ij")
    cases = [c for c in cp.CASES if c.startswith("O") and c.endswith("1")]
    cases.sort(key=lambda c: cp.CASES[c]["spotx"])
    rows = []
    for cid in cases:
        if not cp.registry().get(cid):
            continue
        sel = [t for t, _ in cp.tones_of(cid)]
        crit = np.zeros_like(Z)
        for t in sel:
            m = np.log(cp.read(cid, t, point=(0.0, cp.CASES[cid]["spoty"]))["rho"]
                       / cp.read("C1", t)["rho"])
            crit += np.abs(predict(t, Z, D) - m) ** 2
        i = np.unravel_index(np.argmin(crit), crit.shape)
        rows.append(dict(cid=cid, x=cp.CASES[cid]["spotx"],
                         z_hat=float(zg[i[0]]), d_hat=float(dg[i[1]]),
                         residual=float(np.sqrt(crit[i] / len(sel))),
                         at_edge=bool(i[0] in (0, len(zg) - 1)
                                      or i[1] in (0, len(dg) - 1))))
    z_true = cp.CASES["C1"]["hdepth"]
    near = [r for r in rows if abs(r["x"]) <= 2.0 and r["x"] != 0]
    return dict(rows=rows, unc=unc,
                z_err_aligned=float(1e3 * abs(
                    [r for r in rows if r["x"] == 0][0]["z_hat"] - z_true)),
                z_err_two=float(1e3 * max(abs(r["z_hat"] - z_true)
                                          for r in near)),
                res_two=float(max(r["residual"] for r in near)))

def scaled_plate_block():

    rows = []
    for t5, t10, _ in cp.pair_by_group("SC1", "GA"):
        a, b = cp.read("SC1", t5), cp.read("GA", t10)
        rows.append(dict(mu_over_L=a["mu"] / cp.CASES["SC1"]["th"],
                         rho5=abs(a["rho"]), rho10=abs(b["rho"]),
                         rel=float(abs(abs(a["rho"]) - abs(b["rho"]))
                                   / abs(b["rho"])),
                         dphase=float(abs(np.degrees(
                             np.angle(a["rho"] / b["rho"]))))))
    return dict(rows=rows, rel_max=max(r["rel"] for r in rows),
                dphase_max=max(r["dphase"] for r in rows))

def material_block():

    fe = dict(cp.tones_of("C1"))
    rows = []
    for tag, f_al in cp.tones_of("AL1"):
        t_fe = next(t for t in fe if t[1:] == tag[1:])
        a, b = cp.read("C1", t_fe), cp.read("AL1", tag)
        rows.append(dict(f_fe=a["f"], f_al=b["f"], mu=b["mu"],
                         rel=float(abs(abs(a["rho"]) - abs(b["rho"]))
                                   / abs(a["rho"])),
                         dphase=float(abs(np.degrees(
                             np.angle(b["rho"] / a["rho"]))))))
    bi_fe = cp.HCONV * cp.CASES["C1"]["th"] * 1e-3 / cp.K_STEEL
    bi_al = cp.HCONV * cp.CASES["C1"]["th"] * 1e-3 / cp.K_AL
    return dict(rows=rows, rel_max=max(r["rel"] for r in rows),
                rel_min=min(r["rel"] for r in rows),
                dphase_max=max(r["dphase"] for r in rows),
                alpha_ratio=cp.ALPHA_AL / cp.ALPHA, bi_fe=bi_fe, bi_al=bi_al)

def _inversion_grid(nz=701, nd=331):

    tones = [t for t, _ in cp.common_tones(*cp.depth_series(),
                                           *cp.diameter_series())]
    predict = _separable_surface(cp.depth_series(), cp.diameter_series(),
                                 tones, cp.CASES["C1"]["hd"])
    zs = [cp.CASES[c]["hdepth"] for c in cp.depth_series()]
    ds = [cp.CASES[c]["hd"] for c in cp.diameter_series()]
    zg = np.linspace(min(zs) - 0.25, max(zs) + 0.75, nz)
    dg = np.exp(np.linspace(np.log(min(ds) - 0.1), np.log(max(ds) + 0.5), nd))
    Z, D = np.meshgrid(zg, dg, indexing="ij")
    P = {t: predict(t, Z, D) for t in tones}
    A = sum(np.abs(P[t]) ** 2 for t in tones)

    def invert(m):
        c = A - 2.0 * np.real(sum(np.conj(P[t]) * m[t] for t in tones))
        i = np.unravel_index(np.argmin(c), c.shape)
        return float(zg[i[0]]), float(dg[i[1]])

    return tones, predict, invert

def noise_block(n_draw=300, levels=(1, 3, 10), seed=0):

    tones, _, invert = _inversion_grid()
    sigma = cp.noise_floor() * 1e-3
    rng = np.random.default_rng(seed)
    ref = {t: cp.read("C1", t)["rho"] for t in tones}
    zs_grid = [cp.CASES[c]["hdepth"] for c in cp.depth_series()]
    ds_grid = [cp.CASES[c]["hd"] for c in cp.diameter_series()]
    z_lo, z_hi = min(zs_grid) - 0.25, max(zs_grid) + 0.75
    d_lo, d_hi = min(ds_grid) - 0.1, max(ds_grid) + 0.5
    out = []
    for cid in ("C1", "V1", "W1"):
        base = {t: cp.read(cid, t) for t in tones}
        z_true = cp.CASES[cid]["hdepth"]
        d_true = cp.CASES[cid]["hd"]
        z0, d0 = invert({t: np.log(base[t]["rho"] / ref[t]) for t in tones})
        for k in levels:
            s = sigma * k * np.sqrt(2.0)
            zs = np.empty(n_draw)
            dd = np.empty(n_draw)
            for n in range(n_draw):
                m = {}
                for t in tones:
                    dr = base[t]["d_refl"] + s * (rng.normal() + 1j * rng.normal())
                    dt = base[t]["d_tran"] + s * (rng.normal() + 1j * rng.normal())
                    m[t] = np.log((dr / dt) / ref[t])
                zs[n], dd[n] = invert(m)

            edge = float(np.mean((zs <= z_lo + 1e-9) | (zs >= z_hi - 1e-9)
                                 | (dd <= d_lo + 1e-9) | (dd >= d_hi - 1e-9)))
            out.append(dict(cid=cid, level=k, z_true=z_true, d_true=d_true,
                            z_clean=z0, d_clean=d0,
                            z_bias=float(zs.mean() - z0),
                            z_std=float(zs.std()),
                            d_bias=float((dd.mean() - d0) / d_true),
                            d_std=float(dd.std() / d_true),
                            edge=edge))
    ref_row = [o for o in out if o["cid"] == "C1" and o["level"] == 1][0]
    lvl1 = [o for o in out if o["level"] == 1]
    return dict(rows=out, sigma_mK=cp.noise_floor(),
                sigma_diff_mK=float(cp.noise_floor() * np.sqrt(2.0)),
                n_draw=n_draw,
                z_std_nominal=max(o["z_std"] for o in lvl1),
                z_bias_nominal=max(abs(o["z_bias"]) for o in lvl1),
                z_total_nominal=max(float(np.hypot(o["z_std"], o["z_bias"]))
                                    for o in lvl1),
                edge_max=max(o["edge"] for o in out),
                z_std_ten=max(o["z_std"] for o in out if o["level"] == 10),
                ref=ref_row)

def model_error_block():

    tones, _, invert = _inversion_grid(nz=2001, nd=601)
    fs = np.array([cp.read("C1", t)["f"] for t in tones])
    o = np.argsort(fs)
    tones = [tones[i] for i in o]
    fs = fs[o]
    lf = np.log(fs)

    def shifted(cid, factor):

        rho = np.array([cp.read(cid, t)["rho"] for t in tones])
        lr = np.log(np.abs(rho))
        ph = np.unwrap(np.angle(rho))
        target = np.clip(lf + np.log(factor), lf[0], lf[-1])
        lr_s = np.interp(target, lf, lr)
        ph_s = np.interp(target, lf, ph)
        ref = np.array([cp.read("C1", t)["rho"] for t in tones])
        lr_r = np.log(np.abs(ref))
        ph_r = np.unwrap(np.angle(ref))
        return {t: (lr_s[i] - lr_r[i]) + 1j * (ph_s[i] - ph_r[i])
                for i, t in enumerate(tones)}

    rows = []
    for cid in ("C1", "V1"):
        z0, d0 = invert(shifted(cid, 1.0))
        for label, eps in (("alpha", 0.05), ("alpha", 0.10),
                           ("thickness", 0.02), ("thickness", 0.05)):
            if label == "alpha":

                factor = 1.0 / (1.0 + eps)
                m = shifted(cid, factor)
            else:

                factor = (1.0 + eps) ** 2
                m = shifted(cid, factor)
            z, d = invert(m)
            rows.append(dict(cid=cid, what=label, eps=eps,
                             dz=float(1e3 * (z - z0)),
                             dd=float(100 * (d - d0) / cp.CASES[cid]["hd"])))
    return dict(rows=rows,
                dz_alpha5=max(abs(x["dz"]) for x in rows
                              if x["what"] == "alpha" and x["eps"] == 0.05),
                dz_alpha10=max(abs(x["dz"]) for x in rows
                               if x["what"] == "alpha" and x["eps"] == 0.10),
                dz_thick2=max(abs(x["dz"]) for x in rows
                              if x["what"] == "thickness" and x["eps"] == 0.02),
                dz_thick5=max(abs(x["dz"]) for x in rows
                              if x["what"] == "thickness" and x["eps"] == 0.05))

def total_budget_block(est, noise, model, ts, budget):

    est_of = {r["mode"]: r for r in est["rows"]}
    noise_of = {(r["cid"], r["level"]): r for r in noise["rows"]}
    label = {"V1": "unseen pair, deep and thin",
             "W1": "unseen pair, shallow and thin"}
    rows = []
    for cid in ("V1", "W1"):
        e = est_of[label[cid]]
        n = noise_of[(cid, 1)]
        terms = [
            ("surrogate", abs(1e3 * (e["z_hat"] - e["z_true"]))),
            ("noise", float(np.hypot(1e3 * n["z_std"], 1e3 * n["z_bias"]))),
            ("diffusivity, 5 \\%", model["dz_alpha5"]),
            ("thickness, 2 \\%", model["dz_thick2"]),
            ("time step", 1e3 * ts["z_shift"]),
            ("mismatch, 10 \\%", budget["dz_um"][1]),
        ]
        rss = float(np.sqrt(sum(v ** 2 for _, v in terms)))
        rows.append(dict(cid=cid, terms=terms, rss=rss,
                         numerical=terms[0][1]))
    return dict(rows=rows, names=[t[0] for t in rows[0]["terms"]],
                rss_max=max(r["rss"] for r in rows),
                numerical_max=max(r["numerical"] for r in rows))

def _dphi_at_spot(cid, tone):

    c = cp.CASES[cid]
    ba, _ = cp.registry()[cid][tone]
    xyz = cp.stores()[ba][f"{cid}_{tone}_xyz"]
    front, _ = cp.face_masks(xyz, c["th"])
    d = cp.phase_contrast(cid, tone)
    p = xyz[front] * 1e3
    q = (p[:, 0] - c.get("spotx", 0.0)) ** 2 + (p[:, 1] - c["spoty"]) ** 2
    return float(d[front][int(np.argmin(q))])

def _cross(xs, ys):

    for i in range(len(xs) - 1):
        if np.isfinite(ys[i]) and np.isfinite(ys[i + 1]) and ys[i] * ys[i + 1] < 0:
            t = (0.0 - ys[i]) / (ys[i + 1] - ys[i])
            return float(np.exp(np.log(xs[i]) + t * (np.log(xs[i + 1])
                                                     - np.log(xs[i]))))
    return None

def benchmark_block():

    cases = list(cp.depth_series()) + ["V1", "W1"]
    tones = [t for t, _ in cp.tones_of("C1")]
    fs = np.array([cp.read("C1", t)["f"] for t in tones])
    o = np.argsort(fs)
    tones = [tones[i] for i in o]
    fs = fs[o]

    blind = {}
    for cid in cases:
        y = np.array([_dphi_at_spot(cid, t) for t in tones])
        blind[cid] = _cross(fs, y)

    k = cp.CASES["C1"]["hdepth"] / cp.mu(blind["C1"])

    widths = {cid: np.array([cp.fwhm_along_x(
        *(lambda tr: (tr[1], np.abs(cp.phase_contrast(cid, t)),
                      cp.face_masks(tr[1], cp.CASES[cid]["th"])[1],
                      cp.CASES[cid]["spoty"]))(cp.differential(cid, t)))[0]
        for t in tones]) for cid in cases}

    f_cal = _cross(fs, widths["C1"] - cp.CASES["C1"]["hd"])
    lf = np.log(fs)

    rows = []
    for cid in cases:
        zt = cp.CASES[cid]["hdepth"]
        dt = cp.CASES[cid]["hd"]
        z_hat = k * cp.mu(blind[cid]) if blind[cid] else None
        w = float(np.exp(np.interp(np.log(f_cal), lf, np.log(widths[cid]))))
        rows.append(dict(cid=cid, z_true=zt, d_true=dt, f_blind=blind[cid],
                         z_hat=z_hat, z_err=None if z_hat is None
                         else float(1e3 * (z_hat - zt)),
                         d_hat=w, d_err=float(100 * (w - dt) / dt)))
    held = [x for x in rows if x["cid"] in ("V1", "W1")]
    return dict(rows=rows, k=float(k), f_cal=float(f_cal),
                z_err_max=max(abs(x["z_err"]) for x in held),
                d_err_max=max(abs(x["d_err"]) for x in held),
                k_min=float(min(cp.CASES[x["cid"]]["hdepth"]
                                / cp.mu(x["f_blind"]) for x in rows)),
                k_max=float(max(cp.CASES[x["cid"]]["hdepth"]
                                / cp.mu(x["f_blind"]) for x in rows)))

def compute():
    r = {}
    r["geometry"] = geometry()
    r["campaign"] = campaign_summary()
    r["reference"] = reference_band()
    r["convergence"] = convergence_block()
    r["depth"] = depth_block()
    r["diameter"] = diameter_block()
    r["extent"] = extent_block()
    r["ident"] = identifiability_block(r["depth"], r["diameter"])
    unc = r["convergence"]["declared_uncertainty"]
    tones = [t["tone"] for t in r["ident"]["rows"]]

    worst = min(r["ident"]["rows"], key=lambda x: abs(x["tau"] - x["tau_phase"]))
    r["profiled"] = profiled_residual(
        r["depth"], r["diameter"],
        {"one": [worst["tone"]], "ends": [tones[0], tones[-1]],
         "all": tones}, unc)
    r["estimation"] = estimation_block(unc)
    r["probe"] = surrogate_probe(unc)
    r["similarity"] = similarity_block()
    r["budget"] = budget_block(r["depth"], r["convergence"])
    r["timestep"] = timestep_block()
    r["misalignment"] = misalignment_block(unc)
    r["scaled"] = scaled_plate_block()
    r["material"] = material_block()
    r["noise"] = noise_block()
    r["model_error"] = model_error_block()
    r["total_budget"] = total_budget_block(r["estimation"], r["noise"],
                                           r["model_error"], r["timestep"],
                                           r["budget"])
    r["benchmark"] = benchmark_block()
    return r

def _num(x, dec):
    return f"{x:.{dec}f}"

def latex_macros(r):

    c, ref, cv_, dp = r["campaign"], r["reference"], r["convergence"], r["depth"]
    dm, ex, idt, sm, bg = (r["diameter"], r["extent"], r["ident"],
                           r["similarity"], r["budget"])
    lin = cv_["fits"]["rho_linear"]
    nn = cv_["fits"]["rho_nearest"]
    dr = cv_["fits"]["d_refl_linear"]
    dt = cv_["fits"]["d_tran_linear"]
    P = r["profiled"]
    m = {}

    def put(name, value):

        m["num" + name] = value

    G = r["geometry"]
    put("GeoLength", _num(G["length"], 0))
    put("GeoWidth", _num(G["width"], 0))
    put("GeoThick", _num(G["thickness"], 0))
    put("GeoDiam", _num(G["diameter"], 1))
    put("GeoDepth", _num(G["depth"], 2))
    put("GeoSpotY", _num(G["spot_y"], 0))
    put("GeoSpotR", _num(G["spot_r"], 0))
    put("GeoChanA", _num(G["chan_y0"], 0))
    put("GeoChanB", _num(G["chan_y1"], 0))
    put("GeoClear", _num(G["clearance"], 2))
    put("RatioRef", _num(G["diameter"] / G["clearance"], 2))
    put("DmuTop", _num(G["diameter"] / cp.mu(r["reference"]["f_max"]), 2))

    put("Nruns", f"{c['n_runs']}")
    put("Nbatches", f"{c['n_batches']}")
    put("Alpha", f"{c['alpha']:.2e}".replace("e-0", "\\times 10^{-") + "}")
    put("Kcond", _num(cp.K_STEEL, 0))
    put("RhoDens", _num(cp.RHO_STEEL, 0))
    put("Cheat", _num(cp.CP_STEEL, 0))
    put("Cvol", f"{cp.CV_STEEL:.2e}".replace("e+0", "\\times 10^{") + "}")
    put("NoiseFloor", _num(c["noise_mK"], 3))
    put("Qmax", _num(cp.QMAX / 1e3, 0))
    put("Netd", _num(c["netd_mK"], 0))
    put("Fps", _num(c["fps"], 0))
    put("Duration", _num(c["duration"], 0))
    put("Binning", f"{c['binning']}")

    put("RefTones", f"{ref['n_tones']}")
    put("RefFmin", _num(ref["f_min"], 4))
    put("RefFmax", _num(ref["f_max"], 3))
    put("RefFspan", _num(ref["f_span"], 0))
    put("RefMuLmax", _num(ref["mu_over_L_max"], 2))
    put("RefMuLmin", _num(ref["mu_over_L_min"], 2))
    put("TrueWidth", _num(ref["true_width"], 1))
    put("WidthReflMin", _num(ref["width_refl_min"], 2))
    put("WidthReflMax", _num(ref["width_refl_max"], 2))
    put("WidthTranMin", _num(ref["width_tran_min"], 2))
    put("WidthTranMax", _num(ref["width_tran_max"], 2))
    put("WidthReflSpan", _num(ref["width_refl_span"], 0))
    put("WidthReflUndef", f"{ref['width_refl_undefined']}")
    put("WidthTranUndef", f"{ref['width_tran_undefined']}")
    put("WidthTranSpan", _num(ref["width_tran_span"], 1))
    put("PeakReflMax", _num(ref["peak_refl_max"], 1))
    put("PeakReflMin", _num(ref["peak_refl_min"], 2))
    put("PeakTranMin", _num(ref["peak_tran_min"], 2))
    put("DphiTranMin", _num(abs(ref["dphi_tran_max"]), 1))
    put("DphiTranMax", _num(abs(ref["dphi_tran_min"]), 1))
    put("DphiReflPos", _num(ref["dphi_refl_max"], 2))
    put("DphiReflNeg", _num(abs(ref["dphi_refl_min"]), 2))
    put("DphiReflZeroF", _num(ref["f_dphi_refl_zero"], 2))
    put("DphiReflZeroV", _num(ref["dphi_refl_abs_min"], 2))
    put("WidthReflCross", f"{ref['width_refl_cross']}")
    put("WidthTranCross", f"{ref['width_tran_cross']}")
    put("DphiPeakF", _num(ref["f_dphi_peak"], 2))
    put("DphiPeakMuL", _num(ref["mul_dphi_peak"], 2))
    put("DphiTranEnd", _num(ref["dphi_tran_end"], 1))
    est = r["estimation"]
    a, b_ = est["rows"][0], est["rows"][1]
    put("EstNtones", f"{est['n_tones']}")
    put("EstZhat", _num(a["z_hat"], 2))
    put("EstDhat", _num(a["d_hat"], 2))
    put("EstRes", _num(100 * a["residual"], 2))
    put("EstZlo", _num(a["z_lo"], 2)); put("EstZhi", _num(a["z_hi"], 2))
    put("EstDlo", _num(a["d_lo"], 2)); put("EstDhi", _num(a["d_hi"], 2))
    put("LooZhat", _num(b_["z_hat"], 3))
    put("LooZerr", _num(1e3 * abs(b_["z_hat"] - b_["z_true"]), 0))
    put("LooDhat", _num(b_["d_hat"], 2))
    put("LooDtrue", _num(b_["d_true"], 1))
    put("LooDlo", _num(b_["d_lo"], 2)); put("LooDhi", _num(b_["d_hi"], 2))
    put("LooZlo", _num(b_["z_lo"], 2)); put("LooZhi", _num(b_["z_hi"], 2))
    put("LooRes", _num(100 * b_["residual"], 2))
    put("LooDerrPc", _num(100 * abs(b_["d_hat"] / b_["d_true"] - 1), 0))
    put("LooDmiss", _num(100 * (b_["d_lo"] / b_["d_true"] - 1), 0))
    for tag, row in zip(("Ext", "Mov", "Deep", "Shal"), est["rows"][2:]):
        put(f"Blind{tag}N", f"{row['n_tones']}")
        put(f"Blind{tag}Ztrue", _num(row["z_true"], 2))
        put(f"Blind{tag}Dtrue", _num(row["d_true"], 2))
        put(f"Blind{tag}Zerr", _num(1e3 * abs(row["z_hat"] - row["z_true"]), 0))
        put(f"Blind{tag}Derr", _num(100 * abs(row["d_hat"] / row["d_true"] - 1), 0))
        put(f"Blind{tag}Res", _num(100 * row["residual"], 2))
        put(f"Blind{tag}Rms", _num(100 * row["surface_rms"], 1))
        if row["z_lo"] is None:
            put(f"Blind{tag}In", "empty")
        elif (row["z_lo"] <= row["z_true"] <= row["z_hi"]
              and row["d_lo"] <= row["d_true"] <= row["d_hi"]):
            put(f"Blind{tag}In", "inside")
        else:
            put(f"Blind{tag}In", "outside")
    put("LooRms", _num(100 * est["rows"][1]["surface_rms"], 1))
    put("PeakTranMax", _num(ref["peak_tran_max"], 1))
    put("RhoMinMuL", _num(ref["mu_over_L_at_rho_min"], 2))

    put("ConvN", f"{lin['n']}")
    put("ConvTones", f"{lin['tones']}")
    put("ConvDof", f"{lin['dof']}")
    put("ConvXmin", _num(lin["x_min"], 3))
    put("ConvXmax", _num(lin["x_max"], 2))
    put("ConvXspan", _num(lin["x_max"] / lin["x_min"], 0))
    for tag, o in (("Rho", lin), ("Drefl", dr), ("Dtran", dt)):

        ok = o.get("identified", True)
        put(f"P{tag}", _num(o["p"], 2) if ok else "n/a")
        put(f"P{tag}Lo", _num(o["lo"], 2))
        put(f"P{tag}Hi", _num(o["hi"], 2))
        put(f"P{tag}Int", (f"$[{_num(o['lo'], 2)},\\,{_num(o['hi'], 2)}]$"
                           if ok else "n/a"))
        put(f"Err{tag}", _num(100 * o["err_median"], 2) if ok else "n/a")
        put(f"Chg{tag}", _num(100 * o["change"]["median"], 1))
    put("ErrRhoMin", _num(100 * lin["err_min"], 2))
    put("ErrRhoMax", _num(100 * lin["err_max"], 2))
    put("PRhoNN", _num(nn["p"], 2))
    put("PRhoNNLo", _num(nn["lo"], 2))
    put("PRhoNNHi", _num(nn["hi"], 2))
    put("POverlapLo", _num(cv_["overlap"][0], 2))
    put("POverlapHi", _num(cv_["overlap"][1], 2))
    put("OosWorst", _num(100 * cv_["oos"]["worst"], 2))
    put("OosHfine", _num(cp.CASES[cp.CONV_LEVELS["conv3"][-1]]["esz"], 2))
    put("OosHcoarse", _num(cp.CASES[cp.CONV_LEVELS["conv3"][0]]["esz"], 2))
    put("OosHmid", _num(cp.CASES[cp.CONV_LEVELS["conv3"][1]]["esz"], 2))
    put("BridgeResMin", _num(100 * min(p["residual"] for p in cv_["bridge"]["points"]), 2))
    put("BridgeResMax", _num(100 * cv_["bridge"]["residual_max"], 2))
    put("Uncertainty", _num(100 * cv_["declared_uncertainty"], 1))
    put("ShortCoarse", _num(100 * cv_["shortening_coarse"], 1))
    put("ShortFine", _num(100 * cv_["shortening_fine"], 1))
    put("OffCentreN", f"{cv_['off_centre']['n']}")
    put("OffCentreMed", _num(100 * cv_["off_centre"]["median"], 1))
    put("OffCentreP", _num(100 * cv_["off_centre"]["p90"], 1))
    put("HighToneF", _num(cv_["high_tone_check"]["f"], 3))
    put("HighToneMu", _num(cv_["high_tone_check"]["mu"], 2))
    put("HighToneRel", _num(100 * cv_["high_tone_check"]["rel"], 2))

    zs = np.array(dp["z"])
    mid = int(np.argmin(np.abs(zs - cp.CASES["C1"]["hdepth"])))

    row0 = max(dp["rows"], key=lambda r: r["slope"])
    lz = np.log(np.array(row0["rho"]))
    put("DepthN", f"{len(zs)}")
    put("DepthZlo", _num(zs.min(), 2))
    put("DepthZhi", _num(zs.max(), 2))
    put("StepNppA", "32")
    put("StepNppB", "64")
    put("StepNppC", "128")
    put("DepthChordF", _num(row0["f"], 2))
    put("DepthSlopeSteep",
        _num(abs((lz[mid] - lz[0]) / (zs[mid] - zs[0])), 1))
    put("DepthSlopeShallow",
        _num(abs((lz[-1] - lz[mid]) / (zs[-1] - zs[mid])), 1))
    put("DepthSlopeMin", _num(dp["slope_min"], 2))
    put("DepthSlopeMax", _num(dp["slope_max"], 2))
    put("DepthFatMax", _num(dp["f_at_slope_max"], 2))
    put("DepthFactorMax", _num(dp["factor_max"], 0))
    put("DepthSpan", _num(dp["dz"], 2))

    put("DiamN", f"{len(dm['d'])}")
    put("DiamMin", _num(min(dm["d"]), 1))
    put("DiamMax", _num(max(dm["d"]), 1))
    pr = r["probe"]
    put("ProbeNdiam", f"{pr['n_diameters']}")
    put("ProbeNdepth", f"{pr['n_depths']}")
    g = {(x["deg_z"], x["deg_d"]): x for x in pr["grid"]}
    put("ProbeBaseDeep", _num(100 * g[(2, 2)]["off"]["V1"], 1))
    put("ProbeBaseShal", _num(100 * g[(2, 2)]["off"]["W1"], 1))
    put("ProbeCubDeep", _num(100 * g[(2, 3)]["off"]["V1"], 1))
    put("ProbeCubShal", _num(100 * g[(2, 3)]["off"]["W1"], 1))
    put("ProbeQuartDeep", _num(100 * g[(4, 2)]["off"]["V1"], 1))
    put("ProbeQuartShal", _num(100 * g[(4, 2)]["off"]["W1"], 1))
    put("ProbeFloor", _num(100 * pr["floor"], 1))
    put("ProbeBestZ", f"{pr['best']['deg_z']}")
    put("ProbeBestD", f"{pr['best']['deg_d']}")
    put("ProbeLooZ", f"{pr['loo_best_z']}")
    put("ProbeLooD", f"{pr['loo_best_d']}")
    nz = {x["deg"]: x for x in pr["node_z"]}
    nd = {x["deg"]: x for x in pr["node_d"]}
    put("ProbeNodeZpar", _num(100 * nz[2]["rms"], 1))
    put("ProbeNodeZquart", _num(100 * nz[4]["rms"], 2))
    put("ProbeNodeDpar", _num(100 * nd[2]["rms"], 1))
    put("ProbeNodeDcub", _num(100 * nd[3]["rms"], 1))
    put("ProbeLooZpar", _num(100 * nz[2]["loo"], 0))
    put("ProbeLooZquart", _num(100 * nz[4]["loo"], 0))
    put("ProbeLooDpar", _num(100 * nd[2]["loo"], 0))
    put("ProbeLooDcub", _num(100 * nd[3]["loo"], 0))
    put("DiamNodeRms", _num(100 * r["diameter"]["node_rms"], 1))
    put("DiamNodeMax", _num(100 * r["diameter"]["node_max"], 1))
    put("DiamRtwo", _num(dm["r2_max"], 2))
    put("SlopeSmallMin", _num(dm["slope_small_min"], 3))
    put("SlopeSmallMax", _num(dm["slope_small_max"], 3))
    put("SlopeLargeMin", _num(dm["slope_large_min"], 2))
    put("SlopeLargeMax", _num(dm["slope_large_max"], 2))
    put("SlopeGrowth", _num(dm["slope_growth"], 0))
    put("SmallMoveMin", _num(100 * dm["small_move_min"], 1))
    put("SmallMoveMax", _num(100 * dm["small_move_max"], 1))
    put("DiamSmall", _num(dm["d_small"], 1))
    put("DiamNext", _num(dm["d_next"], 1))
    put("RatioSmall", _num(dm["ratio_small"], 2))
    put("RatioNext", _num(dm["ratio_next"], 2))
    put("RatioLarge", _num(dm["ratio_large"], 2))

    put("ExtentPhaseMax", _num(r["extent"]["dphase_max"], 1))
    put("ExtentLogMax", _num(100 * r["extent"]["dlog_max"], 1))
    put("ExtentRelMax", _num(100 * ex["rel_max"], 1))
    put("ExtentRatio", _num(ex["length_ratio"], 2))
    put("ExtentFull", _num(ex["length_full"], 0))
    put("ExtentTrunc", _num(ex["length_truncated"], 0))

    put("IdentTones", f"{idt['n_tones']}")
    put("IdentFspan", _num(idt["f_span"], 0))
    put("TauMin", _num(idt["tau_min"], 3))
    put("TauMax", _num(idt["tau_max"], 3))
    put("TauSpread", _num(100 * idt["tau_spread"], 0))
    put("TauRef", _num(1e3 * idt["tau_ref"], 0))
    put("CondOne", _num(idt["conds"]["one"], 0))
    put("CondEnds", _num(idt["conds"]["ends"], 0))
    put("CondAll", _num(idt["conds"]["all"], 0))
    put("CondBest", _num(idt["cond_one_best"], 0))
    put("CondWorst", _num(idt["cond_one_worst"], 0))
    put("FbestOne", _num(idt["f_one_best"], 2))
    put("FworstOne", _num(idt["f_one_worst"], 2))
    put("CondRatio", _num(idt["cond_one_worst"] / idt["cond_one_best"], 0))
    put("TaufMin", _num(idt["tauf_min"], 3))
    put("TaufMax", _num(idt["tauf_max"], 3))
    put("GapMin", _num(1e3 * idt["gap_min"], 0))
    put("GapMax", _num(1e3 * idt["gap_max"], 0))
    ztrue = cp.CASES["C1"]["hdepth"]
    dtrue = cp.CASES["C1"]["hd"]
    for key, tag in (("one", "One"), ("ends", "Ends"), ("all", "All")):
        p = P[key]
        put(f"Dlo{tag}", "n/a" if p["d_lo"] is None else _num(p["d_lo"], 2))
        put(f"Dhi{tag}", "n/a" if p["d_hi"] is None else _num(p["d_hi"], 2))
        if p.get("z_lo") is not None:
            put(f"Zlo{tag}", _num(p["z_lo"], 3))
            put(f"Zhi{tag}", _num(p["z_hi"], 3))

            put(f"Zerr{tag}", _num(1e3 * max(abs(p["z_hi"] - ztrue),
                                             abs(ztrue - p["z_lo"])), 0))
            put(f"Derr{tag}", _num(max(abs(p["d_hi"] - dtrue),
                                       abs(dtrue - p["d_lo"])), 2))

    ts = r["timestep"]
    put("StepRelMax", _num(100 * ts["raw_rel_max"], 1))
    put("StepDegMax", _num(ts["raw_deg_max"], 1))
    put("StepCorRelMax", _num(100 * ts["cor_rel_max"], 3))
    put("StepCorDegMax", _num(ts["cor_deg_max"], 3))
    put("StepResidual", _num(100 * ts["residual"], 2))
    put("StepGainRel", _num(ts["raw_rel_max"] / ts["cor_rel_max"], 0))
    put("StepDeepDeg", _num(ts["deep_raw_deg_max"], 2))
    put("StepDeepCorRel", _num(100 * ts["deep_cor_rel_max"], 2))
    put("StepZshift", _num(1e3 * ts["z_shift"], 0))
    put("StepResIn", _num(100 * ts["res32"], 2))
    put("StepResOut", _num(100 * ts["res64"], 2))
    put("StepZshiftRaw", _num(1e3 * ts["z_shift_raw"], 0))
    put("StepResInRaw", _num(100 * ts["res32_raw"], 1))
    put("StepResOutRaw", _num(100 * ts["res64_raw"], 1))
    put("StepRelTop", _num(100 * abs(ts["band"][-1]["raw_rel"]), 1))
    put("StepDegTop", _num(abs(ts["band"][-1]["raw_deg"]), 1))
    put("StepRelBot", _num(100 * abs(ts["band"][0]["raw_rel"]), 1))
    put("StepThetaDeg", _num(180.0 / 32, 2))
    lk = cp.lnkappa(32)
    put("StepKappaDeg", _num(abs(np.degrees(lk.imag)), 2))
    hi = max(ts["levels"], key=lambda x: abs(x["raw_rel"]))
    put("StepTripleRel", _num(100 * abs(hi["raw_rel"]), 1))
    put("StepTripleDeg", _num(abs(hi["raw_deg"]), 1))
    put("StepTripleCorRel", _num(100 * abs(hi["cor_rel"]), 3))
    put("StepTripleCorDeg", _num(abs(hi["cor_deg"]), 3))

    mi = r["misalignment"]
    put("MisN", f"{len(mi['rows'])}")
    put("MisXlo", _num(min(x["x"] for x in mi["rows"]), 0))
    put("MisXhi", _num(max(x["x"] for x in mi["rows"]), 0))
    put("MisZaligned", _num(mi["z_err_aligned"], 0))
    put("MisZtwo", _num(mi["z_err_two"], 0))
    put("MisResTwo", _num(100 * mi["res_two"], 1))

    sc = r["scaled"]
    put("ScaledRelMax", _num(100 * sc["rel_max"], 1))
    put("ScaledDegMax", _num(sc["dphase_max"], 1))
    put("ScaledN", f"{len(sc['rows'])}")

    ma = r["material"]
    put("MatRelMin", _num(100 * ma["rel_min"], 2))
    put("MatRelMax", _num(100 * ma["rel_max"], 2))
    put("MatDegMax", _num(ma["dphase_max"], 2))
    put("MatAlphaRatio", _num(ma["alpha_ratio"], 1))
    put("MatFmin", _num(ma["rows"][0]["f_al"], 2))
    put("MatFmax", _num(ma["rows"][-1]["f_al"], 1))
    put("MatBiFe", f"{ma['bi_fe']:.1e}".replace("e-0", "\\times 10^{-") + "}")
    put("MatBiAl", f"{ma['bi_al']:.1e}".replace("e-0", "\\times 10^{-") + "}")
    put("MatKal", _num(cp.K_AL, 0))
    put("MatRhoAl", _num(cp.RHO_AL, 0))
    put("MatCal", _num(cp.CP_AL, 0))

    bm = r["benchmark"]
    put("BenchK", _num(bm["k"], 3))
    put("BenchKmin", _num(bm["k_min"], 3))
    put("BenchKmax", _num(bm["k_max"], 3))
    put("BenchFcal", _num(bm["f_cal"], 2))
    put("BenchZerrMax", _num(bm["z_err_max"], 0))
    put("BenchDerrMax", _num(bm["d_err_max"], 0))
    for x in bm["rows"]:
        if x["cid"] in ("V1", "W1"):
            tag = "Deep" if x["cid"] == "V1" else "Shal"
            put(f"Bench{tag}Zerr", _num(abs(x["z_err"]), 0))
            put(f"Bench{tag}Derr", _num(abs(x["d_err"]), 0))

    tb = r["total_budget"]
    put("TotalRss", _num(tb["rss_max"], 0))
    put("TotalNumerical", _num(tb["numerical_max"], 0))
    for row in tb["rows"]:
        tag = "Deep" if row["cid"] == "V1" else "Shal"
        put(f"Total{tag}Rss", _num(row["rss"], 0))
    nz = r["noise"]
    put("NoiseSigma", _num(nz["sigma_mK"], 3))
    put("NoiseSigmaDiff", _num(nz["sigma_diff_mK"], 3))
    put("NoiseZbias", _num(1e3 * nz["z_bias_nominal"], 0))
    put("NoiseZtotal", _num(1e3 * nz["z_total_nominal"], 0))
    put("NoiseEdgeMax", _num(100 * nz["edge_max"], 0))
    _w1 = [x for x in nz["rows"] if x["cid"] == "W1" and x["level"] == 1][0]
    put("NoiseShalZbias", _num(1e3 * _w1["z_bias"], 0))
    put("NoiseShalZstd", _num(1e3 * _w1["z_std"], 0))
    put("NoiseShalDbias", _num(100 * abs(_w1["d_bias"]), 1))
    put("NoiseShalDstd", _num(100 * _w1["d_std"], 1))
    put("NoiseShalEdge", _num(100 * _w1["edge"], 0))
    put("NoiseZstd", _num(1e3 * nz["z_std_nominal"], 0))
    put("NoiseZstdTen", _num(1e3 * nz["z_std_ten"], 0))
    put("NoiseZstdRef", _num(1e3 * nz["ref"]["z_std"], 0))
    put("NoiseDstdRef", _num(100 * nz["ref"]["d_std"], 1))
    put("NoiseDraws", f"{nz['n_draw']}")

    me = r["model_error"]
    put("ModelZalphaFive", _num(me["dz_alpha5"], 0))
    put("ModelZalphaTen", _num(me["dz_alpha10"], 0))
    put("ModelZthickTwo", _num(me["dz_thick2"], 0))
    put("ModelZthickFive", _num(me["dz_thick5"], 0))

    put("SimRelMin", _num(100 * sm["rel_min"], 1))
    put("SimRelMax", _num(100 * sm["rel_max"], 1))
    put("SimPhaseMax", _num(sm["dphase_max"], 1))
    put("SimLatWorst", _num(sm["lat10_at_worst"], 2))
    put("SimMuLworst", _num(sm["mu_over_L_at_worst"], 2))
    put("SimLatBest", _num(sm["lat5_at_worst"], 2))

    spell = {5: "five", 10: "ten", 25: "twentyfive"}
    for e, v in zip(bg["eps"], bg["dz_um"]):
        put(f"Dz{spell[int(100 * e)]}", _num(v, 0))
    put("RelNoise", _num(100 * bg["rel_noise"], 2))
    put("DzNoise", _num(bg["dz_noise_um"], 1))

    sph = max(abs(r["dph_dz"]) for r in idt["rows"])
    sph_lo = min(abs(r["dph_dz"]) for r in idt["rows"])
    put("DzPhaseBest", _num(1e3 * np.radians(1.0) / sph, 0))
    put("DzPhaseWorst", _num(1e3 * np.radians(1.0) / sph_lo, 0))
    put("DzNumerical", _num(bg["dz_numerical_um"], 0))
    put("SnrSmallest", _num(bg["snr_smallest"], 0))
    put("SnrSmallestWorst", _num(bg["snr_smallest_worst"], 0))
    put("FbestSmall", _num(bg["f_small"], 2))
    put("FbestRef", _num(bg["f_ref"], 2))
    put("SmallestDtran", _num(bg["smallest_d_tran_mK"], 2))
    put("RefDtran", _num(bg["ref_d_tran_mK"], 1))
    put("DiamSmallest", _num(bg["d_smallest"], 1))
    return m

def latex_tables(r):

    out = {}

    ref = r["reference"]["rows"]
    rows = []
    for x in ref:
        w_r = "n/a" if not np.isfinite(x["width_refl"]) else _num(x["width_refl"], 2)
        rows.append(" & ".join([
            _num(x["f"], 4).rstrip("0").rstrip("."), _num(x["mu_over_L"], 2),
            _num(x["peak_refl"], 2), _num(x["peak_tran"], 2),
            w_r, _num(x["width_tran"], 2), _num(x["abs_rho"], 3)]) + r" \\")
    out["ReferenceBand"] = "\n    ".join(rows)

    out["TimeStep"] = "\n    ".join(
        f"{b['f']:g} & {100*b['raw_rel']:+.2f} & {b['raw_deg']:+.2f} & "
        f"{100*b['cor_rel']:+.3f} & {b['cor_deg']:+.3f}" + r" \\"
        for b in r["timestep"]["band"])

    out["TimeStepLevels"] = "\n    ".join(
        f"{x['na']}/{x['nb']} & {x['f']:g} & {100*x['raw_rel']:+.2f} & "
        f"{x['raw_deg']:+.2f} & {100*x['cor_rel']:+.3f} & {x['cor_deg']:+.3f}"
        + r" \\" for x in r["timestep"]["levels"])

    out["Probe"] = "\n    ".join(
        f"{x['deg_z']} & {x['deg_d']} & {100*x['off']['V1']:.2f} & "
        f"{100*x['off']['W1']:.2f}" + r" \\" for x in r["probe"]["grid"])

    tb = r["total_budget"]
    _tb = {row["cid"]: dict(row["terms"]) for row in tb["rows"]}
    out["TotalBudget"] = "\n    ".join(
        [f"{name} & {_tb['V1'][name]:.0f} & {_tb['W1'][name]:.0f}" + r" \\"
         for name in tb["names"]]
        + [r"\midrule"]
        + [r"sum in quadrature & "
           + " & ".join(f"{row['rss']:.0f}" for row in tb["rows"]) + r" \\"])

    out["Benchmark"] = "\n    ".join(
        f"{x['cid']} & {x['z_true']:.2f} & {x['z_hat']:.2f} & {x['z_err']:+.0f} & "
        f"{x['d_true']:.2f} & {x['d_hat']:.2f} & {x['d_err']:+.0f}" + r" \\"
        for x in r["benchmark"]["rows"])

    out["Noise"] = "\n    ".join(
        f"{x['cid']} & {x['level']} & {1e3*x['z_bias']:+.0f} & {1e3*x['z_std']:.0f} & "
        f"{100*x['d_bias']:+.1f} & {100*x['d_std']:.1f}" + r" \\"
        for x in r["noise"]["rows"])

    sym = {"alpha": r"$\alpha$", "thickness": "$L$"}
    out["ModelError"] = "\n    ".join(
        x["cid"] + " & " + sym[x["what"]] + " & "
        + f"{100*x['eps']:.0f} & {x['dz']:+.0f} & {x['dd']:+.1f}" + r" \\"
        for x in r["model_error"]["rows"])

    out["Misalign"] = "\n    ".join(
        f"{x['x']:+g} & {x['z_hat']:.2f} & {x['d_hat']:.2f} & "
        f"{100*x['residual']:.1f}" + r" \\"
        for x in r["misalignment"]["rows"])

    dm = r["diameter"]
    tone_idx = [0, len(dm["rows"]) // 2, len(dm["rows"]) - 1]
    rows = []
    for i, cid in enumerate(dm["cases"]):
        cells = [_num(dm["d"][i], 1), _num(dm["clearance"][i], 2),
                 _num(dm["ratio"][i], 2)]
        cells += [_num(dm["rows"][j]["rho"][i], 3) for j in tone_idx]
        rows.append(" & ".join(cells) + r" \\")
    out["DiameterRows"] = "\n    ".join(rows)
    out["DiameterHeadA"] = _num(dm["rows"][tone_idx[0]]["f"], 3).rstrip("0").rstrip(".")
    out["DiameterHeadB"] = _num(dm["rows"][tone_idx[1]]["f"], 3).rstrip("0").rstrip(".")
    out["DiameterHeadC"] = _num(dm["rows"][tone_idx[2]]["f"], 3).rstrip("0").rstrip(".")

    dp = r["depth"]
    rows = []
    for x in dp["rows"]:
        cells = [_num(x["f"], 4).rstrip("0").rstrip("."), _num(x["mu_over_L"], 2)]
        cells += [_num(v, 3) for v in x["rho"]]
        cells += [_num(x["slope"], 2), _num(x["factor"], 0)]
        rows.append(" & ".join(cells) + r" \\")
    out["DepthRows"] = "\n    ".join(rows)
    rows = []
    for x in r["estimation"]["rows"]:
        iv = (lambda lo, hi: "empty" if lo is None
              else f'[{_num(lo, 2)},\,{_num(hi, 2)}]')
        rows.append(" & ".join([
            x["mode"], f'{x["n_tones"]}',
            _num(x["z_true"], 2), _num(x["z_hat"], 2),
            iv(x["z_lo"], x["z_hi"]),
            _num(x["d_true"], 1), _num(x["d_hat"], 2),
            iv(x["d_lo"], x["d_hi"])]) + r" \\")
    out["EstimationRows"] = "\n    ".join(rows)

    out["DepthZa"] = _num(dp["z"][0], 2)
    out["DepthZb"] = _num(dp["z"][len(dp["z"]) // 2], 2)
    out["DepthZc"] = _num(dp["z"][-1], 2)
    return out

def write(r):
    with open(os.path.join(OUT, "results.json"), "w") as fh:
        json.dump(r, fh, indent=1, default=float)
    m = latex_macros(r)
    lines = ["% numbers.tex -- generated by code/results.py, do not edit.",
             "% Every macro below is computed from data/*.npz. Re-run",
             "%     python3 code/results.py",
             "% and recompile to refresh them.", ""]
    for k in sorted(m):
        lines.append(f"\\newcommand{{\\{k}}}{{{m[k]}}}")
    t = latex_tables(r)
    for k in sorted(t):
        body = t[k].replace("\\n", "\n")
        lines.append(f"\\newcommand{{\\tab{k}}}{{%\n    {body}}}")
    with open(os.path.join(OUT, "numbers.tex"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    return m

def main():
    quiet = "--quiet" in sys.argv
    if cp.verify(verbose=not quiet):
        raise SystemExit("verification failed, nothing was written")
    r = compute()
    m = write(r)
    if quiet:
        return
    print(f"\n{len(m)} macros written to numbers.tex, "
          f"full structure in results.json\n")
    print(f"campaign: {r['campaign']['n_runs']} runs over "
          f"{r['campaign']['n_batches']} batches")
    lin = r["convergence"]["fits"]["rho_linear"]
    print(f"order of convergence of |rho|: p = {lin['p']:.2f} "
          f"[{lin['lo']:.2f}, {lin['hi']:.2f}] on {lin['dof']} d.o.f.")
    print(f"out of sample prediction, worst case: "
          f"{100*r['convergence']['oos']['worst']:.2f} %")
    print(f"declared numerical uncertainty: "
          f"{100*r['convergence']['declared_uncertainty']:.1f} %")
    print(f"depth sensitivity: {r['depth']['slope_min']:.2f} to "
          f"{r['depth']['slope_max']:.2f} per mm")
    print(f"trade-off direction tau: {r['ident']['tau_min']:.3f} to "
          f"{r['ident']['tau_max']:.3f} mm, spread "
          f"{100*r['ident']['tau_spread']:.0f} % over a factor "
          f"{r['ident']['f_span']:.0f} in frequency")

if __name__ == "__main__":
    main()
