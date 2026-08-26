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

    cases = cp.DEPTH_SERIES
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
    small_move = []
    for tone, f in cp.common_tones(cases[0], cases[1]):
        a = abs(cp.read(cases[0], tone)["rho"])
        b = abs(cp.read(cases[1], tone)["rho"])
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

    tones = [t for t, _ in cp.common_tones(*cp.DEPTH_SERIES, *cp.diameter_series())]
    dz_rows = {r["tone"]: r for r in depth["rows"]}
    dd_rows = {r["tone"]: r for r in diam["rows"]}
    tones.sort(key=lambda t: dz_rows[t]["f"])

    raw = np.array([np.angle(cp.read(cp.DEPTH_SERIES[-1], t)["rho"]
                             / cp.read(cp.DEPTH_SERIES[0], t)["rho"])
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

def _separable_surface(depth_cases, diam_cases, tones, d_ref, deg_d=None):

    def logs(cases, tone):
        ref = cp.read("C1", tone)["rho"]
        return np.array([np.log(cp.read(c, tone)["rho"] / ref) for c in cases])

    zs = np.array([cp.CASES[c]["hdepth"] for c in depth_cases], float)
    ds = np.array([cp.CASES[c]["hd"] for c in diam_cases], float)
    o = np.argsort(zs)
    zs = zs[o]
    depth_cases = [depth_cases[i] for i in o]
    deg_z = 2 if len(zs) >= 3 else 1

    if deg_d is None:
        deg_d = 2 if len(ds) >= 3 else 1
    x = np.log(ds)
    cz = {t: np.polyfit(zs, logs(depth_cases, t), deg_z) for t in tones}
    cd = {t: np.polyfit(x, logs(diam_cases, t), deg_d) for t in tones}

    def predict(tone, z, d):
        base = np.polyval(cz[tone], z)
        corr = (np.polyval(cd[tone], np.log(d))
                - np.polyval(cd[tone], np.log(d_ref)))
        return base + corr

    return predict

def measured(cid, tone):

    return np.log(cp.read(cid, tone)["rho"] / cp.read("C1", tone)["rho"])

def estimate(target, depth_cases, diam_cases, tones, unc, d_ref):

    predict = _separable_surface(depth_cases, diam_cases, tones, d_ref)
    meas = {t: measured(target, t) for t in tones}
    zs = [cp.CASES[c]["hdepth"] for c in cp.DEPTH_SERIES]
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

    tones = [t for t, _ in cp.common_tones(*cp.DEPTH_SERIES, *cp.diameter_series())]
    f_of = {round(cp.read("C1", t)["f"], 4): t for t in tones}
    d_ref = cp.CASES["C1"]["hd"]
    dep, dia = list(cp.DEPTH_SERIES), list(cp.diameter_series())

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

    tones = [t for t, _ in cp.common_tones(*cp.DEPTH_SERIES, *cp.diameter_series())]
    dep, dia = list(cp.DEPTH_SERIES), cp.diameter_series()
    d_ref = cp.CASES["C1"]["hd"]
    ds = np.array([cp.CASES[c]["hd"] for c in dia])
    out = {}
    for deg in (2, 3):

        off = {c: [] for c, _ in cp.HELD_OUT
               if {t for t, _ in cp.tones_of(c)} >= set(tones)}
        node = []
        for t in tones:
            y = np.array([measured(c, t) for c in dia])
            x = np.log(ds)
            cd = np.polyfit(x, y.real, deg) + 1j * np.polyfit(x, y.imag, deg)
            fit = np.polyval(cd.real, x) + 1j * np.polyval(cd.imag, x)
            node.extend(np.abs(fit - y))
        pr = _separable_surface(dep, dia, tones, d_ref, deg_d=deg)
        for cid in off:
            z, d = cp.CASES[cid]["hdepth"], cp.CASES[cid]["hd"]
            off[cid] = float(np.sqrt(np.mean(
                [abs(pr(t, z, d) - measured(cid, t)) ** 2 for t in tones])))
        out[deg] = dict(node_rms=float(np.sqrt(np.mean(np.square(node)))),
                        off=off)
    return dict(n_diameters=len(dia),
                d_values=[float(u) for u in sorted(ds)], by_degree=out)

def profiled_residual(depth, diam, tones, unc):

    predict = _separable_surface(list(cp.DEPTH_SERIES), list(cp.diameter_series()),
                                 tones if isinstance(tones, list)
                                 else sorted({t for v in tones.values()
                                              for t in v}),
                                 cp.CASES["C1"]["hd"])
    zs = np.array([cp.CASES[c]["hdepth"] for c in cp.DEPTH_SERIES])
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

    put("Nruns", f"{c['n_runs']}")
    put("Nbatches", f"{c['n_batches']}")
    put("Alpha", f"{c['alpha']:.2e}".replace("e-0", "\\times 10^{-") + "}")
    put("NoiseFloor", _num(c["noise_mK"], 3))
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
        put(f"P{tag}", _num(o["p"], 2))
        put(f"P{tag}Lo", _num(o["lo"], 2))
        put(f"P{tag}Hi", _num(o["hi"], 2))
        put(f"Err{tag}", _num(100 * o["err_median"], 2))
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
    lz = np.array([np.log(r["rho"][i]) for i in range(3)
                   for r in [dp["rows"][0]]])
    put("DepthSlopeSteep", _num(abs((lz[1] - lz[0]) / (zs[1] - zs[0])), 1))
    put("DepthSlopeShallow", _num(abs((lz[2] - lz[1]) / (zs[2] - zs[1])), 1))
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
    for deg, tag in ((2, "Par"), (3, "Cub")):
        v = pr["by_degree"][deg]
        put(f"Probe{tag}Node", _num(100 * v["node_rms"], 1))
        put(f"Probe{tag}Deep", _num(100 * v["off"]["V1"], 1))
        put(f"Probe{tag}Shal", _num(100 * v["off"]["W1"], 1))
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

    put("SimRelMin", _num(100 * sm["rel_min"], 1))
    put("SimRelMax", _num(100 * sm["rel_max"], 1))
    put("SimPhaseMax", _num(sm["dphase_max"], 1))
    put("SimLatWorst", _num(sm["lat10_at_worst"], 2))
    put("SimMuLworst", _num(sm["mu_over_L_at_worst"], 2))
    put("SimLatBest", _num(sm["rows"][0]["lat10"], 2))

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
    out["DepthZb"] = _num(dp["z"][1], 2)
    out["DepthZc"] = _num(dp["z"][2], 2)
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
