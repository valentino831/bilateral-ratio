import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, NullFormatter

import campaign as cp
import convergence as cv
import results as R

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.abspath(os.path.join(HERE, "..", "figures"))
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Liberation Serif", "DejaVu Serif"],
    "font.size": 8.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "xtick.labelsize": 7.2, "ytick.labelsize": 7.2,
    "legend.fontsize": 7.0, "axes.labelsize": 8.0,
    "mathtext.fontset": "dejavuserif",
    "savefig.bbox": "tight", "savefig.pad_inches": 0.015,
})

FG = "#1a1a1a"
C_R = "#a02c2c"
C_T = "#123f8c"
C_RHO = "#1a1a1a"
GREY = "#8c8c8c"
WARN = "#f0d8d8"

def despine(ax, which=("top", "right")):
    for s in which:
        ax.spines[s].set_visible(False)

def decade_ticks(ax, lo, hi, axis="x"):

    cand = [0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    t = [v for v in cand if lo <= v <= hi]
    if len(t) < 3:
        cand = [0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 0.8, 1.0, 1.5,
                2.0, 3.0, 5.0]
        t = [v for v in cand if lo <= v <= hi]
    a = ax.xaxis if axis == "x" else ax.yaxis
    (ax.set_xticks if axis == "x" else ax.set_yticks)(t)
    a.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    a.set_minor_formatter(NullFormatter())

def tone_colours(n, dark=FG):
    return [plt.cm.viridis(0.12 + 0.72 * i / max(n - 1, 1)) for i in range(n)]

def figure_one_face(r):

    ref = r["reference"]
    rows = ref["rows"]
    f = np.array([x["f"] for x in rows])
    fig = plt.figure(figsize=(7.0, 1.70))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.12], wspace=0.34)

    ax = fig.add_subplot(gs[0, 0])

    dr = np.array([x["dphi_refl"] for x in rows])
    dt = np.array([x["dphi_tran"] for x in rows])
    ax.axhline(0.0, color=GREY, lw=0.7, zorder=1)
    ax.semilogx(f, dr, "o-", color=C_R, ms=3.6, lw=1.1, mfc="white", mew=1.0,
                label="reflection", zorder=3)
    ax.semilogx(f, dt, "s-", color=C_T, ms=3.4, lw=1.4, label="transmission",
                zorder=4)
    ax.set_xlim(f.min() * 0.75, f.max() * 1.3)
    lo, hi = min(dr.min(), dt.min()), max(dr.max(), dt.max())
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.10 * (hi - lo))
    decade_ticks(ax, *ax.get_xlim())
    ax.set_xlabel("modulation frequency [Hz]", labelpad=1.5)
    ax.set_ylabel(r"phase contrast $\Delta\varphi$ [deg]", labelpad=2.0)
    ax.legend(loc="lower left", frameon=False, handlelength=1.5,
              borderpad=0.15, labelspacing=0.28, handletextpad=0.5)
    ax.set_title("(a) contrast at the reading point", fontsize=8.2, pad=3.5)
    despine(ax)

    ax = fig.add_subplot(gs[0, 1])
    tw = ref["true_width"]
    ax.axhline(tw, color=FG, lw=0.9, ls=(0, (4, 2.5)), zorder=1)
    ax.text(0.34, 0.045, "true width", transform=ax.transAxes, fontsize=6.6,
            color=FG, ha="center", va="bottom")
    for key, col, mk, lw, z in (("width_refl", C_R, "o", 1.1, 3),
                                ("width_tran", C_T, "s", 1.4, 4)):
        w = np.array([x[key] for x in rows], float)
        ok = np.isfinite(w)
        ax.semilogx(f[ok], w[ok], mk + "-", color=col, ms=3.5, lw=lw,
                    mfc="white", mew=1.0, zorder=z)
    ax.set_xlim(f.min() * 0.75, f.max() * 1.3)
    ax.set_ylim(0, ref["width_refl_max"] * 1.12)
    decade_ticks(ax, *ax.get_xlim())
    ax.set_xlabel("modulation frequency [Hz]", labelpad=1.5)
    ax.set_ylabel("apparent width [mm]", labelpad=2.0)
    ax.set_title("(b) apparent width", fontsize=8.2, pad=3.5)
    despine(ax)

    ax = fig.add_subplot(gs[0, 2])
    for label, key, col, mk in ((r"$|D_{\mathrm{r}}|$", "d_refl", C_R, "o"),
                                (r"$|D_{\mathrm{t}}|$", "d_tran", C_T, "s"),
                                (r"$|\rho|$", "rho", C_RHO, "D")):
        o = r["convergence"]["fits"][f"{key}_linear"]
        ex = o["extrap"] if "extrap" in o else cv.extrapolate(
            cv.gather("conv2", key), o["p"])
        X, Y = [], []
        for v in ex.values():
            if v["C"] == 0:
                continue
            X.extend(v["x"])
            Y.extend(np.abs(np.array(v["y"]) - v["psi_inf"]) / abs(v["C"]))
        X, Y = np.array(X), np.array(Y)
        ax.loglog(X, Y, mk, color=col, ms=3.2, mfc="white", mew=0.9,
                  label=f"{label}: $p={o['p']:.2f}$", zorder=3)
        xs = np.linspace(X.min() * 0.8, X.max() * 1.25, 50)
        ax.loglog(xs, xs ** o["p"], "-", color=col, lw=1.0, alpha=0.8, zorder=2)
        npts, nt = o["n"], o["tones"]
    decade_ticks(ax, *ax.get_xlim())
    ax.set_ylim(max(Y.min() * 0.2, 1e-8), 60.0)
    ax.set_xlabel(r"$h/\mu$", labelpad=1.5)
    ax.set_ylabel("scaled distance from extrapolation", labelpad=2.0)
    ax.legend(loc="upper left", frameon=False, handlelength=1.1,
              borderpad=0.15, labelspacing=0.28, handletextpad=0.45)
    ax.set_title("(c) pooled grid convergence", fontsize=8.2, pad=3.5)
    ax.text(0.97, 0.03, f"{npts} points, {nt} frequencies", transform=ax.transAxes,
            fontsize=6.5, ha="right", va="bottom", color=GREY)
    despine(ax)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig1_motivation.{ext}"), dpi=400)
    plt.close(fig)
    print("fig1_motivation")

def figure_ratio(r):

    dp, dm, ref = r["depth"], r["diameter"], r["reference"]
    fig = plt.figure(figsize=(7.0, 1.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.05], wspace=0.36)

    ax = fig.add_subplot(gs[0, 0])
    zs = np.array(dp["z"])
    rows = dp["rows"]
    cols = tone_colours(len(rows))
    lo = np.array([min(x["rho"][i] for x in rows) for i in range(len(zs))])
    hi = np.array([max(x["rho"][i] for x in rows) for i in range(len(zs))])
    ax.fill_between(zs, lo, hi, color="#d8d8d8", lw=0, zorder=1)
    for x, c in zip(rows, cols):
        first_last = x is rows[0] or x is rows[-1]
        ax.semilogy(zs, x["rho"], "-", color=c if first_last else GREY,
                    lw=1.5 if first_last else 0.6,
                    marker="o" if first_last else None, ms=3.4, mfc="white",
                    mew=1.0,
                    label=f"{x['f']:g} Hz" if first_last else None,
                    zorder=4 if first_last else 2)
    ax.set_xlim(zs.min() - 0.25, zs.max() + 0.25)
    ax.set_xlabel("depth of the channel axis, $z_0$ [mm]", labelpad=1.5)
    ax.set_ylabel(r"$|\rho|$", labelpad=2.0)
    ax.legend(loc="upper right", frameon=False, handlelength=1.5,
              borderpad=0.15, labelspacing=0.28, handletextpad=0.5,
              title=f"{len(rows)} frequencies", title_fontsize=6.8)
    ax.set_title("(a) modulus against depth", fontsize=8.2, pad=3.5)
    ax.text(0.04, 0.05,
            r"$-\mathrm{d}\ln|\rho|/\mathrm{d}z_0$" "\n"
            rf"$= {dp['slope_min']:.2f}$ to ${dp['slope_max']:.2f}\ "
            r"\mathrm{mm}^{-1}$",
            transform=ax.transAxes, fontsize=6.9, ha="left", va="bottom",
            linespacing=1.5)
    despine(ax)

    ax = fig.add_subplot(gs[0, 1])
    x1 = [q["mu_over_L"] for q in ref["rows"]]
    y1 = [q["abs_rho"] for q in ref["rows"]]
    o = np.argsort(x1)
    ax.plot(np.array(x1)[o], np.array(y1)[o], "o-", color=C_T, ms=3.8, lw=1.3,
            mfc="white", mew=1.0,
            label=f"full plate, {len(x1)} frequencies", zorder=4)
    fit = r["convergence"]["fits"]["rho_linear"]
    xs2, ys2 = [], []
    for tone, v in fit["extrap"].items():
        h = cp.CASES[cp.CONV_LEVELS["conv2"][-1]]["esz"]
        xs2.append(h / (v["h_over_mu"] * cp.CASES["P08"]["th"]))
        ys2.append(v["psi_inf"])
    o2 = np.argsort(xs2)
    ax.plot(np.array(xs2)[o2], np.array(ys2)[o2], "s--", color=C_R, ms=3.4,
            lw=1.2, mfc="white", mew=1.0,
            label=f"short plate, {len(xs2)} frequencies", zorder=3)
    ax.set_xscale("log")
    ax.set_xlim(min(min(x1), min(xs2)) * 0.8, max(max(x1), max(xs2)) * 1.25)
    decade_ticks(ax, *ax.get_xlim())
    ax.set_xlabel(r"$\mu/L$", labelpad=1.5)
    ax.set_ylabel(r"$|\rho|$", labelpad=2.0)
    ax.legend(loc="upper center", frameon=False, handlelength=1.6,
              borderpad=0.15, labelspacing=0.28, handletextpad=0.5)
    ax.set_title("(b) modulus across the band", fontsize=8.2, pad=3.5)
    despine(ax)

    ax = fig.add_subplot(gs[0, 2])
    ratio = np.array(dm["ratio"])
    mid = np.sqrt(ratio[:-1] * ratio[1:])
    cols = tone_colours(len(dm["rows"]))
    for x, c in zip(dm["rows"], cols):
        ax.loglog(mid, np.abs(x["local"]), "o-", color=c, ms=3.2, lw=0.9,
                  mfc="white", mew=0.9, zorder=3)
    ax.axvspan(1.0, 10.0, color=WARN, lw=0, zorder=0)
    ax.text(1.15, 0.045, "approximation\nnot valid", fontsize=6.6,
            color="#8a3a3a", ha="left", va="bottom", linespacing=1.4)
    ax.set_xlim(mid.min() * 0.7, mid.max() * 1.5)
    ax.set_ylim(0.04, 4.0)
    decade_ticks(ax, *ax.get_xlim())
    decade_ticks(ax, *ax.get_ylim(), axis="y")
    ax.set_xlabel(r"$d\,/\,$clearance from the excited face", labelpad=1.5)
    ax.set_ylabel(r"$\partial\ln|\rho|/\partial\ln d$", labelpad=2.0)
    ax.set_title("(c) dependence on diameter", fontsize=8.2, pad=3.5)
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array([])
    ax.text(0.03, 0.95, f"{len(dm['rows'])} frequencies,\n"
                        f"{len(dm['d'])} diameters", transform=ax.transAxes,
            fontsize=6.6, ha="left", va="top", color=GREY, linespacing=1.4)
    despine(ax)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig2_depth.{ext}"), dpi=400)
    plt.close(fig)
    print("fig2_depth")

def figure_identifiability(r):

    idt = r["ident"]
    fig = plt.figure(figsize=(3.40, 2.15))
    ax = fig.add_subplot(1, 1, 1)
    mul = [cp.mu(x["f"]) / cp.CASES["C1"]["th"] for x in idt["rows"]]
    tau = [x["tau"] for x in idt["rows"]]
    o = np.argsort(mul)
    tauf = [x["tau_phase"] for x in idt["rows"]]
    ax.semilogx(np.array(mul)[o], np.array(tau)[o], "o-", color=C_RHO, ms=3.8,
                lw=1.3, mfc="white", mew=1.0, zorder=3, label="modulus")
    ax.semilogx(np.array(mul)[o], np.array(tauf)[o], "s--", color=C_T, ms=3.6,
                lw=1.3, mfc="white", mew=1.0, zorder=3, label="argument")
    ax.legend(loc="upper right", frameon=False, handlelength=1.7,
              borderpad=0.15, labelspacing=0.28, handletextpad=0.5)
    decade_ticks(ax, min(mul) * 0.8, max(mul) * 1.25)
    ax.set_xlim(min(mul) * 0.8, max(mul) * 1.25)
    lo, hi = min(min(tau), min(tauf)), max(max(tau), max(tauf))
    ax.set_ylim(lo - 0.08 * (hi - lo), hi + 0.34 * (hi - lo))
    ax.set_xlabel(r"$\mu/L$", labelpad=1.5)
    ax.set_ylabel(r"trade-off direction [mm per e-fold]", labelpad=2.0)
    ax.text(0.03, 0.96,
            f"separation {1e3*idt['gap_min']:.0f} to "
            f"{1e3*idt['gap_max']:.0f} $\\mathrm{{\\mu}}$m",
            transform=ax.transAxes, fontsize=6.9, ha="left", va="top")
    despine(ax)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig3_identifiability.{ext}"), dpi=400)
    plt.close(fig)
    print("fig3_identifiability")

def figure_similarity(r):
    sm = r["similarity"]["rows"]

    fig = plt.figure(figsize=(3.40, 3.15))
    gs = fig.add_gridspec(2, 1, hspace=0.55)
    x = np.array([q["mu_over_L"] for q in sm])
    L5 = cp.CASES["SC1"]["th"]
    L10 = cp.CASES["G1"]["th"]

    ax = fig.add_subplot(gs[0, 0])
    ax.plot(x, [q["rho5"] for q in sm], "o-", color=C_T, ms=4.2, lw=1.4,
            mfc="white", mew=1.2, label=f"$L={L5:g}$ mm", zorder=3)
    ax.plot(x, [q["rho10"] for q in sm], "s--", color=C_R, ms=4.0, lw=1.4,
            mfc="white", mew=1.2, label=f"$L={L10:g}$ mm", zorder=3)

    align = ["left"] + ["center"] * (len(sm) - 2) + ["right"]
    for q, al in zip(sm, align):
        ax.annotate(f"{100*q['rel']:.0f}%",
                    xy=(q["mu_over_L"], max(q["rho5"], q["rho10"])),
                    xytext=(0, 6), textcoords="offset points",
                    fontsize=6.8, color=GREY, ha=al)

    vals = [v for q in sm for v in (q["rho5"], q["rho10"])]
    lo, hi = min(vals), max(vals)
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.42 * (hi - lo))
    ax.set_xscale("log")
    decade_ticks(ax, x.min() * 0.8, x.max() * 1.3)
    ax.set_xlim(x.min() * 0.8, x.max() * 1.3)
    ax.set_xlabel(r"$\mu/L$", labelpad=1.5)
    ax.set_ylabel(r"$|\rho|$", labelpad=2.0)
    ax.legend(loc="upper center", frameon=False, handlelength=1.8, ncol=2,
              borderpad=0.15, labelspacing=0.28, handletextpad=0.5,
              columnspacing=1.2)
    ax.set_title("(a) modulus", fontsize=8.2, pad=3.5)
    despine(ax)

    ax = fig.add_subplot(gs[1, 0])
    p5 = np.degrees(np.unwrap(np.radians([q["arg5"] for q in sm])))
    p10 = np.degrees(np.unwrap(np.radians([q["arg10"] for q in sm])))
    p5 = p5 - 360.0 * np.round(np.mean(p5 - p10) / 360.0)
    ax.plot(x, p5, "o-", color=C_T, ms=4.2, lw=1.4, mfc="white", mew=1.2,
            label=f"$L={L5:g}$ mm", zorder=3)
    ax.plot(x, p10, "s--", color=C_R, ms=4.0, lw=1.4, mfc="white", mew=1.2,
            label=f"$L={L10:g}$ mm", zorder=3)
    for q, a, b, al in zip(sm, p5, p10, align):
        ax.annotate(f"{q['dphase']:.1f} deg", xy=(q["mu_over_L"], max(a, b)),
                    xytext=(0, 7), textcoords="offset points",
                    fontsize=6.8, color=GREY, ha=al)
    lo, hi = min(p5.min(), p10.min()), max(p5.max(), p10.max())
    ax.set_ylim(lo - 0.10 * (hi - lo), hi + 0.30 * (hi - lo))
    ax.set_xscale("log")
    decade_ticks(ax, x.min() * 0.8, x.max() * 1.3)
    ax.set_xlim(x.min() * 0.8, x.max() * 1.3)
    ax.set_xlabel(r"$\mu/L$", labelpad=1.5)
    ax.set_ylabel(r"$\arg\rho$ [deg]", labelpad=2.0)
    ax.legend(loc="lower left", frameon=False, handlelength=1.8,
              borderpad=0.15, labelspacing=0.28, handletextpad=0.5)
    ax.set_title("(b) phase", fontsize=8.2, pad=3.5)
    despine(ax)

    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(FIG, f"fig4_similarity.{ext}"), dpi=400)
    plt.close(fig)
    print("fig4_similarity")

if __name__ == "__main__":
    if cp.verify(verbose=False):
        raise SystemExit("verification failed, refusing to plot")
    r = R.compute()
    figure_one_face(r)
    figure_ratio(r)
    figure_identifiability(r)
    figure_similarity(r)
    print("\nfigures written to", FIG)
