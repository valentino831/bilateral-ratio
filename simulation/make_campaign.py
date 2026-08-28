import os
import shutil
import sys
import textwrap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

MAPDL = os.environ.get("MAPDL_EXE", "ANSYS.exe")
WORKROOT = os.environ.get("ANSYS_WORKROOT", os.path.join(ROOT, "work"))
APDLDIR = os.path.join(ROOT, "apdl")
RUNROOT = os.path.join(ROOT, "runs")

QMAX = 35000.0

def case(cid, note, th=5, lxp=80, hd=2.0, hdepth=2.50, hy0=5, hy1=15, spoty=9,
         spotx=0, defect=1, esz=1.0, fref=0, asz=0, twin=None, tones=None,
         lyp=None, spotr=None, npp=32, kmat=None, rmat=None, cmat=None):
    return dict(id=cid, note=note, th=th, lxp=lxp, hd=hd, hdepth=hdepth,
                hy0=hy0, hy1=hy1, spoty=spoty, spotx=spotx, defect=defect,
                esz=esz, fsz=esz, fref=fref, asz=asz, twin=twin, tones=tones,
                lyp=lyp, spotr=spotr, npp=npp,
                kmat=kmat, rmat=rmat, cmat=cmat)

def band(c):

    return c["tones"] or BANDS[c["th"]]

CONV = [
    case("S16", "short 40 mm, mesh 1.60", lxp=40, esz=1.60, twin="R16"),
    case("R16", "short 40 mm, mesh 1.60, sound", lxp=40, esz=1.60, defect=0),
    case("S11", "short 40 mm, mesh 1.13", lxp=40, esz=1.13, twin="R11"),
    case("R11", "short 40 mm, mesh 1.13, sound", lxp=40, esz=1.13, defect=0),
    case("S08", "short 40 mm, mesh 0.80", lxp=40, esz=0.80, twin="R08"),
    case("R08", "short 40 mm, mesh 0.80, sound", lxp=40, esz=0.80, defect=0),
    case("L12", "full 80 mm, mesh 1.25", esz=1.25, twin="M12"),
    case("M12", "full 80 mm, mesh 1.25, sound", esz=1.25, defect=0),
    case("L10", "full 80 mm, mesh 1.00", esz=1.00, twin="M10"),
    case("M10", "full 80 mm, mesh 1.00, sound", esz=1.00, defect=0),
]

ESZ_FISICA = 1.0

SIMBAND = [("T004", 0.0432), ("T026", 0.2596), ("T156", 1.556)]

def _f(**kw):
    return case(esz=ESZ_FISICA, **kw)

FISICA = [
    _f(cid="A0", note="sound twin for B, C, D", defect=0),
    _f(cid="B1", note="shallow inclusion", hdepth=1.25, twin="B0"),
    _f(cid="B0", note="sound twin of B1", hdepth=1.25, defect=0),
    _f(cid="C1", note="reference specimen", twin="A0"),
    _f(cid="D1", note="deep inclusion", hdepth=3.75, twin="D0"),
    _f(cid="D0", note="sound twin of D1", hdepth=3.75, defect=0),
    _f(cid="E1", note="shortened channel", hy1=13, twin="E0"),
    _f(cid="E0", note="sound twin of E1", hy1=13, defect=0),
    _f(cid="F1", note="small inclusion", hd=0.5, twin="F0"),
    _f(cid="F0", note="sound twin of F1", hd=0.5, defect=0),
    _f(cid="I1", note="channel moved, symmetric", hy0=-5, hy1=5, spoty=0,
       twin="I0"),
    _f(cid="I0", note="sound twin of I1", hy0=-5, hy1=5, spoty=0, defect=0),
] + [

    case("SC1", "self-similarity, frequencies of G times four", twin="SC0", tones=SIMBAND),
    case("SC0", "sound twin of SC1", defect=0, tones=SIMBAND),

    case("QC1", "grid check at 1.556 Hz", lxp=40, esz=0.80,
         twin="QC0", tones=SIMBAND[2:]),
    case("QC0", "sound twin of QC1", lxp=40, esz=0.80, defect=0,
         tones=SIMBAND[2:]),
]

G10 = [
    case("G1", "10 mm plate, scaled inclusion", th=10, hd=4.0, hdepth=5.00,
         esz=1.4, twin="G0"),
    case("G0", "sound twin of G1", th=10, hd=4.0, hdepth=5.00,
         esz=1.4, defect=0),
]

CONV2_TONES = [("T050", 0.05), ("T075", 0.075), ("T110", 0.11), ("T160", 0.16),
               ("T240", 0.24), ("T360", 0.36), ("T530", 0.53), ("T800", 0.80)]

CONV2 = []
for _tag, _h in [("16", 1.60), ("11", 1.13), ("08", 0.80)]:
    CONV2.append(case(f"P{_tag}", f"short 40 mm, mesh {_h:.2f}, eight frequencies",
                      lxp=40, esz=_h, twin=f"Q{_tag}", tones=CONV2_TONES))
    CONV2.append(case(f"Q{_tag}", f"short 40 mm, mesh {_h:.2f}, sound",
                      lxp=40, esz=_h, defect=0, tones=CONV2_TONES))

CONV3_TONES = [("T400", 0.40), ("T800", 0.80)]

CONV3 = []
for _tag, _h in [("11", 1.13), ("08", 0.80), ("06", 0.60)]:
    CONV3.append(case(f"X{_tag}", f"short 20 mm, mesh {_h:.2f}, fine end",
                      lxp=20, esz=_h, twin=f"Y{_tag}", tones=CONV3_TONES))
    CONV3.append(case(f"Y{_tag}", f"short 20 mm, mesh {_h:.2f}, sound",
                      lxp=20, esz=_h, defect=0, tones=CONV3_TONES))

BAND9 = [("T050", 0.05), ("T110", 0.11), ("F02", 0.20), ("T240", 0.24),
         ("F04", 0.40), ("T530", 0.53), ("F08", 0.80), ("T120", 1.20),
         ("T180", 1.80)]
BAND_NEW = [t for t in BAND9 if t[0] not in ("F02", "F04", "F08")]

BAND5 = [t for t in BAND9 if t[0] in ("T050", "T110", "F02", "F04", "F08")]

BANDA = [
    _f(cid="B1", note="shallow inclusion, new frequencies", hdepth=1.25,
       twin="B0", tones=BAND_NEW),
    _f(cid="B0", note="sound twin of B1", hdepth=1.25, defect=0,
       tones=BAND_NEW),
    _f(cid="C1", note="reference specimen, new frequencies", twin="A0", tones=BAND_NEW),
    _f(cid="A0", note="sound twin of C1", defect=0, tones=BAND_NEW),
    _f(cid="D1", note="deep inclusion, new frequencies", hdepth=3.75, twin="D0",
       tones=BAND_NEW),
    _f(cid="D0", note="sound twin of D1", hdepth=3.75, defect=0,
       tones=BAND_NEW),
]

DIAMETRI = [
    _f(cid="J1", note="diameter 1.0 mm", hd=1.0, twin="J0", tones=BAND9),
    _f(cid="J0", note="sound twin of J1", hd=1.0, defect=0, tones=BAND9),
    _f(cid="K1", note="diameter 3.0 mm", hd=3.0, twin="K0", tones=BAND9),
    _f(cid="K0", note="sound twin of K1", hd=3.0, defect=0, tones=BAND9),
    _f(cid="F1", note="diameter 0.5 mm, new frequencies", hd=0.5, twin="F0",
       tones=BAND_NEW),
    _f(cid="F0", note="sound twin of F1", hd=0.5, defect=0, tones=BAND_NEW),
]

CIECO = [
    _f(cid="V1", note="held out, deep and thin", hd=1.40, hdepth=3.10,
       twin="V0", tones=BAND9),
    _f(cid="V0", note="sound twin of V1", hd=1.40, hdepth=3.10, defect=0,
       tones=BAND9),
    _f(cid="W1", note="held out, shallow and thin", hd=0.70, hdepth=1.90,
       twin="W0", tones=BAND9),
    _f(cid="W0", note="sound twin of W1", hd=0.70, hdepth=1.90, defect=0,
       tones=BAND9),
]

DIAMETRI2 = [
    _f(cid="N1", note="diameter 0.75 mm", hd=0.75, twin="N0", tones=BAND9),
    _f(cid="N0", note="sound twin of N1", hd=0.75, defect=0, tones=BAND9),
    _f(cid="U1", note="diameter 1.5 mm", hd=1.50, twin="U0", tones=BAND9),
    _f(cid="U0", note="sound twin of U1", hd=1.50, defect=0, tones=BAND9),
    _f(cid="Z1", note="diameter 2.5 mm", hd=2.50, twin="Z0", tones=BAND9),
    _f(cid="Z0", note="sound twin of Z1", hd=2.50, defect=0, tones=BAND9),
]

SPOT_X = [-6, -4, -2, 0, 2, 4, 6]

SPOT_TAG = "ABDEFGH"
SORGENTE = []
for _i, _x in enumerate(SPOT_X):
    _t = SPOT_TAG[_i]
    SORGENTE += [
        _f(cid=f"O{_t}1", note=f"spot at x = {_x} mm", spotx=_x,
           twin=f"O{_t}0", tones=BAND5),
        _f(cid=f"O{_t}0", note=f"sound twin, spot at x = {_x} mm",
           spotx=_x, defect=0, tones=BAND5),
    ]

PASSO = [
    _f(cid="A64", note="reference specimen, 64 substeps", npp=64, twin="B64",
       tones=BAND9),
    _f(cid="B64", note="sound twin, 64 substeps", npp=64, defect=0,
       tones=BAND9),
    _f(cid="A28", note="reference specimen, 128 substeps", npp=128, twin="B28",
       tones=[("T180", 1.80), ("F02", 0.2)]),
    _f(cid="B28", note="sound twin, 128 substeps", npp=128, defect=0,
       tones=[("T180", 1.80), ("F02", 0.2)]),
    _f(cid="C64", note="deep inclusion, 64 substeps", hdepth=3.75, npp=64,
       twin="E64", tones=[("T180", 1.80), ("F02", 0.2)]),
    _f(cid="E64", note="sound twin of C64", hdepth=3.75, npp=64, defect=0,
       tones=[("T180", 1.80), ("F02", 0.2)]),
    _f(cid="C28", note="deep inclusion, 128 substeps", hdepth=3.75, npp=128,
       twin="E28", tones=[("T180", 1.80)]),
    _f(cid="E28", note="sound twin of C28", hdepth=3.75, npp=128, defect=0,
       tones=[("T180", 1.80)]),
]

PROFONDITA = [
    _f(cid="H1", note="depth 1.875 mm", hdepth=1.875, twin="H0",
       tones=BAND9),
    _f(cid="H0", note="sound twin of H1", hdepth=1.875, defect=0,
       tones=BAND9),
    _f(cid="T1", note="depth 3.125 mm", hdepth=3.125, twin="T0",
       tones=BAND9),
    _f(cid="T0", note="sound twin of T1", hdepth=3.125, defect=0,
       tones=BAND9),
]

G10B = [
    case("GA", "10 mm plate, every length scaled", th=10, lxp=160, lyp=60,
         hd=4.0, hdepth=5.00, hy0=10, hy1=30, spoty=18, spotr=8, esz=2.0,
         twin="GB"),
    case("GB", "sound twin of GA", th=10, lxp=160, lyp=60, hd=4.0,
         hdepth=5.00, hy0=10, hy1=30, spoty=18, spotr=8, esz=2.0, defect=0),
]

ALPHA_RATIO = (170.0 / (2700.0 * 900.0)) / (30.0 / (7850.0 * 500.0))

ALBAND = [("A" + _tag[1:], round(_f0 * ALPHA_RATIO, 3)) for _tag, _f0 in BAND9]
ALLUMINIO = [
    _f(cid="AL1", note="aluminium alloy, band scaled", twin="AL0",
       tones=ALBAND, kmat=170.0, rmat=2700.0, cmat=900.0),
    _f(cid="AL0", note="sound twin of AL1", defect=0, tones=ALBAND,
       kmat=170.0, rmat=2700.0, cmat=900.0),
]

BATCHES = {"conv": CONV, "conv2": CONV2, "conv3": CONV3,
           "banda": BANDA, "diametri": DIAMETRI, "diametri2": DIAMETRI2,
           "fisica": FISICA, "g10": G10, "cieco": CIECO,
           "sorgente": SORGENTE, "passo": PASSO, "profondita": PROFONDITA,
           "g10b": G10B, "alluminio": ALLUMINIO}

MESHTESTS = ["MESHT1", "MESHT2", "MESHT3", "MESHT4", "MESHT5"]

BANDS = {
    5:  [("F02", 0.2), ("F04", 0.4), ("F08", 0.8)],
    10: [("F0108", 0.0108), ("F0649", 0.0649), ("F0389", 0.389)],
}

COMMON = dict(LYP=30, HXP=0, SPOTX=0, SPOTR=4, HCONV=10, TAMB=20)

MATERIAL = dict(k=30.0, rho=7850.0, c=500.0)

def win(*parts):

    return "\\".join(p.rstrip("\\") for p in parts)

def engine_tail():

    src = os.path.join(APDLDIR, "RUN_1F.inp")
    with open(src, encoding="utf-8", errors="replace") as fh:
        txt = fh.read()
    return txt[txt.index("TOLZ  = ZBACK/1000"):]

def write_case(c, tag, freq, tail, outdir, workdir):
    runid = f"{c['id']}_{tag}"
    if len(runid) > 8:
        raise ValueError(f"RUNID '{runid}' exceeds the 8 characters of an APDL name")
    head = textwrap.dedent(f"""\
        ! ==================================================================
        ! {runid}   {c['note']}
        ! generated by make_campaign.py, do not edit by hand
        ! ==================================================================
        FINISH
        /CLEAR, NOSTART
        /CONFIG, NRES, 100000
        /CWD,'{workdir}'

        ! --- geometry and mesh, EVERY parameter written out --------------
        TH     = {c['th']}
        LXP    = {c['lxp']}
        LYP    = {c.get('lyp') or COMMON['LYP']}
        HD     = {c['hd']}
        HDEPTH = {c['hdepth']}
        HXP    = {COMMON['HXP']}
        HY0    = {c['hy0']}
        HY1    = {c['hy1']}
        SPOTX  = {c.get('spotx', COMMON['SPOTX'])}
        SPOTY  = {c['spoty']}
        SPOTR  = {c.get('spotr') or COMMON['SPOTR']}
        DEFECT = {c['defect']}
        ESZ    = {c['esz']}
        FSZ    = {c['fsz']}
        FREF   = {c['fref']}
        ASZ    = {c['asz']}
        HCONV  = {COMMON['HCONV']}
        TAMB   = {COMMON['TAMB']}
        KMAT   = {c.get('kmat') or MATERIAL['k']}
        RMAT   = {c.get('rmat') or MATERIAL['rho']}
        CMAT   = {c.get('cmat') or MATERIAL['c']}
        /INPUT,BUILD_PLATE,mac

        ! --- run ----------------------------------------------------------
        RUNID = '{runid}'
        FREQ  = {freq}
        NPP   = {c['npp']}
        QMAX  = {QMAX}
        ZBACK = {c['th']/1000.0}
        """)
    with open(os.path.join(outdir, f"{runid}.inp"), "w",
              encoding="utf-8", newline="\r\n") as fh:
        fh.write(head + "\n" + tail)
    return runid

def build(batch):
    cases = BATCHES[batch]
    outdir = os.path.join(RUNROOT, batch)
    workdir = win(WORKROOT, batch)
    os.makedirs(outdir, exist_ok=True)
    tail = engine_tail()
    runids = []
    for c in cases:
        for tag, freq in band(c):
            runids.append(write_case(c, tag, freq, tail, outdir, workdir))

    shutil.copy(os.path.join(APDLDIR, "BUILD_PLATE.mac"), outdir)

    if batch == "conv":
        c0 = dict(cases[0], id="OUTT", defect=1, twin=None,
                  note="diagnostic, OUTRES without a component")
        tag, freq = band(c0)[0]
        patched = tail.replace("OUTRES, NSOL, NSKIP, FACCE",
                               "OUTRES, NSOL, NSKIP        ! without a component")
        if patched == tail:
            raise RuntimeError("OUTRES line not found in RUN_1F.inp")
        runid = write_case(c0, tag, freq, patched, outdir, workdir)
        os.replace(os.path.join(outdir, f"{runid}.inp"),
                   os.path.join(outdir, "OUTTEST.inp"))
        with open(os.path.join(outdir, "run_outtest.bat"), "w",
                  newline="\r\n") as fh:
            fh.write("@echo off\r\n")
            fh.write(f'if not exist "{workdir}" mkdir "{workdir}"\r\n')
            fh.write(f'copy /Y "%~dp0BUILD_PLATE.mac" "{workdir}" >nul\r\n')
            fh.write(f'cd /d "{workdir}"\r\n')
            fh.write(f'"{MAPDL}" -b -i "%~dp0OUTTEST.inp" '
                     f'-o "OUTTEST.out" -j "OUTT_F02"\r\n')
            fh.write("echo Done. Compare OUTTEST.csv with S16_F02.csv\r\n")
            fh.write("pause\r\n")

    for k in MESHTESTS:
        shutil.copy(os.path.join(APDLDIR, f"{k}.inp"), outdir)
    with open(os.path.join(outdir, "run_meshtest.bat"), "w",
              newline="\r\n") as fh:
        fh.write("@echo off\r\n")
        fh.write('cd /d "%~dp0"\r\n')
        fh.write("if exist MESHTEST.txt del MESHTEST.txt\r\n")
        for k in MESHTESTS:
            fh.write(f"echo === {k} ===\r\n")
            fh.write(f'"{MAPDL}" -b -i {k}.inp -o {k}.out -j {k.lower()}\r\n')
        fh.write("echo.\r\n")
        fh.write("echo var  ESZ    TH  front_NSEL back_NSEL front_COORD back_COORD elementi\r\n")
        fh.write("type MESHTEST.txt\r\n")
        fh.write("echo.\r\n")
        fh.write("if exist MESHT5.txt type MESHT5.txt\r\n")
        fh.write("pause\r\n")

    bat = os.path.join(outdir, "run_all.bat")
    with open(bat, "w", newline="\r\n") as fh:
        fh.write("@echo off\r\n")
        fh.write(f'if not exist "{workdir}" mkdir "{workdir}"\r\n')
        fh.write(f'copy /Y "%~dp0BUILD_PLATE.mac" "{workdir}" >nul\r\n')
        fh.write(f'cd /d "{workdir}"\r\n')
        fh.write(f"echo Batch {batch}, {len(runids)} runs\r\n")

        for r in runids:
            fh.write(f'if exist "{r}.csv" (\r\n')
            fh.write(f"  echo === {r} gia' fatto, salto ===\r\n")
            fh.write(") else (\r\n")
            fh.write(f"  echo === {r} ===\r\n")
            fh.write(f'  "{MAPDL}" -b -i "%~dp0{r}.inp" -o "{r}.out" -j "{r}"\r\n')
            fh.write(f"  if errorlevel 1 echo WARNING: {r} ended with an error\r\n")
            fh.write(")\r\n")
        fh.write(f"echo Done. Now run:  python check_campaign.py {batch}\r\n")
        fh.write("pause\r\n")

    with open(os.path.join(RUNROOT, "run_all_batches.bat"), "w",
              newline="\r\n") as fh:
        fh.write("@echo off\r\n")
        fh.write("echo Runs every batch in turn. This takes days.\r\n")
        for name in BATCHES:
            fh.write(f"echo ############ batch {name} ############\r\n")
            fh.write(f'call "%~dp0{name}\\run_all.bat" < nul\r\n')
        fh.write("echo Done. Now run check_campaign.py on each batch\r\n")
        fh.write("pause\r\n")

    print(f"batch '{batch}': {len(runids)} runs in {outdir}")
    print(f"working directory: {workdir}")
    print(f"launch: {bat}")

def main():
    if len(sys.argv) < 2 or sys.argv[1] not in BATCHES:

        print("available batches:\n")
        for name, cases in BATCHES.items():
            n = sum(len(band(c)) for c in cases)
            print(f"  {name} ({len(cases)} cases, {n} runs)")
            print(f"    {'id':6s} {'TH':>4s} {'LXP':>4s} {'HD':>5s} "
                  f"{'HDEPTH':>7s} {'ESZ':>5s} {'FSZ':>5s} {'DEF':>4s}  twin")
            for c in cases:
                print(f"    {c['id']:6s} {c['th']:4g} {c['lxp']:4g} "
                      f"{c['hd']:5g} {c['hdepth']:7g} {c['esz']:5g} "
                      f"{c['fsz']:5g} {c['defect']:4d}  {c['twin'] or ''}")
            print()
        print("usage: python make_campaign.py <batch>")
        return
    build(sys.argv[1])

if __name__ == "__main__":
    main()
