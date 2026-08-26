# Bilateral ratio for depth identification in diffusive media

This repository holds the complete numerical chain behind the paper

> V. Razza, *A gain-invariant observable for depth identification in diffusive
> media*

Everything is here, from the generation of the finite element campaign to the
LaTeX macros that carry the numbers into the manuscript. No result quoted in the
paper is transcribed by hand: every number is written by `analysis/results.py`
into `numbers.tex`, and the paper reads that file.

The observable studied is the complex ratio

    rho(omega) = D_r(omega) / D_t(omega)

between the differential responses measured at two conjugate points on opposite
faces of a plate, the plate being heated on one face by an intensity-modulated
source. `D = T_hat - T_hat_0` is the difference between the response of the body
containing an inclusion and the response of the sound body, both driven by the
same excitation. The campaign is a set of finite element runs in which each
defective configuration is paired with a sound twin of identical mesh, so that
the differential is an exact nodal difference.

## Layout

    apdl/         the Ansys Mechanical APDL model and run engine
    simulation/   generation of the campaign, verification, phasor extraction
    analysis/     everything downstream of the phasors
    data/         the extracted phasors, one .npz per batch
    figures/      output of analysis/figures.py

Two files are written at the top level by `analysis/results.py`, `numbers.tex`
and `results.json`.

## Requirements

The analysis needs Python 3.9 or later with `numpy`, `scipy`, `pandas` and
`matplotlib`. Ansys Mechanical APDL is needed only to regenerate the raw data;
the campaign in this repository was solved with the Student edition, whose
element cap is what dictates several choices in the batch design.

## Quick start

The phasors of the whole campaign are in `data/`, so the analysis runs without
Ansys and without the raw CSVs.

    cd analysis
    python campaign.py     # loads data/*.npz and verifies the invariants
    python results.py      # writes ../numbers.tex and ../results.json
    python figures.py      # writes ../figures

`campaign.py` reports how many defective and twin pairs share a mesh exactly,
and it must print that all invariants hold before anything else is run.
`results.py` prints the headline quantities and writes one LaTeX macro per
scalar. `figures.py` writes each figure as both PDF and PNG. A fourth script,

    python convergence.py conv2

reports the grid convergence study on its own, one quantity and one interpolation
at a time, which is the detail behind the single line `results.py` prints.

## The full chain

### 1. Generate a batch

    python simulation/make_campaign.py               # lists the batches
    python simulation/make_campaign.py banda         # writes runs/banda

Each run of the campaign is one pair (configuration, frequency) and gets its own
self-contained `.inp` file with every APDL parameter written out explicitly, so
that no parameter can survive from a previous run. The batch directory also
receives a copy of the `BUILD_PLATE.mac` it was generated with, and a
`run_all.bat` that launches the runs in sequence, each in a fresh MAPDL process.
The `.bat` is incremental: a run whose CSV already exists is skipped, so deleting
one CSV is how a single run is repeated.

Two paths depend on the machine and are read from the environment, so that
nothing in the repository points at a particular installation:

    set MAPDL_EXE=<full path to the MAPDL executable>
    set ANSYS_WORKROOT=<directory where the runs are solved>

Without them the generator falls back to `ANSYS.exe`, to be resolved on the
search path, and to a `work` directory beside this file. A single batch writes
some gigabytes of CSV, so the working root is better placed on a local disk and
outside any synchronised folder.

### 2. Solve

    runs\<batch>\run_all.bat

Each run solves eight periods of the modulated load with 32 steps per period and
exports every second step, that is sixteen samples per period, starting after
the first four periods so that the recorded window is in the periodic steady
state. It writes `<RUNID>.csv`, one row per node and time step with the columns

    time [s], node id, x [m], y [m], z [m], temperature [C]

and `<RUNID>.meta`, which carries the parameters the run was actually executed
with. Nodes are exported only on the two observed faces, which is what keeps the
files to about 100 MB each.

### 3. Check before analysing

    python simulation/check_campaign.py <batch> [csv_directory]

This verifies, run by run, the invariants that have been violated at least once
each during the campaign: the frequency inferred from the time step against the
frequency requested, the parameters in the `.meta` against the parameters the
generator intended, the node count of each observed face against the element
size, the balance between the two faces, the position of the heated spot, and
the exact coincidence of the meshes of a defective run and of its sound twin.
Nothing downstream should be run until this prints all pass.

### 4. Extract the phasors

    python simulation/extract_phasors.py <csv_directory> data/<batch>.npz

The analysis uses two numbers per node instead of sixty-four, so this step turns
a batch of CSVs into a file of a few megabytes. For every node the phasor is
estimated by ordinary least squares on

    T(t) = c0 + c1 t + c2 t^2 + c3 t^3 + c4 cos(2 pi f t) + c5 sin(2 pi f t)

with `T_hat = c4 - i c5`. The cubic absorbs the drift that is not extinguished
within the recorded window and that would otherwise leak into the harmonic
component. The regressors are common to all nodes, so one factorisation solves
the problem for a whole face. The frequency is taken from the exported time step,
`f = 1/(16 dt)`, and never from the file name, so that a mistyped name cannot
falsify the analysis.

### 5. Analyse

As in the quick start above.

## Batches

| batch       | what it is for |
|-------------|----------------|
| `conv`      | grid convergence, three levels on a shortened plate plus two bridge levels on the full plate |
| `conv2`     | the same three levels over eight frequencies, which is what the pooled fit uses |
| `conv3`     | a finer level on a further shortened plate, used as an out-of-sample check of the fitted order |
| `banda`     | the depth series, three depths over the band |
| `diametri`  | the diameter series |
| `diametri2` | three more diameters, used to test whether the surrogate is limited by the sampling of that axis |
| `fisica`    | the reference configuration and the invariance tests, shortened channel and channel moved |
| `g10`       | a plate of double thickness with the dimensionless groups matched |
| `cieco`     | two configurations held out of every series, used as the out-of-sample test of the estimator |
| `sorgente`  | the heated spot displaced along x, not used in the paper |

## Data format

Each `data/<batch>.npz` holds five arrays per run, named `<RUNID>_<field>` where
`RUNID` is `<CASE>_<TONE>`:

| field  | content |
|--------|---------|
| `_f`   | modulation frequency [Hz], inferred from the time step |
| `_ids` | node numbers, sorted |
| `_xyz` | node coordinates [m], in the order of `_ids` |
| `_ph`  | complex phasor of each node [K] |
| `_dc`  | time average of each node over the recorded window [C] |

A case identifier ending in an odd digit is a defective configuration and the
identifier of its sound twin is declared in `CASES` in `analysis/campaign.py`.
The two runs of a pair share the mesh exactly, node by node.

## The model

`apdl/BUILD_PLATE.mac` builds the geometry parametrically instead of importing
it, so the sound plate and the defective plate can be built from the same
volumes. The cylinder of the channel is subtracted with the `KEEP` option and
then glued back, which is why the two meshes coincide and why the sound twin must
never be built by skipping the subtraction. Parameters are given in millimetres
and converted inside the macro, which works in metres:

| parameter | meaning |
|-----------|---------|
| `TH`      | plate thickness |
| `LXP`, `LYP` | plate plan dimensions |
| `HD`      | channel diameter |
| `HDEPTH`  | depth of the channel axis below the excited face |
| `HXP`     | position of the axis along x |
| `HY0`, `HY1` | ends of the channel along y |
| `SPOTX`, `SPOTY`, `SPOTR` | centre and radius of the heated spot |
| `DEFECT`  | 1 for the channel filled with air, 0 for the sound plate |
| `ESZ`     | element size, `FSZ` is forced equal to it |
| `FREF`, `ASZ` | face refinement level and element size on the channel areas |
| `HCONV`, `TAMB` | convection coefficient and ambient temperature |

The default configuration is a steel plate of 80 by 30 by 5 mm, with
`k = 50 W/(m K)`, `rho = 7850 kg/m^3` and `c_p = 490 J/(kg K)`, a channel of
2 mm diameter with its axis 2.5 mm below the excited face, and a spot of 4 mm
radius centred at `(0, 9) mm` on that face. The peak absorbed flux is
35 kW/m^2 for every run.

The component holding the two observed faces is called `FACCE` and the component
holding the heated area is called `superficie_laser`. Both names are stored in
the model database and are left as they are.

## Reproducing the numbers of the paper

`analysis/results.py` writes `numbers.tex`, which the manuscript inputs, and
`results.json`, which holds the same quantities in a form meant for inspection.
Re-running the three analysis scripts after any change to `data/` updates the
text and the figures of the paper together, which is the reason the chain is
built this way.

