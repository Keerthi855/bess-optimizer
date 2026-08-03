# BESS Optimizer

A Flask web application that sizes a Battery Energy Storage System (BESS) for a
solar-equipped site using Mixed-Integer Linear Programming. Given an hourly load
profile, PV generation profile and export price series, it determines the battery
energy capacity (kWh) and power rating (kW) that minimise total annual cost — and
decides whether installing a battery is worthwhile at all.

Built with Pyomo and solved with Gurobi, HiGHS, CBC or GLPK, whichever is available.
Published results were produced with Gurobi; HiGHS provides an open-source path that
requires no licence.

---

## Table of contents

1. [What the tool does](#what-the-tool-does)
2. [Mathematical formulation](#mathematical-formulation)
3. [Installation](#installation)
4. [Running the app](#running-the-app)
5. [Input data format](#input-data-format)
6. [Parameters](#parameters)
7. [Understanding the results](#understanding-the-results)
8. [API reference](#api-reference)
9. [Project structure](#project-structure)
10. [Interpreting diagnostics](#interpreting-diagnostics)
11. [Troubleshooting](#troubleshooting)
12. [Corrections in this version](#corrections-in-this-version)

---

## What the tool does

The optimiser answers four linked questions in a single solve:

| Question | Decision variable |
| --- | --- |
| Should a battery be installed? | `y` (binary) |
| How much energy capacity? | `E` (kWh) |
| What power rating? | `P` (kW) |
| How should it be dispatched, hour by hour? | `ch_sol`, `ch_grd`, `dis`, `imp`, `exp` |

Sizing and dispatch are co-optimised. The battery is not sized first and dispatched
afterwards; the solver evaluates every candidate size against the dispatch schedule it
would enable, and picks the combination with the lowest annualised cost.

The model is built for markets with **negative spot prices**. When the export price
goes below zero, pushing PV onto the grid costs money. The objective handles this
without any special-case logic: export revenue enters with a minus sign, so negative
prices make exporting expensive and the optimiser keeps solar on site — serving load
first, then charging the battery, and only exporting when it genuinely pays.

---

## Mathematical formulation

### Objective

Minimise total annual cost:

```
min   CAPEX_annualised  +  import_cost  -  export_revenue
```

where

```
CAPEX_annualised = ann_e_cost · E  +  ann_p_cost · P
import_cost      = Σ_t  import_price · imp[t]
export_revenue   = Σ_t  export_price[t] · exp[t]
```

Capital cost is annualised with the **Capital Recovery Factor**:

```
CRF = r(1+r)^n / ((1+r)^n − 1)
```

with `r` the discount rate and `n` the project lifetime in years. Energy and power
components are costed separately — power electronics are assumed to add 20% on top of
the energy cost:

```
ann_e_cost = battery_cost · CRF          €/kWh/year
ann_p_cost = battery_cost · 0.20 · CRF   €/kW/year
```

### Decision variables

| Symbol | Domain | Unit | Meaning |
| --- | --- | --- | --- |
| `E` | continuous ≥ 0 | kWh | Battery energy capacity |
| `P` | continuous ≥ 0 | kW | Battery power rating |
| `soc[t]` | continuous ≥ 0 | kWh | State of charge at start of hour t |
| `ch_sol[t]` | continuous ≥ 0 | kW | Charging from PV surplus |
| `ch_grd[t]` | continuous ≥ 0 | kW | Charging from the grid |
| `dis[t]` | continuous ≥ 0 | kW | Discharging |
| `imp[t]` | continuous ≥ 0 | kW | Grid import |
| `exp[t]` | continuous ≥ 0 | kW | Export to grid |
| `curt[t]` | continuous ≥ 0 | kW | PV curtailment (fixed to 0 unless enabled) |
| `u[t]` | binary | — | 1 = charging mode, 0 = discharging mode |
| `z[t]` | binary | — | 1 = importing, 0 = exporting |
| `y` | binary | — | 1 = battery installed, 0 = not installed |

### Constraints

**C1 — Power balance.** Every hour, supply equals demand:

```
solar[t] + dis[t] + imp[t]  =  load[t] + ch_sol[t] + ch_grd[t] + exp[t] + curt[t]
```

**C2 — Export limit.** Export cannot exceed the PV surplus over load:

```
exp[t] ≤ max(0, solar[t] − load[t])
```

**C3 — Solar charging limit.** The battery charges from PV only when there is surplus:

```
ch_sol[t] ≤ max(0, solar[t] − load[t])
```

**C4 — Surplus split.** Export, solar charging and curtailment share one pool of surplus
and cannot double-count it:

```
exp[t] + ch_sol[t] + curt[t] ≤ max(0, solar[t] − load[t])
```

**C5 — SOC dynamics.** Charge and discharge efficiencies applied on the correct sides:

```
soc[0] = 0.5 · E
soc[t] = soc[t−1] + (ch_sol[t−1] + ch_grd[t−1]) · η_ch − dis[t−1] / η_dis
```

**C6, C7 — SOC window.** Both bounds anchored to the chosen capacity, not the input cap:

```
soc_min_pct · E  ≤  soc[t]  ≤  soc_max_pct · E
```

**C7b — Cyclic condition.** The battery may not end the horizon emptier than it started,
which would otherwise book free energy into the objective:

```
soc[N] ≥ soc[0]
```

**C8, C9 — Power limits.**

```
ch_sol[t] + ch_grd[t] ≤ P
dis[t] ≤ P
```

**C10 — C-rate link.** Power and energy are tied by the cell chemistry:

```
P ≤ c_rate · E
```

`P` is additionally bounded above by `min(e_max_kwh · c_rate, solar_kw)` — the battery
shares an inverter and grid connection with the PV array, so it can never discharge
faster than the PV inverter rating.

### The three binary conditions

**1. No simultaneous charge and discharge.** Without this, an LP can charge and
discharge in the same hour to exploit round-trip efficiency asymmetries, which is
physically meaningless.

```
ch_sol[t] + ch_grd[t] ≤ M_p · u[t]
dis[t]                ≤ M_p · (1 − u[t])
```

with `M_p = min(e_max_kwh · c_rate, solar_kw)` — the tightest valid bound on `P`.

**2. No simultaneous import and export.** A single metering point cannot flow both ways:

```
imp[t] ≤ M_g · z[t]
exp[t] ≤ M_g · (1 − z[t])
```

with `M_g = max(load) + max(solar)`.

**3. Install-or-not with a minimum viable size.** The solver may decline to install, but
if it does install, the battery must be at least 100 kWh — otherwise the optimum drifts
toward a trivially small, commercially meaningless battery:

```
E ≤ e_max_kwh · y
E ≥ 100 · y
```

Big-M values throughout are derived from the data rather than set to arbitrary large
constants, which keeps the LP relaxation tight and the branch-and-bound tree small.

---

## Installation

### Requirements

- Python 3.9 or newer
- A MILP solver. HiGHS installs automatically via pip and needs no licence. Gurobi is
  used in preference if present — see [Solver selection](#solver-selection).

### Steps

```bash
git clone https://github.com/Keerthi855/bess-optimizer.git
cd bess-optimizer

# Windows
python -m venv venv
venv\Scripts\activate

# macOS / Linux
python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt
```

If PowerShell blocks activation with "running scripts is disabled on this system":

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
```

Then activate again.

### Solver selection

At solve time the app tries solvers in this order and uses the first one available:

| Order | Solver | How to get it |
| --- | --- | --- |
| 1 | Gurobi | Commercial; free academic licence. Used automatically if installed and licensed. |
| 2 | HiGHS (`appsi_highs`) | `pip install highspy` — included in requirements.txt |
| 3 | HiGHS (`highs`) | Same package, older Pyomo interface |
| 4 | CBC | Separate binary; awkward on Windows |
| 5 | GLPK | Separate binary |

**Results published from this repository were produced with Gurobi**, under an
academic licence. Gurobi sits first in the chain, so on a machine where it is
installed it is selected automatically and silently.

No licence is required to run the model. HiGHS arrives with
`pip install -r requirements.txt`, is open-source, and is entirely sufficient for a
problem of this size — the two solvers converge to solutions agreeing within the
specified MIP gap. If Gurobi is absent, the app falls through to HiGHS with no
change to inputs, configuration or interface.

Because selection is automatic, the solver actually used is reported back in the
JSON response under the `solver` key. Check it before quoting any figure. To confirm
what is available on your machine:

```bash
python -c "import pyomo.environ as pyo; print('gurobi:', pyo.SolverFactory('gurobi').available(exception_flag=False)); print('highs:', pyo.SolverFactory('appsi_highs').available(exception_flag=False))"
```

---

## Running the app

```bash
python app_milp_corrected.py
```

Then open <http://127.0.0.1:5000> in a browser.

Upload your CSV, adjust the parameters in the form, and submit. Results appear as KPI
cards, charts and an hourly dispatch table. Two exports are available: the full hourly
schedule as CSV, and a formatted Word report with charts and validation checks.

Stop the server with `Ctrl+C`.

The development server binds to localhost only. For anything beyond local use, run it
behind a production WSGI server such as Waitress or Gunicorn and set `debug=False`.

---

## Input data format

A CSV with one row per hour. Three columns are required; any others are ignored.

| Column | Unit | Description |
| --- | --- | --- |
| `load_demand` | kW | Site electrical demand |
| `solar_yield` | kW | PV generation |
| `export_price` | €/kWh | Spot price for exported energy; may be negative |

Example:

```csv
load_demand,solar_yield,export_price
120.5,0.0,0.045
118.2,0.0,0.041
125.0,15.3,0.038
140.7,180.4,-0.012
```

Notes on parsing:

- UTF-8 BOMs are stripped automatically, so files exported from Excel work as-is
- Whitespace around column headers is trimmed
- Rows with missing values in any required column are dropped, and the count is
  reported back as `rows_dropped`
- A full year is 8760 rows, but any horizon length works. Since capital cost is
  annualised, a shorter horizon under-represents CAPEX relative to operating cost —
  use a full year for investment decisions.

---

## Parameters

Every field falls back to its default if left blank.

| Parameter | Default | Unit | Description |
| --- | --- | --- | --- |
| `battery_cost` | 300 | €/kWh | Upfront energy-capacity cost. Power electronics add 20%. |
| `lifetime_yrs` | 10 | years | Project lifetime, feeds the CRF |
| `discount_rate` | 0.08 | fraction | Discount rate. 0.08 = 8%. |
| `import_price` | 0.25 | €/kWh | Flat import tariff |
| `solar_kw` | 500 | kWp | Installed PV. Also caps battery power via the shared inverter. |
| `eta_charge` | 0.95 | fraction | Charging efficiency |
| `eta_discharge` | 0.95 | fraction | Discharging efficiency |
| `soc_min_pct` | 0.10 | fraction | Minimum state of charge, as a share of `E` |
| `soc_max_pct` | 0.95 | fraction | Maximum state of charge, as a share of `E` |
| `c_rate` | 0.5 | 1/h | Power-to-energy ratio. 0.5 means a 1000 kWh battery is capped at 500 kW. |
| `e_max_kwh` | 10000 | kWh | Upper bound on capacity the solver may choose |
| `time_limit` | 300 | seconds | Solver wall-clock limit |
| `mip_gap` | 0.001 | fraction | Relative MIP gap. 0.001 = 0.1%. |
| `allow_curtailment` | 0 | 0/1 | Set to 1 to permit PV curtailment |

Validation rules: `0 < soc_min_pct < soc_max_pct ≤ 1`, and `e_max_kwh`, `solar_kw`,
`c_rate`, `lifetime_yrs` must all be positive. Violations return HTTP 400 with an
explanatory message.

### On curtailment

With curtailment off (default), the power balance is a strict equality with no slack —
surplus PV is *forced* onto the grid even when the price is negative. This is realistic
for sites without curtailment capability, but it can make the battery look better than
it is, since it gets credited for absorbing energy the site would otherwise be penalised
for exporting.

With curtailment on, the no-battery baseline is also allowed to curtail during negative
price hours, keeping the comparison like-for-like.

### On the MIP gap

The default of 0.1% is deliberately tight. On an objective of order 10⁵ €/year, a 1% gap
admits roughly 10³ €/year of tolerance — the same order as the annual savings being
measured. Loosening the gap to speed up the solve can therefore swamp the quantity of
interest.

---

## Understanding the results

### Sizing

| Field | Meaning |
| --- | --- |
| `E_kwh` | Optimal energy capacity |
| `P_kw` | Optimal power rating |
| `y_installed` | 1 if the model recommends installing, 0 if not |

`y_installed = 0` is a legitimate and informative result. It means that under the given
prices, no battery pays for itself.

### Economics

| Field | Meaning |
| --- | --- |
| `crf` | Capital Recovery Factor applied |
| `ann_capex` | Annualised capital cost, €/year |
| `ann_import_cost` | Annual grid import cost, €/year |
| `ann_export_rev` | Annual export revenue, €/year. Can be negative. |
| `ann_net_cost` | CAPEX + import − export, €/year |
| `base_cost` | Same site with no battery, €/year |
| `annual_savings` | `base_cost − ann_net_cost` |
| `upfront_capex_energy` | Upfront cost of energy capacity only |
| `upfront_capex_total` | Including the 20% power-electronics component |
| `payback_energy_only` | Years, against energy CAPEX only |
| `payback_total` | Years, against full CAPEX. **Use this one.** |
| `cost_per_kwh_stored` | Annualised CAPEX ÷ energy discharged, €/kWh |
| `cycles_per_year` | Energy discharged ÷ capacity |

Both payback figures are reported because the two definitions circulate
interchangeably in practice and differ by roughly 20% here. `payback_total` is the
honest one. Both are `null` when savings are zero or negative.

### Energy flows and mix

`total_load`, `total_solar`, `total_imp`, `total_exp`, `total_dis`, `total_ch_sol`,
`total_ch_grd`, `total_curtailed` — all in kWh over the horizon.

`solar_pct`, `battery_pct` and `grid_pct` decompose how the site load was covered.
They are disjoint and sum to 100% by construction, derived directly from the power
balance: PV that was stored or exported is not also counted as serving load.

`self_suff` is self-sufficiency, `(1 − total_imp / total_load) × 100`.

### Negative price statistics

`neg_price_hours` counts hours where the export price was below zero.
`neg_price_curtailed` gives the kWh still exported during those hours — ideally near
zero, since exporting then costs money.

---

## API reference

### `GET /`

Serves the web interface.

### `POST /optimize`

Multipart form. Requires `csv_file` plus any parameters from the table above. Returns
the full result JSON: sizing, economics, energy flows, mix percentages, diagnostics,
the complete hourly schedule under `hourly`, and the input parameters echoed back so
downstream consumers can reproduce the calculation.

Errors return HTTP 400 (bad input) or 500 (solver failure) with an `error` key, and a
`trace` key on unhandled exceptions.

### `POST /download`

Accepts the result JSON, returns `bess_results.csv` — the hourly dispatch schedule.

### `POST /report`

Accepts the result JSON, returns `bess_report.docx` — a formatted Word report with a
KPI table, dashboard, individual charts, validation checks and an investment verdict.

The hourly schedule contains, per hour: `load`, `solar`, `export_price`, `ch_solar`,
`ch_grid`, `discharge`, `soc`, `grid_import`, `solar_export`, `curtailed`.

---

## Project structure

```
bess-optimizer/
├── app_milp_corrected.py    Flask routes and the Pyomo MILP model
├── report_builder.py        Word report generation with matplotlib charts
├── templates/
│   └── index.html           Single-page UI
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Interpreting diagnostics

Three boolean flags tell you whether the reported size is a true economic optimum or
an artefact of the bounds you set. Any of them being `true` means the answer is being
shaped by a constraint rather than by the economics.

| Flag | Meaning if true | What to do |
| --- | --- | --- |
| `e_at_bound` | `E` hit `e_max_kwh` | Raise `e_max_kwh` and re-solve |
| `p_at_solar_cap` | `P` hit the PV inverter limit | The inverter is binding, not the battery economics |
| `p_at_crate` | `P = c_rate · E` | Chemistry is binding; a higher C-rate cell would change the answer |

Also check `status`. `optimal` means the gap was closed. `maxTimeLimit` means the
solver ran out of time and returned the best incumbent — the answer is feasible but
not proven optimal, so raise `time_limit` before quoting the number.

---

## Troubleshooting

**`ModuleNotFoundError`** — a dependency is missing. `pip install -r requirements.txt`
inside the activated virtual environment.

**`No MILP solver available`** — `pip install highspy`.

**`TemplateNotFound: index.html`** — `index.html` must sit inside a `templates`
subfolder, not in the project root. Flask will not find it otherwise.

**`Missing columns: [...]`** — check spelling and case of the three required headers.
The error message lists the columns actually found, which usually makes the problem
obvious.

**`Require 0 < SOC_min < SOC_max <= 1`** — SOC limits are fractions, not percentages.
Enter 0.10, not 10.

**Solve is slow** — reduce `time_limit`, or loosen `mip_gap` to 0.005 while
experimenting. Tighten it again for final numbers. A full 8760-hour year with binaries
on every hour is a large model; HiGHS typically handles it in seconds to minutes.

**Files appear online-only under OneDrive** — right-click the project folder and choose
"Always keep on this device" before running.

---

## Corrections in this version

This file is the corrected build of the original `app_milp.py`. Every change is marked
inline with `[FIX n]` — search the source for `[FIX` to review them in place. Route
names, variable names and JSON keys are unchanged, so `index.html` and
`report_builder.py` work against it without modification.

| Fix | Correction |
| --- | --- |
| 1 | `soc[0]` anchored to `E`, not `e_max_kwh`. The old form silently forced `E ≥ 0.526 · e_max_kwh` at every solve, pinning the optimum to an arbitrary input and making the "do not install" branch infeasible. |
| 2 | SOC lower bound anchored to `E` rather than a fixed floor derived from `e_max_kwh`. |
| 3 | Cyclic condition `soc[N] ≥ soc[0]` added, preventing free energy from being booked into the objective. |
| 4 | Default MIP gap tightened from 1% to 0.1%. |
| 5 | Optional PV curtailment, with the no-battery baseline allowed to curtail too so the comparison stays like-for-like. |
| 6 | Solver auto-selection with fallback, replacing hard-coded Gurobi. The solver used is now reported. |
| 7 | UTF-8 BOM stripped and header whitespace trimmed on CSV import. |
| 8 | Blank form fields fall back to defaults instead of raising `ValueError` on `float('')`. |
| 9 | Load-coverage shares made disjoint. The originals overlapped and summed to well over 100%. |
| 10 | Both payback definitions computed and labelled explicitly. |
