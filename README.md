# BESS Optimizer

A Flask web app that sizes a battery energy storage system (BESS) using mixed-integer
linear programming. Given an hourly load profile, solar yield and export prices, it
finds the battery energy and power capacity that minimise total annual cost.

## Model

Objective (minimised):

```
annualised CAPEX + grid import cost - solar export revenue
```

CAPEX is annualised with the capital recovery factor `CRF = r(1+r)^n / ((1+r)^n - 1)`.

Binary variables enforce three conditions:

- `u[t]` — the battery cannot charge and discharge in the same hour
- `z[t]` — the site cannot import and export in the same hour
- `y` — install-or-not, so a trivially tiny battery is never chosen

Negative export prices make exporting cost money, so the optimiser keeps solar
on site rather than pushing it to the grid.

## Setup

```bash
git clone https://github.com/Keerthi855/bess-optimizer.git
cd bess-optimizer

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
python app_milp_corrected.py
```

Open <http://127.0.0.1:5000>.

The app looks for a MILP solver in this order: Gurobi, HiGHS, CBC, GLPK. HiGHS
arrives via the `highspy` package in `requirements.txt`, so no separate install is
needed. Gurobi will be used automatically if a licensed copy is present.

## Input CSV

One row per hour. Three columns are required:

| Column | Unit | Meaning |
| --- | --- | --- |
| `load_demand` | kW | Site electrical demand |
| `solar_yield` | kW | PV generation |
| `export_price` | €/kWh | Spot price for export; may be negative |

Rows with missing values in these columns are dropped. Extra columns are ignored.

## Parameters

Set in the web form; each falls back to the default if left blank.

| Parameter | Default | Meaning |
| --- | --- | --- |
| `battery_cost` | 300 | €/kWh upfront |
| `lifetime_yrs` | 10 | Project lifetime |
| `discount_rate` | 0.08 | Discount rate as a fraction |
| `import_price` | 0.25 | €/kWh flat import tariff |
| `solar_kw` | 500 | Installed PV, also caps battery power |
| `eta_charge` / `eta_discharge` | 0.95 | Round-trip efficiency components |
| `soc_min_pct` / `soc_max_pct` | 0.10 / 0.95 | State-of-charge window |
| `c_rate` | 0.5 | Power-to-energy ratio |
| `e_max_kwh` | 10000 | Upper bound on capacity |
| `time_limit` | 300 | Solver time limit, seconds |
| `mip_gap` | 0.001 | Relative MIP gap |
| `allow_curtailment` | 0 | Set to 1 to let PV be curtailed |

## Project layout

```
BESS-Optimizer-version/
├── app_milp_corrected.py    # Flask app and Pyomo MILP model
├── report_builder.py        # Builds the .docx report
├── templates/
│   └── index.html           # UI
├── requirements.txt
├── .gitignore
└── README.md
```
