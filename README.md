# ⚡ BESS Optimizer

**Battery Energy Storage System sizing tool** — finds the economically optimal battery size for a site with solar generation and Germany espot market export pricing.

Built with **Flask + Pyomo + Gurobi** (MILP). Runs a full 8760-hour optimization across one year of hourly data.

---

## Features

- **MILP model** (Mixed Integer Linear Programming) via Gurobi
- **Green battery mode** — solar-only charging, no grid-to-battery
- **Hourly espot export prices** — supports negative prices (Germany market)
- **Fixed import tariff** — all-in electricity price
- **Three binary upgrades**:
  - `u[t]` — prevents simultaneous charge and discharge
  - `z[t]` — prevents simultaneous import and export
  - `y` — install-or-not investment decision
- **Free power sizing** — Gurobi finds optimal P (kW) independently
- **Interactive dashboard** — daily dispatch, SOC profile, annual overview charts
- **CSV export** — full hourly results download

---

## Screenshots

| Dashboard | Dispatch Chart |
|-----------|---------------|
| KPIs, energy mix, economics | Daily solar / battery / grid breakdown |

---

## Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/bess-optimizer.git
cd bess-optimizer
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Gurobi

Gurobi requires a licence. Options:

| Licence Type | How to Get |
|---|---|
| **Academic** | Free at [gurobi.com/academia](https://www.gurobi.com/academia/academic-program-and-licenses/) |
| **Commercial** | Paid — [gurobi.com](https://www.gurobi.com) |
| **Trial** | Free 30-day trial, limited to 2000 variables |

After installing, activate your licence:
```bash
grbgetkey YOUR-LICENCE-KEY
```

### 4. Run the app

```bash
python app.py
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000)

### 5. Upload sample data and run

```bash
# Sample CSV is included at sample_data/sample.csv
# Or generate a fresh one:
python generate_sample.py
```

---

## CSV Format

Upload a CSV with exactly these three columns — **8760 rows** (one per hour, full year):

| Column | Unit | Description |
|---|---|---|
| `load_demand` | kW | Site electricity consumption |
| `solar_yield` | kW | Solar PV generation |
| `export_price` | €/kWh | Espot market price (can be negative) |

**Important:** `export_price` must be in **€/kWh**, not €/MWh.  
EPEX Spot publishes in €/MWh — divide by 1000 before uploading.

Example:
```
load_demand,solar_yield,export_price
85.2,0.0,0.0412
82.1,0.0,0.0387
79.4,0.0,-0.0123
...
```

---

## Model Parameters

| Parameter | Default | Description |
|---|---|---|
| Solar installed capacity | 500 kWp | Used for reference only (P is free) |
| Battery cost | €300/kWh | Upfront installed cost |
| Lifetime | 10 years | Battery service life |
| Discount rate | 8% | WACC / cost of capital |
| Import tariff | €0.25/kWh | All-in grid import price |
| Round-trip efficiency | 95% | Charge and discharge efficiency each |
| Min SOC | 10% | Minimum state of charge |
| Max SOC | 95% | Maximum state of charge |
| C-rate | 0.5 | Power-to-energy ratio (0.5 = 2-hour battery) |
| Max battery size | 10,000 kWh | Upper bound on optimizer |

---

## Model Architecture

### Objective (minimize)

```
CAPEX_annual + grid_import_cost − solar_export_revenue
```

Where `CAPEX_annual = battery_cost × CRF × E_kWh`  
and `CRF = r(1+r)^n / ((1+r)^n − 1)` (Capital Recovery Factor)

### Decision Variables

| Variable | Type | Description |
|---|---|---|
| `E` | Continuous | Battery energy capacity (kWh) — main sizing output |
| `P` | Continuous | Battery power capacity (kW) |
| `soc[t]` | Continuous | State of charge each hour (kWh) |
| `ch_sol[t]` | Continuous | Charge from solar surplus (kW) |
| `dis[t]` | Continuous | Discharge power (kW) |
| `imp[t]` | Continuous | Grid import (kW) |
| `exp[t]` | Continuous | Solar export to grid (kW) |
| `u[t]` | Binary | 1=charging mode, 0=discharging mode |
| `z[t]` | Binary | 1=importing, 0=exporting |
| `y` | Binary | 1=battery installed, 0=no battery |

### Key Constraints

- Power balance every hour: `solar + discharge + import = load + charge + export`
- SOC dynamics: `soc[t] = soc[t-1] + ch_sol × η − dis / η`
- SOC bounds: `E × soc_min ≤ soc[t] ≤ E × soc_max`
- C-rate: `P ≤ c_rate × E`
- No simultaneous charge/discharge (binary `u[t]`)
- No simultaneous import/export (binary `z[t]`)
- Install decision (binary `y`)

### Negative Export Price Handling

When `export_price[t] < 0`, exporting costs money (appears as positive cost in objective). Gurobi naturally avoids `exp[t]` → solar surplus goes to load, then battery, then grid only if unavoidable. No hard constraint needed — pure economics.

---

## Project Structure

```
bess-optimizer/
├── app.py                  # Flask backend + Pyomo/Gurobi MILP model
├── templates/
│   └── index.html          # Frontend dashboard (Chart.js)
├── sample_data/
│   └── sample.csv          # 8760-hour sample dataset
├── generate_sample.py      # Script to generate new sample data
├── requirements.txt        # Python dependencies
├── .gitignore
└── README.md
```

---

## Deployment Notes

### Local (development)
```bash
python app.py
# Runs on http://127.0.0.1:5000 with debug=True
```

### Production (Gunicorn)
```bash
pip install gunicorn
gunicorn -w 1 -b 0.0.0.0:8000 app:app
```
> Use `workers=1` — Gurobi licence is per-machine, parallel workers may conflict.

### Environment Variables (optional)
```bash
export FLASK_ENV=production
export GRB_LICENSE_FILE=/path/to/gurobi.lic
```

### Cloud Deployment
Gurobi requires the licence server to be reachable. Options:
- **Render / Railway / Fly.io** — works if you set `GRB_LICENSE_FILE` env var pointing to a WLS (Web Licence Service) licence
- **AWS EC2 / Azure VM** — full Gurobi installation, easiest path
- **Docker** — see Gurobi's official Docker images at [hub.docker.com/u/gurobi](https://hub.docker.com/u/gurobi)

---

## Validation Checks

Before trusting any result, verify:

1. **Power balance**: `solar + discharge + import = load + charge + export` in every hour (tolerance < 0.01 kW)
2. **SOC continuity**: Reconstruct SOC from scratch and compare to output
3. **No simultaneous charge + discharge** in any hour
4. **Annual cycles**: `total_discharge / E_kwh` should be 150–500/year
5. **Self-sufficiency ≤ theoretical maximum**: `(direct_solar + surplus × η_rt) / total_load`

---

## Licence

MIT — free to use, modify, and distribute.

---

## Author

Built for energy engineering and master's research applications.  
Model based on MILP formulation with Pyomo + Gurobi backend.
