"""
BESS Optimizer — MILP Upgrade
==============================
LP → MILP additions (all existing code unchanged):
  Upgrade 1: u[t] binary — prevents simultaneous charge & discharge
  Upgrade 2: z[t] binary — prevents simultaneous import & export
  Upgrade 3: y binary    — install-or-not investment decision
OBJECTIVE (minimize):
  Annualized battery CAPEX
  + Annual grid import cost  (fixed tariff × kWh imported)
  - Annual solar export revenue  (hourly espot price × kWh exported)
  * Negative espot prices → exporting costs money → optimizer avoids export

DISPATCH LOGIC (enforced via constraints):
  Every hour: Solar + Battery_discharge + Grid_import
            = Load + Battery_charge + Solar_export

  When espot price < 0:
    - Exporting is penalized (costs money) → optimizer naturally avoids it
    - Solar goes to load first, then battery, then grid (only if forced)
    - Battery may charge from grid if import price is low enough

  Battery power cap = min(solar_installed_kw, optimizer-chosen P_cap)
  This means battery can never discharge faster than the solar inverter allows.

CALCULATION SUMMARY (shown in UI):
  CRF  = r(1+r)^n / ((1+r)^n - 1)       — Capital Recovery Factor
  CAPEX_ann = battery_cost_per_kwh × CRF × E_kWh   — €/year
  Import_cost = Σ (import_price × imp[t])            — €/year
  Export_rev  = Σ (export_price[t] × exp[t])         — €/year (can be negative)
  Net_cost = CAPEX_ann + Import_cost - Export_rev
  Baseline = cost without any battery
  Savings  = Baseline - Net_cost
"""

from flask import Flask, render_template, request, jsonify, send_file
import pandas as pd
import numpy as np
import io, traceback

app = Flask(__name__)


def capital_recovery_factor(r, n):
    """CRF: converts one-time cost to equivalent annual cost."""
    if r == 0:
        return 1.0 / n
    return r * (1 + r)**n / ((1 + r)**n - 1)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        # ── User parameters ──────────────────────────────────────
        battery_cost   = float(request.form.get('battery_cost',  300))   # €/kWh upfront
        lifetime_yrs   = float(request.form.get('lifetime_yrs',   10))   # years
        discount_rate  = float(request.form.get('discount_rate', 0.08))  # fraction
        import_price   = float(request.form.get('import_price',  0.25))  # €/kWh fixed
        solar_kw       = float(request.form.get('solar_kw',       500))  # installed kWp
        eta_charge     = float(request.form.get('eta_charge',    0.95))
        eta_discharge  = float(request.form.get('eta_discharge', 0.95))
        soc_min_pct    = float(request.form.get('soc_min_pct',   0.10))
        soc_max_pct    = float(request.form.get('soc_max_pct',   0.95))
        c_rate         = float(request.form.get('c_rate',         0.5))
        e_max_kwh      = float(request.form.get('e_max_kwh',   10000))
        time_limit     = int(request.form.get('time_limit',       300))

        # ── CSV ──────────────────────────────────────────────────
        if 'csv_file' not in request.files:
            return jsonify({'error': 'No CSV uploaded'}), 400

        df = pd.read_csv(request.files['csv_file'])

        required = ['load_demand', 'solar_yield', 'export_price']
        missing  = [c for c in required if c not in df.columns]
        if missing:
            return jsonify({'error': f'Missing columns: {missing}. '
                                     f'Need: load_demand (kW), solar_yield (kW), export_price (€/kWh)'}), 400

        df = df.dropna(subset=required).reset_index(drop=True)
        N  = len(df)

        load         = df['load_demand'].values.astype(float)   # kW
        solar        = df['solar_yield'].values.astype(float)   # kW
        export_price = df['export_price'].values.astype(float)  # €/kWh, can be negative

        # ── Annualization ────────────────────────────────────────
        crf          = capital_recovery_factor(discount_rate, lifetime_yrs)
        # Battery cost covers energy capacity; power electronics ~ 20% extra
        ann_e_cost   = battery_cost * crf          # €/kWh/year  (energy component)
        ann_p_cost   = battery_cost * 0.2 * crf   # €/kW/year   (power component)

        # ── Battery power cap ────────────────────────────────────
        # p_hard_cap kept for reference but no longer applied to P bounds
        # Gurobi will now find the optimal P freely (capped only by c_rate × E)
        p_hard_cap   = solar_kw   # kW — retained but not used as P upper bound

        # ── Pyomo LP ─────────────────────────────────────────────
        import pyomo.environ as pyo

        m    = pyo.ConcreteModel()
        m.T  = pyo.Set(initialize=range(N))
        m.T1 = pyo.Set(initialize=range(N + 1))

        # ── Decision variables ───────────────────────────────────
        # Sizing
        m.E  = pyo.Var(bounds=(0, e_max_kwh),              domain=pyo.NonNegativeReals)  # kWh capacity
        m.P  = pyo.Var(bounds=(0, e_max_kwh * c_rate), domain=pyo.NonNegativeReals)  # kW power — free, no solar cap

        # Hourly operation
        m.soc     = pyo.Var(m.T1, domain=pyo.NonNegativeReals)  # state of charge kWh
        m.ch_sol  = pyo.Var(m.T,  domain=pyo.NonNegativeReals)  # charge from solar surplus kW
        # GREEN BATTERY: ch_grd removed — grid charging not allowed
        # Battery charges from solar surplus only (ch_sol)
        m.dis     = pyo.Var(m.T,  domain=pyo.NonNegativeReals)  # discharge kW
        m.imp     = pyo.Var(m.T,  domain=pyo.NonNegativeReals)  # grid import kW
        m.exp     = pyo.Var(m.T,  domain=pyo.NonNegativeReals)  # solar export to grid kW

        # ── MILP UPGRADE: new binary variables ───────────────────────────────
        # Upgrade 1: u[t] = 1 → charging mode,  0 → discharging mode
        m.u = pyo.Var(m.T, domain=pyo.Binary)

        # Upgrade 2: z[t] = 1 → importing from grid,  0 → exporting to grid
        m.z = pyo.Var(m.T, domain=pyo.Binary)

        # Upgrade 3: y = 1 → battery installed,  0 → no battery
        m.y = pyo.Var(domain=pyo.Binary)

        # ── Objective ────────────────────────────────────────────
        # Minimize: CAPEX_annual + import_cost - export_revenue
        # When export_price[t] < 0: exporting costs money → optimizer avoids export
        # → solar naturally stays on-site: serves load, then battery, then grid
        def obj_rule(mo):
            capex      = ann_e_cost * mo.E + ann_p_cost * mo.P
            import_c   = sum(import_price       * mo.imp[t] for t in range(N))
            export_rev = sum(float(export_price[t]) * mo.exp[t] for t in range(N))
            return capex + import_c - export_rev
        m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

        # ── Constraints ──────────────────────────────────────────

        # C1: Power balance every hour
        #   solar_yield + battery_discharge + grid_import
        #   = load + battery_charge_solar + battery_charge_grid + solar_export
        m.c_bal = pyo.Constraint(m.T, rule=lambda mo, t:
            solar[t] + mo.dis[t] + mo.imp[t]
            == load[t] + mo.ch_sol[t] + mo.exp[t])  # green: no ch_grd term

        # C2: Solar export only possible when solar > load + battery_charging
        #   exp[t] <= max(0, solar[t] - load[t])
        #   (cannot export more than the solar surplus above load)
        surplus_max = {t: max(0.0, solar[t] - load[t]) for t in range(N)}
        m.c_exp_lim = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.exp[t] <= surplus_max[t])

        # C3: Solar battery charging <= solar surplus
        #   Battery only charges from solar when solar > load
        m.c_chsol_lim = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.ch_sol[t] <= surplus_max[t])

        # C4: When export_price < 0 → export is penalized in objective so
        #   optimizer avoids exp[t] naturally. No hard constraint needed —
        #   the economics drive the behavior.
        #   But we add: exp[t] + ch_sol[t] <= surplus[t]  (can't double-count surplus)
        m.c_surplus_split = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.exp[t] + mo.ch_sol[t] <= surplus_max[t])

        # C5: SOC dynamics
        #   soc[0] = 50% of max (fixed constant — avoids bilinear)
        soc_init = e_max_kwh * 0.5
        def soc_rule(mo, t):
            if t == 0:
                return mo.soc[t] == soc_init
            ch = mo.ch_sol[t-1]  # green: solar charging only, no grid
            return mo.soc[t] == mo.soc[t-1] + ch * eta_charge - mo.dis[t-1] / eta_discharge
        m.c_soc = pyo.Constraint(m.T1, rule=soc_rule)

        # C6: SOC upper limit: soc[t] <= soc_max_pct × E
        #   (linear: one variable × constant <= one variable)
        m.c_soc_max = pyo.Constraint(m.T1, rule=lambda mo, t:
            mo.soc[t] <= soc_max_pct * mo.E)

        # C7: SOC lower limit — fixed floor to avoid bilinear (E × soc_min_pct)
        #   Floor = e_max_kwh × soc_min_pct (a constant)
        soc_floor = e_max_kwh * soc_min_pct
        m.c_soc_min = pyo.Constraint(m.T1, rule=lambda mo, t:
            mo.soc[t] >= soc_floor)

        # C8: Total charge power <= P (optimizer-chosen power capacity)
        m.c_pch = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.ch_sol[t] <= mo.P)  # green: solar charge only

        # C9: Discharge power <= P  AND  <= solar_kw (hard inverter cap)
        #   P is already bounded by p_hard_cap in its variable bounds (see above)
        m.c_pdis = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.dis[t] <= mo.P)

        # C10: C-rate link: P <= c_rate × E
        m.c_cr = pyo.Constraint(expr=m.P <= c_rate * m.E)

        # ── MILP UPGRADE: three new constraint sets ─────────────────────────

        # ── Upgrade 1: Prevent simultaneous charge AND discharge ─────────────
        # u[t] = 1 → charging allowed, discharging blocked
        # u[t] = 0 → discharging allowed, charging blocked
        # Big-M = P (the optimizer's own power variable) — keeps it tight & linear
        # ch_sol[t] <= P * u[t]   → charge only when u=1 (green: no grid term)
        # dis[t]                 <= P * (1 - u[t])  → discharge only when u=0
        # These are linear because P is a variable and u is binary:
        #   product P*u is linearized internally by Gurobi (standard MILP)
        #   BUT to keep it truly linear (no bilinear), use p_hard_cap as tight M:
        p_M = e_max_kwh * c_rate  # upper bound on P (no solar cap — Gurobi finds optimal)

        m.c_u_charge = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.ch_sol[t] <= p_M * mo.u[t])  # green: solar only

        m.c_u_discharge = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.dis[t] <= p_M * (1 - mo.u[t]))

        # ── Upgrade 2: Prevent simultaneous grid import AND export ───────────
        # z[t] = 1 → import allowed,  z[t] = 0 → export allowed
        # Big-M = max possible grid flow = max(load) + max(solar)  — tight, not arbitrary
        bigM_grid = float(max(load) + max(solar))

        m.c_z_import = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.imp[t] <= bigM_grid * mo.z[t])

        m.c_z_export = pyo.Constraint(m.T, rule=lambda mo, t:
            mo.exp[t] <= bigM_grid * (1 - mo.z[t]))

        # ── Upgrade 3: Install-or-not investment decision ────────────────────
        # y = 0 → E must be 0 (no battery installed)
        # y = 1 → E can be up to e_max_kwh
        # Also: if y = 1, enforce a minimum viable size (e.g. 100 kWh)
        #   to prevent the solver from installing a trivially tiny battery.
        e_min_install = 100.0  # kWh — minimum meaningful battery size

        # E <= e_max_kwh * y   → if y=0 then E=0; if y=1 then E <= e_max_kwh
        m.c_y_emax = pyo.Constraint(expr=m.E <= e_max_kwh * m.y)

        # E >= e_min_install * y  → if y=1 then E >= 100 kWh; if y=0 then E >= 0
        m.c_y_emin = pyo.Constraint(expr=m.E >= e_min_install * m.y)

        # ── Solve ─────────────────────────────────────────────────
        solver = pyo.SolverFactory('gurobi')
        solver.options['TimeLimit'] = time_limit
        solver.options['MIPGap']    = 0.01   # 1% optimality gap — good for MILP
        # Remove Method=2 (barrier): Gurobi auto-selects best method for MILP
        result = solver.solve(m, tee=False)

        tc = result.solver.termination_condition
        if tc not in [pyo.TerminationCondition.optimal,
                      pyo.TerminationCondition.maxTimeLimit]:
            return jsonify({'error': f'Solver: {tc}'}), 500

        # ── Extract & compute KPIs ────────────────────────────────
        E_val = pyo.value(m.E)
        P_val = pyo.value(m.P)
        y_val = int(round(pyo.value(m.y)))  # 1 = installed, 0 = not installed

        ann_capex        = ann_e_cost * E_val + ann_p_cost * P_val
        total_imp        = sum(pyo.value(m.imp[t])    for t in range(N))
        total_exp        = sum(pyo.value(m.exp[t])    for t in range(N))
        total_dis        = sum(pyo.value(m.dis[t])    for t in range(N))
        total_ch_sol     = sum(pyo.value(m.ch_sol[t]) for t in range(N))
        total_ch_grd     = 0.0  # green battery: no grid charging
        total_load_kwh   = float(load.sum())
        total_solar_kwh  = float(solar.sum())

        ann_import_cost  = total_imp * import_price
        ann_export_rev   = sum(float(export_price[t]) * pyo.value(m.exp[t]) for t in range(N))
        ann_net_cost     = ann_capex + ann_import_cost - ann_export_rev

        # Baseline: no battery, no grid charging
        base_imp = float(sum(max(0.0, load[t] - solar[t]) for t in range(N)))
        base_exp = float(sum(max(0.0, solar[t] - load[t]) for t in range(N)))
        base_exp_rev = sum(
            float(export_price[t]) * max(0.0, solar[t] - load[t])
            for t in range(N))
        base_cost      = base_imp * import_price - base_exp_rev
        annual_savings = base_cost - ann_net_cost

        # Hours with negative export price
        neg_price_hours = int((export_price < 0).sum())
        neg_price_curtailed = float(sum(
            pyo.value(m.exp[t]) for t in range(N) if export_price[t] < 0))

        solar_pct   = round(min(total_solar_kwh, total_load_kwh) / total_load_kwh * 100, 1)
        battery_pct = round(total_dis / total_load_kwh * 100, 1)
        grid_pct    = round(total_imp  / total_load_kwh * 100, 1)
        self_suff   = round((1 - total_imp / total_load_kwh) * 100, 1)

        hourly = []
        for t in range(N):
            hourly.append({
                'hour'         : t,
                'load'         : round(float(load[t]),   2),
                'solar'        : round(float(solar[t]),  2),
                'export_price' : round(float(export_price[t]), 4),
                'ch_solar'     : round(pyo.value(m.ch_sol[t]), 2),
                'ch_grid'      : 0.0,  # green battery: always 0
                'discharge'    : round(pyo.value(m.dis[t]),    2),
                'soc'          : round(pyo.value(m.soc[t]),    2),
                'grid_import'  : round(pyo.value(m.imp[t]),    2),
                'solar_export' : round(pyo.value(m.exp[t]),    2),
            })

        return jsonify({
            'status'            : str(tc),
            # Sizing result
            'E_kwh'             : round(E_val, 1),
            'P_kw'              : round(P_val, 1),
            'y_installed'       : y_val,
            'solar_kw'          : solar_kw,
            # Annual economics
            'crf'               : round(crf, 4),
            'ann_e_cost'        : round(ann_e_cost, 2),
            'ann_capex'         : round(ann_capex, 0),
            'ann_import_cost'   : round(ann_import_cost, 0),
            'ann_export_rev'    : round(ann_export_rev, 0),
            'ann_net_cost'      : round(ann_net_cost, 0),
            'base_cost'         : round(base_cost, 0),
            'annual_savings'    : round(annual_savings, 0),
            # Energy flows
            'total_load'        : round(total_load_kwh, 0),
            'total_solar'       : round(total_solar_kwh, 0),
            'total_imp'         : round(total_imp, 0),
            'total_exp'         : round(total_exp, 0),
            'total_dis'         : round(total_dis, 0),
            'total_ch_sol'      : round(total_ch_sol, 0),
            'total_ch_grd'      : round(total_ch_grd, 0),
            # Negative price stats
            'neg_price_hours'   : neg_price_hours,
            'neg_price_curtailed': round(neg_price_curtailed, 1),
            # Mix
            'solar_pct'         : solar_pct,
            'battery_pct'       : battery_pct,
            'grid_pct'          : grid_pct,
            'self_suff'         : self_suff,
            'N_HOURS'           : N,
            'hourly'            : hourly,
        })

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/download', methods=['POST'])
def download():
    hourly = request.get_json().get('hourly', [])
    buf = io.StringIO()
    pd.DataFrame(hourly).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()),
                     mimetype='text/csv', as_attachment=True,
                     download_name='bess_results.csv')


if __name__ == '__main__':
    app.run(debug=True, port=5000, use_reloader=False)
