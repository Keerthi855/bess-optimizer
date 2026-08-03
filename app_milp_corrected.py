"""
BESS Optimizer — MILP battery sizing and dispatch (corrected)
=============================================================

One file, two ways to run it:

  Web application (unchanged interface, works with the existing
  templates/index.html and report_builder.py):

      python app_milp_corrected.py
      -> http://127.0.0.1:5000

  Batch / sensitivity runs from the command line (no browser, no Flask):

      # single run
      python app_milp_corrected.py --csv site.csv --battery-cost 210 \
             --lifetime 15 --import-price 0.30 --e-max 1500

      # one-at-a-time sweep, results written to a CSV
      python app_milp_corrected.py --csv site.csv --e-max 1500 \
             --sweep import_price=0.15,0.20,0.25,0.30,0.35,0.40 \
             --out sweep_import_price.csv

Both paths call the same solve_bess() function below, so the web result and
the batch result for identical inputs are identical by construction.

-----------------------------------------------------------------------------
CORRECTIONS vs. the original app_milp.py — each marked [FIX n] in the code
-----------------------------------------------------------------------------

[FIX 1]  soc[0] anchored to E, not e_max_kwh.
         Original:  soc[0] == e_max_kwh * 0.5
         Corrected: soc[0] == 0.5 * E
[FIX 2]  SOC lower bound anchored to E, not e_max_kwh.
         Original:  soc[t] >= e_max_kwh * soc_min_pct
         Corrected: soc[t] >= soc_min_pct * E
         Both corrected constraints are (constant x variable), i.e. linear —
         the same algebraic form as the upper bound soc[t] <= soc_max_pct * E,
         which the original already expressed correctly. Neither term is
         bilinear, so no McCormick reformulation is required. Under the
         original form these two constraints together forced
             E >= 0.5 * e_max_kwh / soc_max_pct = 0.526 * e_max_kwh
         at every solve, so the reported "optimum" was confined to
         [0.526 * e_max_kwh, e_max_kwh] — a window set by a user input that is
         not a physical property of any battery.
[FIX 3]  Cyclic SOC condition soc[N] >= soc[0]: the battery may not end the
         horizon emptier than it started, which would book unpaid-for energy
         into the objective.
[FIX 4]  y = 0 ("do not install") is now reachable. Previously E = 0 made
         soc[0] = 0.5 * e_max_kwh infeasible, so the model installed a battery
         at any price — at 5000 EUR/kWh it still installed and reported
         -129,704 EUR/yr of "savings".
[FIX 5]  PV curtailment variable curt[t], and the no-battery baseline is
         allowed to curtail too. Without it the power balance is an equality
         with no slack, so all PV must be consumed, stored or exported —
         forcing export at negative prices and measuring the battery against a
         baseline that cannot switch the inverter off.
[FIX 6]  Load-coverage shares form a disjoint decomposition summing to 100 %.
         The original solar share used min(total_solar, total_load), which
         double-counted stored and exported solar; the three shares summed to
         116-155 %.
[FIX 7]  Default MIP gap tightened 1 % -> 0.1 %. On an objective of order
         1e5 EUR/yr a 1 % gap admits ~1,000 EUR/yr of slack, comparable to the
         annual savings being measured.
[FIX 8]  Solver auto-selection with fallback (HiGHS / Gurobi / CBC / GLPK);
         the solver actually used is reported.
[FIX 9]  Blank form fields fall back to defaults instead of raising
         ValueError on float('').
[FIX 10] Power-balance residual computed from unrounded solver output, so the
         validation check does not fail on two-decimal CSV rounding.
[FIX 11] Grid charging must be backed by real grid import: ch_grd[t] <= imp[t].
         Nothing previously linked the two. Because ch_sol competes with exp
         under the surplus cap while ch_grd did not, the solver routed
         solar-sourced charging through ch_grd to escape that cap: a test run
         reported 101,881 kWh of "grid charging" in hours with zero grid
         import. Energy conservation still held so the cost was right, but the
         reported split was not — and the "green battery" variant, which
         simply deletes ch_grd, therefore restricted nothing.
"""

import argparse
import io
import sys
import traceback

import numpy as np
import pandas as pd


# ═══════════════════════════════════════════════════════════════════════════
#  Financial helper
# ═══════════════════════════════════════════════════════════════════════════

def capital_recovery_factor(r, n):
    """Convert a one-time capital cost into an equivalent uniform annual cost."""
    if r == 0:
        return 1.0 / n
    return r * (1 + r) ** n / ((1 + r) ** n - 1)


def _pick_solver(pyo, preferred=('appsi_highs', 'gurobi', 'cbc', 'glpk')):
    """[FIX 8] Return (name, solver) for the first available solver."""
    for name in preferred:
        try:
            s = pyo.SolverFactory(name)
            if s is not None and s.available(exception_flag=False):
                return name, s
        except Exception:
            continue
    raise RuntimeError('No MILP solver available. Install one of: '
                       + ', '.join(preferred))


# ═══════════════════════════════════════════════════════════════════════════
#  THE MODEL — used by both the web app and the command line
# ═══════════════════════════════════════════════════════════════════════════

def solve_bess(load, solar, export_price,
               battery_cost=300.0, lifetime_yrs=10.0, discount_rate=0.08,
               import_price=0.25, solar_kw=500.0,
               eta_charge=0.95, eta_discharge=0.95,
               soc_min_pct=0.10, soc_max_pct=0.95, c_rate=0.5,
               e_max_kwh=10000.0, allow_curtailment=True,
               mip_gap=0.001, time_limit=300, include_hourly=True):
    """
    Co-optimise battery energy capacity E (kWh), power capacity P (kW) and the
    full hourly dispatch, minimising annualised CAPEX plus net grid cost.

    load, solar        : hourly arrays in kW
    export_price       : hourly array in EUR/kWh (may be negative)

    Returns a dict of results (the same structure the /optimize route returns).
    """
    import pyomo.environ as pyo

    load = np.asarray(load, dtype=float)
    solar = np.asarray(solar, dtype=float)
    export_price = np.asarray(export_price, dtype=float)
    N = len(load)

    crf = capital_recovery_factor(discount_rate, lifetime_yrs)
    ann_e_cost = battery_cost * crf            # EUR/kWh/year
    ann_p_cost = battery_cost * 0.2 * crf      # EUR/kW/year
    p_hard_cap = solar_kw                      # shared inverter / connection

    m = pyo.ConcreteModel()
    m.T = pyo.Set(initialize=range(N))
    m.T1 = pyo.Set(initialize=range(N + 1))

    # ── Decision variables ────────────────────────────────────────────────
    m.E = pyo.Var(bounds=(0, e_max_kwh), domain=pyo.NonNegativeReals)
    m.P = pyo.Var(bounds=(0, min(e_max_kwh * c_rate, p_hard_cap)),
                  domain=pyo.NonNegativeReals)
    m.soc = pyo.Var(m.T1, domain=pyo.NonNegativeReals)
    m.ch_sol = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.ch_grd = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.dis = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.imp = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.exp = pyo.Var(m.T, domain=pyo.NonNegativeReals)
    m.curt = pyo.Var(m.T, domain=pyo.NonNegativeReals)      # [FIX 5]

    m.u = pyo.Var(m.T, domain=pyo.Binary)   # charge / discharge exclusivity
    m.z = pyo.Var(m.T, domain=pyo.Binary)   # import / export exclusivity
    m.y = pyo.Var(domain=pyo.Binary)        # install-or-not

    # ── Objective ─────────────────────────────────────────────────────────
    def obj_rule(mo):
        capex = ann_e_cost * mo.E + ann_p_cost * mo.P
        import_c = sum(import_price * mo.imp[t] for t in range(N))
        export_rev = sum(float(export_price[t]) * mo.exp[t] for t in range(N))
        return capex + import_c - export_rev
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    # ── C1: hourly power balance  [FIX 5: curtailment sink added] ─────────
    m.c_bal = pyo.Constraint(m.T, rule=lambda mo, t:
        solar[t] + mo.dis[t] + mo.imp[t]
        == load[t] + mo.ch_sol[t] + mo.ch_grd[t] + mo.exp[t] + mo.curt[t])

    # ── C2-C4: solar surplus allocation ───────────────────────────────────
    surplus_max = {t: max(0.0, solar[t] - load[t]) for t in range(N)}
    m.c_exp_lim = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.exp[t] <= surplus_max[t])
    m.c_chsol_lim = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.ch_sol[t] <= surplus_max[t])
    m.c_surplus_split = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.exp[t] + mo.ch_sol[t] + mo.curt[t] <= surplus_max[t])

    if not allow_curtailment:
        m.c_no_curt = pyo.Constraint(m.T, rule=lambda mo, t: mo.curt[t] == 0)

    # ── C4b: [FIX 11] grid charging must be backed by real import ─────────
    m.c_chgrd_imp = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.ch_grd[t] <= mo.imp[t])

    # ── C5: SOC dynamics.  [FIX 1] initial condition scales with E ────────
    def soc_rule(mo, t):
        if t == 0:
            return mo.soc[0] == 0.5 * mo.E
        ch = mo.ch_sol[t - 1] + mo.ch_grd[t - 1]
        return mo.soc[t] == mo.soc[t - 1] + ch * eta_charge \
                            - mo.dis[t - 1] / eta_discharge
    m.c_soc = pyo.Constraint(m.T1, rule=soc_rule)

    # ── C6/C7: SOC bounds.  [FIX 2] lower bound scales with E ─────────────
    m.c_soc_max = pyo.Constraint(m.T1, rule=lambda mo, t:
        mo.soc[t] <= soc_max_pct * mo.E)
    m.c_soc_min = pyo.Constraint(m.T1, rule=lambda mo, t:
        mo.soc[t] >= soc_min_pct * mo.E)

    # ── C7b: [FIX 3] cyclic condition ─────────────────────────────────────
    m.c_soc_cyclic = pyo.Constraint(expr=m.soc[N] >= m.soc[0])

    # ── C8-C10: power limits and C-rate link ──────────────────────────────
    m.c_pch = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.ch_sol[t] + mo.ch_grd[t] <= mo.P)
    m.c_pdis = pyo.Constraint(m.T, rule=lambda mo, t: mo.dis[t] <= mo.P)
    m.c_cr = pyo.Constraint(expr=m.P <= c_rate * m.E)

    # ── Mixed-integer extensions ──────────────────────────────────────────
    p_M = min(e_max_kwh * c_rate, p_hard_cap)      # tightest valid big-M on P
    m.c_u_charge = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.ch_sol[t] + mo.ch_grd[t] <= p_M * mo.u[t])
    m.c_u_discharge = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.dis[t] <= p_M * (1 - mo.u[t]))

    bigM_grid = float(load.max() + solar.max())
    m.c_z_import = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.imp[t] <= bigM_grid * mo.z[t])
    m.c_z_export = pyo.Constraint(m.T, rule=lambda mo, t:
        mo.exp[t] <= bigM_grid * (1 - mo.z[t]))

    # [FIX 4] with soc[0] = 0.5*E, y = 0 => E = 0 => soc = 0 is feasible
    e_min_install = 100.0
    m.c_y_emax = pyo.Constraint(expr=m.E <= e_max_kwh * m.y)
    m.c_y_emin = pyo.Constraint(expr=m.E >= e_min_install * m.y)

    # ── Solve ─────────────────────────────────────────────────────────────
    solver_name, solver = _pick_solver(pyo)
    if solver_name == 'appsi_highs':
        solver.config.time_limit = time_limit
        solver.config.mip_gap = mip_gap
        result = solver.solve(m)
    else:
        keys = {'gurobi': ('TimeLimit', 'MIPGap'),
                'cbc': ('seconds', 'ratioGap'),
                'glpk': ('tmlim', 'mipgap')}
        tk, gk = keys.get(solver_name, ('TimeLimit', 'MIPGap'))
        solver.options[tk] = time_limit
        solver.options[gk] = mip_gap
        result = solver.solve(m, tee=False)

    tc = result.solver.termination_condition
    ok = [pyo.TerminationCondition.optimal,
          pyo.TerminationCondition.maxTimeLimit,
          pyo.TerminationCondition.feasible]
    if tc not in ok:
        raise RuntimeError(f'Solver terminated with condition: {tc}')

    # ── Extract ───────────────────────────────────────────────────────────
    E_val, P_val = pyo.value(m.E), pyo.value(m.P)
    y_val = int(round(pyo.value(m.y)))
    arr = lambda v, n=N: np.array([pyo.value(v[t]) for t in range(n)], float)
    imp_a, exp_a, dis_a = arr(m.imp), arr(m.exp), arr(m.dis)
    chs_a, chg_a, cur_a = arr(m.ch_sol), arr(m.ch_grd), arr(m.curt)
    soc_a = arr(m.soc, N + 1)

    ann_capex = ann_e_cost * E_val + ann_p_cost * P_val
    total_imp, total_exp, total_dis = imp_a.sum(), exp_a.sum(), dis_a.sum()
    total_ch_sol, total_ch_grd, total_curt = chs_a.sum(), chg_a.sum(), cur_a.sum()
    total_load_kwh, total_solar_kwh = load.sum(), solar.sum()

    ann_import_cost = total_imp * import_price
    ann_export_rev = float((export_price * exp_a).sum())
    ann_net_cost = ann_capex + ann_import_cost - ann_export_rev

    # ── Baseline: no battery.  [FIX 5] baseline may curtail too ───────────
    base_surplus = np.maximum(0.0, solar - load)
    base_imp = float(np.maximum(0.0, load - solar).sum())
    base_exported = (np.where(export_price > 0, base_surplus, 0.0)
                     if allow_curtailment else base_surplus)
    base_exp_rev = float((export_price * base_exported).sum())
    base_cost = base_imp * import_price - base_exp_rev
    annual_savings = base_cost - ann_net_cost

    # ── [FIX 6] disjoint coverage decomposition (sums to load) ────────────
    sol_to_load = np.clip(solar - chs_a - exp_a - cur_a, 0, None)
    grd_to_load = np.clip(imp_a - chg_a, 0, None)
    cov = sol_to_load.sum() + dis_a.sum() + grd_to_load.sum()
    k = (total_load_kwh / cov) if cov > 0 else 0.0
    solar_pct = round(float(sol_to_load.sum()) * k / total_load_kwh * 100, 1)
    battery_pct = round(float(dis_a.sum()) * k / total_load_kwh * 100, 1)
    grid_pct = round(float(grd_to_load.sum()) * k / total_load_kwh * 100, 1)
    self_suff = round((1 - total_imp / total_load_kwh) * 100, 1)

    # ── [FIX 10] residual from unrounded values ───────────────────────────
    residual = (solar + dis_a + imp_a - load - chs_a - chg_a - exp_a - cur_a)
    power_balance_err = float(np.abs(residual).max())

    neg = export_price < 0
    upfront_energy = battery_cost * E_val
    upfront_total = upfront_energy + battery_cost * 0.2 * P_val
    pb_e = (upfront_energy / annual_savings) if annual_savings > 0 else None
    pb_t = (upfront_total / annual_savings) if annual_savings > 0 else None

    out = {
        'status': str(tc), 'solver': solver_name, 'mip_gap': mip_gap,
        'E_kwh': round(E_val, 1), 'P_kw': round(P_val, 1),
        'y_installed': y_val, 'solar_kw': solar_kw,
        'crf': round(crf, 4), 'ann_e_cost': round(ann_e_cost, 2),
        'ann_capex': round(ann_capex, 0),
        'ann_import_cost': round(ann_import_cost, 0),
        'ann_export_rev': round(ann_export_rev, 0),
        'ann_net_cost': round(ann_net_cost, 0),
        'base_cost': round(base_cost, 0),
        'annual_savings': round(annual_savings, 0),
        'upfront_capex_energy': round(upfront_energy, 0),
        'upfront_capex_total': round(upfront_total, 0),
        'payback_energy_only': round(pb_e, 1) if pb_e else None,
        'payback_total': round(pb_t, 1) if pb_t else None,
        'total_load': round(float(total_load_kwh), 0),
        'total_solar': round(float(total_solar_kwh), 0),
        'total_imp': round(float(total_imp), 0),
        'total_exp': round(float(total_exp), 0),
        'total_dis': round(float(total_dis), 0),
        'total_ch_sol': round(float(total_ch_sol), 0),
        'total_ch_grd': round(float(total_ch_grd), 0),
        'total_curtailed': round(float(total_curt), 0),
        'neg_price_hours': int(neg.sum()),
        'neg_price_curtailed': round(float(exp_a[neg].sum()), 1),
        'neg_max_export': round(float(exp_a[neg].max()), 1) if neg.any() else 0.0,
        'solar_pct': solar_pct, 'battery_pct': battery_pct,
        'grid_pct': grid_pct, 'self_suff': self_suff,
        'cycles_per_year': round(float(total_dis / E_val), 1) if E_val > 1e-6 else 0.0,
        'cost_per_kwh_stored': round(float(ann_capex / total_dis), 4) if total_dis > 0 else None,
        'power_balance_err': power_balance_err,
        'soc_floor_implied': round(0.5 * e_max_kwh / soc_max_pct, 1),
        'e_at_bound': bool(E_val >= e_max_kwh - max(1.0, 1e-3 * e_max_kwh)),
        'p_at_solar_cap': bool(P_val >= p_hard_cap - 1e-6),
        'p_at_crate': bool(P_val >= c_rate * E_val - 1e-6),
        'allow_curtailment': bool(allow_curtailment),
        'N_HOURS': N,
        'battery_cost': battery_cost, 'lifetime_yrs': lifetime_yrs,
        'discount_rate': discount_rate, 'import_price': import_price,
        'e_max_kwh': e_max_kwh, 'soc_min_pct': soc_min_pct,
        'soc_max_pct': soc_max_pct, 'eta_charge': eta_charge,
        'eta_discharge': eta_discharge, 'c_rate': c_rate,
    }

    if include_hourly:
        out['hourly'] = [{
            'hour': t,
            'load': round(float(load[t]), 2),
            'solar': round(float(solar[t]), 2),
            'export_price': round(float(export_price[t]), 4),
            'ch_solar': round(float(chs_a[t]), 2),
            'ch_grid': round(float(chg_a[t]), 2),
            'discharge': round(float(dis_a[t]), 2),
            'soc': round(float(soc_a[t]), 2),
            'grid_import': round(float(imp_a[t]), 2),
            'solar_export': round(float(exp_a[t]), 2),
            'curtailed': round(float(cur_a[t]), 2),
        } for t in range(N)]

    return out


REQUIRED_COLS = ['load_demand', 'solar_yield', 'export_price']


def read_site_csv(source):
    """Read an hourly CSV from a path or file-like object. Returns (L, S, P, n_dropped)."""
    df = pd.read_csv(source, encoding='utf-8-sig')      # utf-8-sig strips Excel BOM
    df.columns = [str(c).strip() for c in df.columns]
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f'Missing columns: {missing}. Need: load_demand (kW), '
                         f'solar_yield (kW), export_price (EUR/kWh)')
    before = len(df)
    df = df.dropna(subset=REQUIRED_COLS).reset_index(drop=True)
    return (df['load_demand'].values.astype(float),
            df['solar_yield'].values.astype(float),
            df['export_price'].values.astype(float),
            before - len(df))


# ═══════════════════════════════════════════════════════════════════════════
#  WEB APPLICATION
# ═══════════════════════════════════════════════════════════════════════════

from flask import Flask, render_template, request, jsonify, send_file

try:
    from report_builder import build_report_docx
except ImportError:                       # CLI use does not need the reporter
    build_report_docx = None

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/optimize', methods=['POST'])
def optimize():
    try:
        def ff(name, default):
            """[FIX 9] blank field -> default instead of float('') ValueError."""
            raw = request.form.get(name, '')
            return float(default) if raw is None or str(raw).strip() == '' else float(raw)

        if 'csv_file' not in request.files:
            return jsonify({'error': 'No CSV uploaded'}), 400
        load, solar, price, dropped = read_site_csv(request.files['csv_file'])

        res = solve_bess(
            load, solar, price,
            battery_cost=ff('battery_cost', 300),
            lifetime_yrs=ff('lifetime_yrs', 10),
            discount_rate=ff('discount_rate', 0.08),
            import_price=ff('import_price', 0.25),
            solar_kw=ff('solar_kw', 500),
            eta_charge=ff('eta_charge', 0.95),
            eta_discharge=ff('eta_discharge', 0.95),
            soc_min_pct=ff('soc_min_pct', 0.10),
            soc_max_pct=ff('soc_max_pct', 0.95),
            c_rate=ff('c_rate', 0.5),
            e_max_kwh=ff('e_max_kwh', 10000),
            allow_curtailment=bool(int(ff('allow_curtailment', 1))),
            mip_gap=ff('mip_gap', 0.001),
            time_limit=int(ff('time_limit', 300)))
        res['rows_dropped'] = dropped
        return jsonify(res)

    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


@app.route('/download', methods=['POST'])
def download():
    hourly = request.get_json().get('hourly', [])
    buf = io.StringIO()
    pd.DataFrame(hourly).to_csv(buf, index=False)
    buf.seek(0)
    return send_file(io.BytesIO(buf.getvalue().encode()), mimetype='text/csv',
                     as_attachment=True, download_name='bess_results.csv')


@app.route('/report', methods=['POST'])
def report():
    """Takes the JSON that /optimize returns and builds the Word report."""
    try:
        if build_report_docx is None:
            return jsonify({'error': 'report_builder not available'}), 500
        result_json = request.get_json()
        if not result_json or 'hourly' not in result_json:
            return jsonify({'error': 'No result data provided. Run /optimize first.'}), 400
        return send_file(
            io.BytesIO(build_report_docx(result_json)),
            mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            as_attachment=True, download_name='bess_report.docx')
    except Exception as e:
        return jsonify({'error': str(e), 'trace': traceback.format_exc()}), 500


# ═══════════════════════════════════════════════════════════════════════════
#  COMMAND LINE — single runs and sensitivity sweeps
# ═══════════════════════════════════════════════════════════════════════════

REPORT_KEYS = ['E_kwh', 'P_kw', 'y_installed', 'annual_savings', 'payback_total',
               'cycles_per_year', 'cost_per_kwh_stored', 'self_suff',
               'total_curtailed', 'neg_max_export', 'e_at_bound', 'solver']


def _cli():
    ap = argparse.ArgumentParser(
        description='BESS MILP sizing — batch runner. Omit --csv to start the web app.')
    ap.add_argument('--csv', help='hourly CSV: load_demand, solar_yield, export_price')
    ap.add_argument('--battery-cost', type=float, default=300)
    ap.add_argument('--lifetime', type=float, default=10)
    ap.add_argument('--discount-rate', type=float, default=0.08)
    ap.add_argument('--import-price', type=float, default=0.25)
    ap.add_argument('--solar-kw', type=float, default=500)
    ap.add_argument('--eta-charge', type=float, default=0.95)
    ap.add_argument('--eta-discharge', type=float, default=0.95)
    ap.add_argument('--soc-min', type=float, default=0.10)
    ap.add_argument('--soc-max', type=float, default=0.95)
    ap.add_argument('--c-rate', type=float, default=0.5)
    ap.add_argument('--e-max', type=float, default=10000)
    ap.add_argument('--no-curtailment', action='store_true')
    ap.add_argument('--mip-gap', type=float, default=0.001)
    ap.add_argument('--time-limit', type=int, default=300)
    ap.add_argument('--sweep', help='PARAM=v1,v2,v3  e.g. import_price=0.20,0.25,0.30')
    ap.add_argument('--out', help='write sweep results to this CSV')
    a = ap.parse_args()

    if not a.csv:
        print('No --csv given; starting web application on http://127.0.0.1:5000')
        app.run(debug=True, port=5000, use_reloader=False)
        return

    load, solar, price, dropped = read_site_csv(a.csv)
    if dropped:
        print(f'note: dropped {dropped} incomplete rows')

    base = dict(battery_cost=a.battery_cost, lifetime_yrs=a.lifetime,
                discount_rate=a.discount_rate, import_price=a.import_price,
                solar_kw=a.solar_kw, eta_charge=a.eta_charge,
                eta_discharge=a.eta_discharge, soc_min_pct=a.soc_min,
                soc_max_pct=a.soc_max, c_rate=a.c_rate, e_max_kwh=a.e_max,
                allow_curtailment=not a.no_curtailment,
                mip_gap=a.mip_gap, time_limit=a.time_limit,
                include_hourly=False)

    if not a.sweep:
        r = solve_bess(load, solar, price, **base)
        for k in REPORT_KEYS:
            print(f'  {k:22s} {r[k]}')
        return

    param, vals = a.sweep.split('=', 1)
    param = param.strip()
    if param not in base:
        sys.exit(f'Unknown sweep parameter "{param}". '
                 f'Choose from: {", ".join(sorted(base))}')
    values = [float(v) for v in vals.split(',')]

    rows = []
    for v in values:
        kw = dict(base); kw[param] = v
        if param == 'eta_charge':
            kw['eta_discharge'] = v          # sweep round-trip symmetrically
        r = solve_bess(load, solar, price, **kw)
        r[param] = v
        rows.append(r)
        print(f"{param}={v:<8g} E*={r['E_kwh']:8.1f}  P*={r['P_kw']:7.1f}  "
              f"savings={r['annual_savings']:10,.0f}  payback={r['payback_total']}  "
              f"bound_hit={r['e_at_bound']}", flush=True)

    if a.out:
        pd.DataFrame(rows).to_csv(a.out, index=False)
        print(f'\nwrote {len(rows)} runs -> {a.out}')


if __name__ == '__main__':
    _cli()
