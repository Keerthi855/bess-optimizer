"""
BESS Optimizer â€” Streamlit UI
Green Battery MILP (Solar-only charging)
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import io
import traceback

st.set_page_config(
    page_title="BESS Optimizer",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# â”€â”€ Custom CSS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stApp { background-color: #0f1117; }
    
    .kpi-card {
        background: linear-gradient(135deg, #1a1f2e, #252d3d);
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 4px;
    }
    .kpi-value {
        font-size: 2rem;
        font-weight: 700;
        color: #00d4aa;
        line-height: 1.1;
    }
    .kpi-label {
        font-size: 0.8rem;
        color: #8892a4;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-top: 4px;
    }
    .kpi-unit {
        font-size: 0.9rem;
        color: #4a90d9;
        font-weight: 500;
    }
    .section-header {
        background: linear-gradient(90deg, #1a3a5c, #0f1117);
        border-left: 4px solid #00d4aa;
        padding: 10px 16px;
        border-radius: 0 8px 8px 0;
        margin: 20px 0 12px 0;
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
    }
    .green-badge {
        background: #0d4a2e;
        border: 1px solid #00d4aa;
        color: #00d4aa;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 1px;
    }
    .warning-box {
        background: #2d1f00;
        border: 1px solid #f59e0b;
        border-radius: 8px;
        padding: 12px 16px;
        color: #fcd34d;
        font-size: 0.9rem;
    }
    div[data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 12px;
    }
</style>
""", unsafe_allow_html=True)


# â”€â”€ Helper functions â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def capital_recovery_factor(r, n):
    if r == 0:
        return 1.0 / n
    return r * (1 + r)**n / ((1 + r)**n - 1)


def run_optimizer(df, battery_cost, lifetime_yrs, discount_rate, import_price,
                  solar_kw, eta_charge, eta_discharge, soc_min_pct, soc_max_pct,
                  c_rate, e_max_kwh, time_limit):

    import pyomo.environ as pyo

    load         = df['load_demand'].values.astype(float)
    solar        = df['solar_yield'].values.astype(float)
    export_price = df['export_price'].values.astype(float)
    N            = len(df)

    crf        = capital_recovery_factor(discount_rate, lifetime_yrs)
    ann_e_cost = battery_cost * crf
    ann_p_cost = battery_cost * 0.2 * crf

    m    = pyo.ConcreteModel()
    m.T  = pyo.Set(initialize=range(N))
    m.T1 = pyo.Set(initialize=range(N + 1))

    m.E      = pyo.Var(bounds=(0, e_max_kwh),          domain=pyo.NonNegativeReals)
    m.P      = pyo.Var(bounds=(0, e_max_kwh * c_rate), domain=pyo.NonNegativeReals)
    m.soc    = pyo.Var(m.T1, domain=pyo.NonNegativeReals)
    m.ch_sol = pyo.Var(m.T,  domain=pyo.NonNegativeReals)
    m.dis    = pyo.Var(m.T,  domain=pyo.NonNegativeReals)
    m.imp    = pyo.Var(m.T,  domain=pyo.NonNegativeReals)
    m.exp    = pyo.Var(m.T,  domain=pyo.NonNegativeReals)
    m.u      = pyo.Var(m.T,  domain=pyo.Binary)
    m.z      = pyo.Var(m.T,  domain=pyo.Binary)
    m.y      = pyo.Var(domain=pyo.Binary)

    def obj_rule(mo):
        capex      = ann_e_cost * mo.E + ann_p_cost * mo.P
        import_c   = sum(import_price * mo.imp[t] for t in range(N))
        export_rev = sum(float(export_price[t]) * mo.exp[t] for t in range(N))
        return capex + import_c - export_rev
    m.obj = pyo.Objective(rule=obj_rule, sense=pyo.minimize)

    surplus_max = {t: max(0.0, solar[t] - load[t]) for t in range(N)}

    m.c_bal          = pyo.Constraint(m.T, rule=lambda mo, t:
        solar[t] + mo.dis[t] + mo.imp[t] == load[t] + mo.ch_sol[t] + mo.exp[t])
    m.c_exp_lim      = pyo.Constraint(m.T, rule=lambda mo, t: mo.exp[t] <= surplus_max[t])
    m.c_chsol_lim    = pyo.Constraint(m.T, rule=lambda mo, t: mo.ch_sol[t] <= surplus_max[t])
    m.c_surplus_split= pyo.Constraint(m.T, rule=lambda mo, t: mo.exp[t] + mo.ch_sol[t] <= surplus_max[t])

    soc_init = e_max_kwh * 0.5
    def soc_rule(mo, t):
        if t == 0:
            return mo.soc[t] == soc_init
        return mo.soc[t] == mo.soc[t-1] + mo.ch_sol[t-1] * eta_charge - mo.dis[t-1] / eta_discharge
    m.c_soc     = pyo.Constraint(m.T1, rule=soc_rule)
    m.c_soc_max = pyo.Constraint(m.T1, rule=lambda mo, t: mo.soc[t] <= soc_max_pct * mo.E)
    soc_floor   = e_max_kwh * soc_min_pct
    m.c_soc_min = pyo.Constraint(m.T1, rule=lambda mo, t: mo.soc[t] >= soc_floor)
    m.c_pch     = pyo.Constraint(m.T, rule=lambda mo, t: mo.ch_sol[t] <= mo.P)
    m.c_pdis    = pyo.Constraint(m.T, rule=lambda mo, t: mo.dis[t] <= mo.P)
    m.c_cr      = pyo.Constraint(expr=m.P <= c_rate * m.E)

    p_M = e_max_kwh * c_rate
    m.c_u_charge    = pyo.Constraint(m.T, rule=lambda mo, t: mo.ch_sol[t] <= p_M * mo.u[t])
    m.c_u_discharge = pyo.Constraint(m.T, rule=lambda mo, t: mo.dis[t] <= p_M * (1 - mo.u[t]))

    bigM_grid = float(max(load) + max(solar))
    m.c_z_import = pyo.Constraint(m.T, rule=lambda mo, t: mo.imp[t] <= bigM_grid * mo.z[t])
    m.c_z_export = pyo.Constraint(m.T, rule=lambda mo, t: mo.exp[t] <= bigM_grid * (1 - mo.z[t]))

    m.c_y_emax = pyo.Constraint(expr=m.E <= e_max_kwh * m.y)
    m.c_y_emin = pyo.Constraint(expr=m.E >= 100.0 * m.y)

    solver = pyo.SolverFactory('appsi_highs')
    solver.options['time_limit'] = time_limit
    solver.options['mip_rel_gap'] = 0.01
    result = solver.solve(m, tee=False)

    tc = result.solver.termination_condition
    if tc not in [pyo.TerminationCondition.optimal, pyo.TerminationCondition.maxTimeLimit]:
        raise RuntimeError(f"Solver status: {tc}")

    E_val = pyo.value(m.E)
    P_val = pyo.value(m.P)
    y_val = int(round(pyo.value(m.y)))

    ann_capex       = ann_e_cost * E_val + ann_p_cost * P_val
    total_imp       = sum(pyo.value(m.imp[t])    for t in range(N))
    total_exp       = sum(pyo.value(m.exp[t])    for t in range(N))
    total_dis       = sum(pyo.value(m.dis[t])    for t in range(N))
    total_ch_sol    = sum(pyo.value(m.ch_sol[t]) for t in range(N))
    total_load_kwh  = float(load.sum())
    total_solar_kwh = float(solar.sum())

    ann_import_cost = total_imp * import_price
    ann_export_rev  = sum(float(export_price[t]) * pyo.value(m.exp[t]) for t in range(N))
    ann_net_cost    = ann_capex + ann_import_cost - ann_export_rev

    base_imp     = float(sum(max(0.0, load[t] - solar[t]) for t in range(N)))
    base_exp_rev = sum(float(export_price[t]) * max(0.0, solar[t] - load[t]) for t in range(N))
    base_cost    = base_imp * import_price - base_exp_rev
    annual_savings = base_cost - ann_net_cost

    neg_price_hours     = int((export_price < 0).sum())
    neg_price_curtailed = float(sum(pyo.value(m.exp[t]) for t in range(N) if export_price[t] < 0))

    hourly_records = []
    for t in range(N):
        hourly_records.append({
            'hour'        : t,
            'load'        : round(float(load[t]), 2),
            'solar'       : round(float(solar[t]), 2),
            'export_price': round(float(export_price[t]), 4),
            'ch_solar'    : round(pyo.value(m.ch_sol[t]), 2),
            'discharge'   : round(pyo.value(m.dis[t]), 2),
            'soc'         : round(pyo.value(m.soc[t]), 2),
            'grid_import' : round(pyo.value(m.imp[t]), 2),
            'solar_export': round(pyo.value(m.exp[t]), 2),
        })

    return {
        'E_kwh'              : round(E_val, 1),
        'P_kw'               : round(P_val, 1),
        'y_installed'        : y_val,
        'crf'                : round(crf, 4),
        'ann_capex'          : round(ann_capex, 0),
        'ann_import_cost'    : round(ann_import_cost, 0),
        'ann_export_rev'     : round(ann_export_rev, 0),
        'ann_net_cost'       : round(ann_net_cost, 0),
        'base_cost'          : round(base_cost, 0),
        'annual_savings'     : round(annual_savings, 0),
        'total_load'         : round(total_load_kwh, 0),
        'total_solar'        : round(total_solar_kwh, 0),
        'total_imp'          : round(total_imp, 0),
        'total_exp'          : round(total_exp, 0),
        'total_dis'          : round(total_dis, 0),
        'total_ch_sol'       : round(total_ch_sol, 0),
        'neg_price_hours'    : neg_price_hours,
        'neg_price_curtailed': round(neg_price_curtailed, 1),
        'solar_pct'          : round(min(total_solar_kwh, total_load_kwh) / total_load_kwh * 100, 1),
        'battery_pct'        : round(total_dis / total_load_kwh * 100, 1),
        'grid_pct'           : round(total_imp / total_load_kwh * 100, 1),
        'self_suff'          : round((1 - total_imp / total_load_kwh) * 100, 1),
        'hourly'             : hourly_records,
        'status'             : str(tc),
    }


# â”€â”€ CHARTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
COLORS = {
    'solar'    : '#f59e0b',
    'battery'  : '#00d4aa',
    'grid'     : '#4a90d9',
    'load'     : '#e2e8f0',
    'export'   : '#a78bfa',
    'soc'      : '#34d399',
    'price'    : '#f87171',
    'bg'       : '#1a1f2e',
    'grid_line': '#2d3748',
}

CHART_LAYOUT = dict(
    paper_bgcolor='#0f1117',
    plot_bgcolor='#1a1f2e',
    font=dict(color='#e2e8f0', family='Arial'),
    xaxis=dict(gridcolor='#2d3748', showgrid=True),
    yaxis=dict(gridcolor='#2d3748', showgrid=True),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.15),
)

def chart_dispatch(df_h, day_start, n_days=7):
    h0 = day_start * 24
    h1 = min(h0 + n_days * 24, len(df_h))
    d  = df_h.iloc[h0:h1]
    x  = list(range(len(d)))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=d['solar'],        name='Solar',        fill='tozeroy', line=dict(color=COLORS['solar'],   width=1), fillcolor='rgba(245,158,11,0.3)'))
    fig.add_trace(go.Scatter(x=x, y=d['load'],         name='Load',         line=dict(color=COLORS['load'],    width=2, dash='dot')))
    fig.add_trace(go.Bar(    x=x, y=d['ch_solar'],     name='Charging',     marker_color='rgba(0,212,170,0.7)'))
    fig.add_trace(go.Bar(    x=x, y=d['discharge'],    name='Discharge',    marker_color='rgba(74,144,217,0.7)'))
    fig.add_trace(go.Scatter(x=x, y=d['grid_import'],  name='Grid Import',  line=dict(color=COLORS['grid'],    width=1.5), fill='tozeroy', fillcolor='rgba(74,144,217,0.15)'))
    fig.add_trace(go.Scatter(x=x, y=d['solar_export'], name='Solar Export', line=dict(color=COLORS['export'],  width=1.5)))
    fig.update_layout(**CHART_LAYOUT, title='Hourly Dispatch', barmode='overlay',
                      xaxis_title=f'Hours (day {day_start+1} â†’ {day_start+n_days})', yaxis_title='kW')
    return fig

def chart_soc(df_h, day_start, n_days=7):
    h0 = day_start * 24
    h1 = min(h0 + n_days * 24, len(df_h))
    d  = df_h.iloc[h0:h1]
    x  = list(range(len(d)))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.65, 0.35], vertical_spacing=0.08)
    fig.add_trace(go.Scatter(x=x, y=d['soc'], name='SOC (kWh)', fill='tozeroy', line=dict(color='#34d399', width=2), fillcolor='rgba(52,211,153,0.2)'), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=d['export_price']*1000, name='Export Price (EUR/MWh)', line=dict(color='#f87171', width=1.5), fill='tozeroy', fillcolor='rgba(248,113,113,0.15)'), row=2, col=1)
    fig.update_layout(**CHART_LAYOUT, title='State of Charge + Export Price', height=500)
    fig.update_yaxes(gridcolor='#2d3748')
    return fig

def chart_annual_energy(res):
    cats   = ['Solar Direct', 'Battery Discharge', 'Grid Import']
    values = [res['total_solar'] - res['total_ch_sol'], res['total_dis'], res['total_imp']]
    colors = [COLORS['solar'], COLORS['battery'], COLORS['grid']]
    fig = go.Figure(go.Pie(labels=cats, values=values, marker_colors=colors,
                           hole=0.55, textinfo='label+percent',
                           textfont=dict(color='#e2e8f0', size=12)))
    fig.update_layout(**CHART_LAYOUT, title='Annual Energy Mix', showlegend=False,
                      annotations=[dict(text=f"{res['self_suff']}%<br>Self-Suff.",
                                        x=0.5, y=0.5, font_size=14, font_color='#00d4aa',
                                        showarrow=False)])
    return fig

def chart_economics(res):
    cats   = ['CAPEX (ann.)', 'Import Cost', 'Export Revenue', 'Net Cost', 'Baseline Cost']
    values = [res['ann_capex'], res['ann_import_cost'], -res['ann_export_rev'],
              res['ann_net_cost'], res['base_cost']]
    colors = ['#4a90d9','#f59e0b','#00d4aa','#e2e8f0','#f87171']
    fig = go.Figure(go.Bar(x=cats, y=values, marker_color=colors,
                           text=[f"EUR{v:,.0f}" for v in values],
                           textposition='outside', textfont=dict(color='#e2e8f0')))
    fig.update_layout(**CHART_LAYOUT, title='Annual Economics (EUR/year)', yaxis_title='EUR/year',
                      showlegend=False)
    return fig

def chart_monthly(df_h):
    df_h = df_h.copy()
    df_h['month'] = (df_h['hour'] // 720).clip(0, 11)
    months = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    grp = df_h.groupby('month').sum()
    fig = go.Figure()
    fig.add_trace(go.Bar(x=months, y=grp['solar'],       name='Solar',        marker_color=COLORS['solar']))
    fig.add_trace(go.Bar(x=months, y=grp['ch_solar'],    name='Battery Charged', marker_color=COLORS['battery']))
    fig.add_trace(go.Bar(x=months, y=grp['discharge'],   name='Battery Discharged', marker_color='rgba(0,212,170,0.5)'))
    fig.add_trace(go.Bar(x=months, y=grp['grid_import'], name='Grid Import',   marker_color=COLORS['grid']))
    fig.add_trace(go.Bar(x=months, y=grp['solar_export'],name='Solar Export',  marker_color=COLORS['export']))
    fig.update_layout(**CHART_LAYOUT, title='Monthly Energy Flows (kWh)', barmode='group',
                      yaxis_title='kWh', xaxis_title='Month')
    return fig

def chart_price_heatmap(df_h):
    df_h = df_h.copy()
    df_h['day'] = df_h['hour'] // 24
    df_h['hour_of_day'] = df_h['hour'] % 24
    pivot = df_h.pivot_table(index='hour_of_day', columns='day', values='export_price', aggfunc='mean')
    fig = go.Figure(go.Heatmap(z=pivot.values, colorscale='RdYlGn',
                                zmid=0, colorbar=dict(title='EUR/kWh'),
                                x=list(range(pivot.shape[1])),
                                y=list(range(24))))
    fig.update_layout(**CHART_LAYOUT, title='Export Price Heatmap (Hour of Day Ã— Day of Year)',
                      xaxis_title='Day of Year', yaxis_title='Hour of Day',
                      yaxis=dict(autorange='reversed'))
    return fig


# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
# MAIN APP
# â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•

# â”€â”€ Header â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
col_t1, col_t2 = st.columns([3, 1])
with col_t1:
    st.markdown("# BESS Optimizer")
    st.markdown("**Battery Energy Storage System** â€” Green MILP Sizing Tool")
with col_t2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="green-badge">GREEN BATTERY | GUROBI MILP</span>', unsafe_allow_html=True)

st.divider()

# â”€â”€ Sidebar â€” Parameters â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
with st.sidebar:
    st.markdown("## âš™ï¸ Parameters")

    st.markdown("### Data Upload")
    uploaded = st.file_uploader("Upload CSV (8760 rows)", type=['csv'],
                                 help="Columns: load_demand (kW), solar_yield (kW), export_price (EUR/kWh)")
    st.caption("Need sample data? Download below.")

    # Sample CSV download
    sample_path = "sample_data/sample.csv"
    try:
        with open(sample_path, 'rb') as f:
            st.download_button("Download Sample CSV", f, "sample.csv", "text/csv")
    except FileNotFoundError:
        pass

    st.markdown("---")
    st.markdown("### Site")
    solar_kw = st.number_input("Solar Installed (kWp)", 50, 50000, 500, 50)

    st.markdown("### Battery")
    battery_cost  = st.number_input("Battery Cost (EUR/kWh)", 50, 1000, 300, 10)
    e_max_kwh     = st.number_input("Max Battery Size (kWh)", 100, 100000, 10000, 100)
    c_rate        = st.slider("C-Rate", 0.1, 1.0, 0.5, 0.05,
                               help="Power-to-energy ratio. 0.5 = 2-hour battery")
    eta_charge    = st.slider("Charge Efficiency", 0.80, 1.0, 0.95, 0.01)
    eta_discharge = st.slider("Discharge Efficiency", 0.80, 1.0, 0.95, 0.01)
    soc_min_pct   = st.slider("Min SOC (%)", 0, 30, 10, 1) / 100
    soc_max_pct   = st.slider("Max SOC (%)", 70, 100, 95, 1) / 100

    st.markdown("### Financial")
    import_price  = st.number_input("Import Tariff (EUR/kWh)", 0.05, 1.0, 0.25, 0.01,
                                     help="All-in electricity price including network charges and taxes")
    lifetime_yrs  = st.number_input("Battery Lifetime (years)", 5, 30, 10, 1)
    discount_rate = st.slider("Discount Rate (%)", 1, 20, 8, 1) / 100

    st.markdown("### Solver")
    time_limit = st.slider("Time Limit (seconds)", 60, 600, 300, 30)

    st.markdown("---")
    run_btn = st.button("Run Optimization", type="primary", use_container_width=True)


# â”€â”€ Main area â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
if uploaded is None:
    # Landing state
    st.markdown("""
    <div style="text-align:center; padding: 60px 20px;">
        <div style="font-size: 5rem;"></div>
        <h2 style="color: #00d4aa;">Upload your CSV to get started</h2>
        <p style="color: #8892a4; max-width: 500px; margin: 0 auto;">
            Upload an 8760-row CSV with <code>load_demand</code>, <code>solar_yield</code>, 
            and <code>export_price</code> columns. Then click <strong>Run Optimization</strong>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### CSV Format")
    st.dataframe(pd.DataFrame({
        'load_demand': [85.2, 82.1, 79.4, '...'],
        'solar_yield': [0.0,  0.0,  12.3, '...'],
        'export_price': [0.0412, 0.0387, -0.0123, '...']
    }), use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**load_demand** â€” Site consumption in kW (hourly average)")
    with col2:
        st.info("**solar_yield** â€” Solar PV output in kW")
    with col3:
        st.info("**export_price** â€” Espot price in EUR/kWh (can be negative)")

else:
    # â”€â”€ Load and preview data â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        df_input = pd.read_csv(uploaded)
        required = ['load_demand', 'solar_yield', 'export_price']
        missing  = [c for c in required if c not in df_input.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
            st.stop()

        df_input = df_input[required].dropna().reset_index(drop=True)

        # Data preview
        with st.expander("Data Preview", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", f"{len(df_input):,}")
            col2.metric("Total Load", f"{df_input['load_demand'].sum():,.0f} kWh")
            col3.metric("Total Solar", f"{df_input['solar_yield'].sum():,.0f} kWh")
            neg_hrs = (df_input['export_price'] < 0).sum()
            col4.metric("Negative Price Hours", f"{neg_hrs}")
            st.dataframe(df_input.head(24), use_container_width=True)

    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    # â”€â”€ Run optimizer â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if run_btn:
        with st.spinner("Running Gurobi MILP optimization... (this may take 1-5 minutes)"):
            try:
                res = run_optimizer(
                    df_input, battery_cost, lifetime_yrs, discount_rate, import_price,
                    solar_kw, eta_charge, eta_discharge, soc_min_pct, soc_max_pct,
                    c_rate, e_max_kwh, time_limit
                )
                st.session_state['result'] = res
                st.success(f"Optimization complete â€” Status: {res['status']}")
            except Exception as e:
                st.error(f"Optimization failed: {e}")
                with st.expander("Traceback"):
                    st.code(traceback.format_exc())
                st.stop()

    # â”€â”€ Show results â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if 'result' in st.session_state:
        res  = st.session_state['result']
        df_h = pd.DataFrame(res['hourly'])

        # â”€â”€ KPI Row 1 â€” Sizing â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown('<div class="section-header">Optimal Battery Sizing</div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Energy Capacity", f"{res['E_kwh']:,.0f} kWh",
                  help="Optimal battery energy size")
        k2.metric("Power Capacity", f"{res['P_kw']:,.0f} kW",
                  help="Optimal battery power size")
        k3.metric("Install Decision", "Installed" if res['y_installed'] else "Not Installed",
                  help="Binary install-or-not result")
        k4.metric("C-Rate (actual)", f"{res['P_kw']/res['E_kwh']:.2f}C" if res['E_kwh'] > 0 else "â€”")
        k5.metric("Duration", f"{res['E_kwh']/res['P_kw']:.1f} hrs" if res['P_kw'] > 0 else "â€”",
                  help="Energy / Power = storage duration")

        # â”€â”€ KPI Row 2 â€” Economics â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown('<div class="section-header">Annual Economics</div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("CAPEX (annualized)", f"EUR{res['ann_capex']:,.0f}")
        k2.metric("Import Cost",        f"EUR{res['ann_import_cost']:,.0f}")
        k3.metric("Export Revenue",     f"EUR{res['ann_export_rev']:,.0f}")
        k4.metric("Net Annual Cost",    f"EUR{res['ann_net_cost']:,.0f}")
        savings_delta = f"vs EUR{res['base_cost']:,.0f} baseline"
        k5.metric("Annual Savings",     f"EUR{res['annual_savings']:,.0f}", savings_delta,
                  delta_color="normal")

        # â”€â”€ KPI Row 3 â€” Energy â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown('<div class="section-header"> Annual Energy Flows</div>', unsafe_allow_html=True)
        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Self-Sufficiency",  f"{res['self_suff']}%")
        k2.metric("Solar Coverage",    f"{res['solar_pct']}%")
        k3.metric("Battery Coverage",  f"{res['battery_pct']}%")
        k4.metric("Grid Coverage",     f"{res['grid_pct']}%")
        k5.metric("Neg. Price Hours",  f"{res['neg_price_hours']} hrs")

        st.divider()

        # â”€â”€ CHARTS â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Dispatch", "SOC & Price", "Monthly", "Economics", "Price Heatmap"
        ])

        with tab1:
            col_ctrl1, col_ctrl2 = st.columns([1, 3])
            with col_ctrl1:
                max_day = len(df_h) // 24 - 7
                day_start = st.slider("Start day", 0, max(1, max_day), 172,
                                      help="Day 172 = peak summer")
                n_days = st.slider("Days shown", 1, 14, 7)
            with col_ctrl2:
                st.plotly_chart(chart_dispatch(df_h, day_start, n_days),
                                use_container_width=True)

        with tab2:
            col_ctrl1, col_ctrl2 = st.columns([1, 3])
            with col_ctrl1:
                day_start2 = st.slider("Start day ", 0, max(1, max_day), 172)
                n_days2    = st.slider("Days shown ", 1, 14, 7)
            with col_ctrl2:
                st.plotly_chart(chart_soc(df_h, day_start2, n_days2),
                                use_container_width=True)

        with tab3:
            st.plotly_chart(chart_monthly(df_h), use_container_width=True)

        with tab4:
            col1, col2 = st.columns([1, 1])
            with col1:
                st.plotly_chart(chart_economics(res), use_container_width=True)
            with col2:
                st.plotly_chart(chart_annual_energy(res), use_container_width=True)

        with tab5:
            st.plotly_chart(chart_price_heatmap(df_h), use_container_width=True)

        st.divider()

        # â”€â”€ Validation checks â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.markdown('<div class="section-header">Validation Checks</div>', unsafe_allow_html=True)
        df_h['balance_error'] = (df_h['solar'] + df_h['discharge'] + df_h['grid_import']
                                 - df_h['load'] - df_h['ch_solar'] - df_h['solar_export'])
        max_balance_err = df_h['balance_error'].abs().max()
        sim_violations  = ((df_h['ch_solar'] > 0.1) & (df_h['discharge'] > 0.1)).sum()
        imp_exp_viol    = ((df_h['grid_import'] > 0.1) & (df_h['solar_export'] > 0.1)).sum()
        ann_cycles      = res['total_dis'] / res['E_kwh'] if res['E_kwh'] > 0 else 0

        vc1, vc2, vc3, vc4 = st.columns(4)
        vc1.metric("Max Balance Error", f"{max_balance_err:.4f} kW",
                   delta="PASS" if max_balance_err < 0.1 else "FAIL",
                   delta_color="off")
        vc2.metric("Sim. Charge+Discharge", f"{sim_violations} hrs",
                   delta="PASS" if sim_violations == 0 else "FAIL",
                   delta_color="off")
        vc3.metric("Sim. Import+Export", f"{imp_exp_viol} hrs",
                   delta="PASS" if imp_exp_viol == 0 else "FAIL",
                   delta_color="off")
        vc4.metric("Annual Cycles", f"{ann_cycles:.0f}",
                   delta="Normal" if 50 < ann_cycles < 1000 else "Check",
                   delta_color="off")

        # â”€â”€ Download â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        st.divider()
        st.markdown('<div class="section-header">Export Results</div>', unsafe_allow_html=True)
        csv_buf = io.StringIO()
        df_h.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Hourly Results CSV",
            csv_buf.getvalue(),
            "bess_results.csv",
            "text/csv",
            use_container_width=False
        )

        # Summary table
        with st.expander("Full Results Summary"):
            summary = {
                'Parameter': ['E_kwh', 'P_kw', 'CRF', 'Ann. CAPEX (EUR)', 'Ann. Import Cost (EUR)',
                               'Ann. Export Revenue (EUR)', 'Ann. Net Cost (EUR)', 'Baseline Cost (EUR)',
                               'Annual Savings (EUR)', 'Self-Sufficiency (%)', 'Annual Cycles',
                               'Neg. Price Hours', 'Neg. Price Curtailed (kWh)'],
                'Value': [res['E_kwh'], res['P_kw'], res['crf'], res['ann_capex'],
                          res['ann_import_cost'], res['ann_export_rev'], res['ann_net_cost'],
                          res['base_cost'], res['annual_savings'], res['self_suff'],
                          f"{ann_cycles:.0f}", res['neg_price_hours'], res['neg_price_curtailed']]
            }
            st.dataframe(pd.DataFrame(summary), use_container_width=True)

