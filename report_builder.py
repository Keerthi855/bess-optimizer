"""
report_builder.py
==================
Turns the JSON result already produced by /optimize into a downloadable
Word report: KPI table, combined dashboard, individual charts, validation
checks (per the project's Validation & Sensitivity Guide), and an
investment verdict. No extra solving needed — just the same data the
browser already has.

Usage (from a Flask route):
    from report_builder import build_report_docx
    docx_bytes = build_report_docx(result_json)
    return send_file(io.BytesIO(docx_bytes), as_attachment=True,
                      download_name='bess_report.docx',
                      mimetype='application/vnd.openxmlformats-officedocument.wordprocessingml.document')
"""
import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 9,
    'axes.grid': True, 'grid.color': '#dddddd', 'grid.linewidth': 0.5,
    'figure.facecolor': 'white', 'axes.facecolor': 'white',
})

C_LOAD='#1f2937'; C_SOLAR='#f59e0b'; C_IMPORT='#3b82f6'
C_DIS='#16a34a'; C_EXPORT='#dc2626'; C_SOC='#7c3aed'; C_PRICE='#0891b2'


# ────────────────────────────────────────────────────────────────
# Chart builders — each returns a BytesIO PNG buffer
# ────────────────────────────────────────────────────────────────

def _buf(fig):
    b = io.BytesIO()
    fig.savefig(b, dpi=200, bbox_inches='tight')
    plt.close(fig)
    b.seek(0)
    return b


def chart_cost_breakdown(d):
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    labels = ['Baseline', 'CAPEX', 'Import cost', 'Export revenue', 'Net cost']
    values = [d['base_cost'], d['ann_capex'], d['ann_import_cost'], -d['ann_export_rev'], d['ann_net_cost']]
    colors = ['#94a3b8', '#f59e0b', C_IMPORT, C_DIS, '#111827']
    bars = ax.bar(labels, values, color=colors, width=0.55)
    span = max(abs(v) for v in values) or 1
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v + (span*0.02 if v >= 0 else -span*0.04),
                f"€{v:,.0f}", ha='center', va='bottom' if v >= 0 else 'top', fontsize=8, fontweight='bold')
    ax.axhline(0, color='#333', linewidth=0.8)
    ax.set_ylabel('€ / year')
    ax.set_title('Annual Cost Breakdown', fontsize=11, fontweight='bold', loc='left')
    plt.tight_layout()
    return _buf(fig)


def _week_slice(df, mode='summer'):
    """Pick the 168h window with the highest (summer) or lowest (winter) mean solar."""
    n_weeks = len(df) // 168
    if n_weeks < 1:
        return df
    means = [df['solar'].iloc[i*168:(i+1)*168].mean() for i in range(n_weeks)]
    idx = int(np.argmax(means)) if mode == 'summer' else int(np.argmin(means))
    return df.iloc[idx*168:(idx+1)*168].reset_index(drop=True)


def chart_week_dispatch(df, mode, title):
    w = _week_slice(df, mode)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.6), sharex=True,
                                    gridspec_kw={'height_ratios': [2.1, 1]})
    x = w['hour']
    ax1.plot(x, w['load'], color=C_LOAD, lw=1.2, label='Load')
    ax1.plot(x, w['solar'], color=C_SOLAR, lw=1.2, label='Solar')
    ax1.plot(x, w['grid_import'], color=C_IMPORT, lw=1.0, label='Grid import')
    ax1.plot(x, w['discharge'], color=C_DIS, lw=1.0, label='Discharge')
    ax1.plot(x, w['solar_export'], color=C_EXPORT, lw=1.0, label='Export')
    ax1.set_ylabel('kW')
    ax1.legend(fontsize=7, ncol=3, loc='upper right', frameon=False)
    ax1.set_title(title, fontsize=11, fontweight='bold', loc='left')

    ax2b = ax2.twinx()
    ax2.plot(x, w['soc'], color=C_SOC, lw=1.2)
    ax2b.plot(x, w['export_price'], color=C_PRICE, lw=0.9, ls='--')
    ax2b.axhline(0, color=C_PRICE, lw=0.5, alpha=0.5)
    ax2.set_ylabel('SOC (kWh)', color=C_SOC)
    ax2b.set_ylabel('€/kWh', color=C_PRICE)
    ax2.set_xlabel('Hour of window (168h = 1 week)')
    ax2.tick_params(axis='y', colors=C_SOC)
    ax2b.tick_params(axis='y', colors=C_PRICE)
    plt.tight_layout()
    return _buf(fig)


def chart_soc_annual(df, e_kwh, soc_floor=None, soc_ceiling=None):
    df = df.copy()
    df['day'] = df['hour'] // 24
    daily = df.groupby('day')['soc'].agg(['mean', 'min', 'max'])
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.fill_between(daily.index, daily['min'], daily['max'], color=C_SOC, alpha=0.18, label='Daily range')
    ax.plot(daily.index, daily['mean'], color=C_SOC, lw=1.2, label='Daily mean')
    if soc_floor is not None:
        ax.axhline(soc_floor, color='#b91c1c', lw=0.9, ls=':', label=f'Observed min ({soc_floor:.0f} kWh)')
    if soc_ceiling is not None:
        ax.axhline(soc_ceiling, color='#065f46', lw=0.9, ls=':', label=f'Observed max ({soc_ceiling:.0f} kWh)')
    ax.set_xlabel('Day of year')
    ax.set_ylabel('SOC (kWh)')
    ax.set_title('Battery State of Charge, Full Year', fontsize=11, fontweight='bold', loc='left')
    ax.legend(loc='upper right', fontsize=7, ncol=2, frameon=False)
    plt.tight_layout()
    return _buf(fig)


def chart_negative_price_zoom(df):
    neg = df[df['export_price'] < 0]
    if len(neg) == 0:
        fig, ax = plt.subplots(figsize=(7.0, 1.4))
        ax.axis('off')
        ax.text(0.5, 0.5, 'No negative-price hours present in this dataset.',
                ha='center', va='center', fontsize=10, color='#6b7280')
        plt.tight_layout()
        return _buf(fig)
    target = int(neg['hour'].values[len(neg)//2])
    zoom_start = max(0, target - 24)
    w = df.iloc[zoom_start:zoom_start+72]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.0, 4.0), sharex=True,
                                    gridspec_kw={'height_ratios': [1.6, 1]})
    x = w['hour']
    ax1.plot(x, w['solar'], color=C_SOLAR, lw=1.2, label='Solar')
    ax1.plot(x, w['solar_export'], color=C_EXPORT, lw=1.4, label='Export')
    ax1.plot(x, w['ch_solar'], color=C_DIS, lw=1.2, label='Charge (solar)')
    ax1.set_ylabel('kW')
    ax1.legend(fontsize=7, loc='upper right', frameon=False)
    ax1.set_title('Dispatch Around a Negative Export-Price Window', fontsize=11, fontweight='bold', loc='left')
    ax2.plot(x, w['export_price'], color=C_PRICE, lw=1.2)
    ax2.axhline(0, color='#333', lw=0.8)
    ax2.fill_between(x, w['export_price'], 0, where=(w['export_price'] < 0), color=C_EXPORT, alpha=0.15)
    ax2.set_ylabel('€/kWh')
    ax2.set_xlabel('Hour of year')
    plt.tight_layout()
    return _buf(fig)


def chart_combined_dashboard(d, df):
    e_kwh = d['E_kwh']
    fig = plt.figure(figsize=(11, 7.3))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.1, 1, 1], hspace=0.55, wspace=0.32)

    axA = fig.add_subplot(gs[0, 0])
    labels = ['Base', 'CAPEX', 'Import', '-Export', 'Net']
    values = [d['base_cost'], d['ann_capex'], d['ann_import_cost'], -d['ann_export_rev'], d['ann_net_cost']]
    axA.bar(labels, values, color=['#94a3b8', '#f59e0b', C_IMPORT, C_DIS, '#111827'])
    axA.axhline(0, color='#333', lw=0.7)
    axA.set_title('A. Annual Cost (€/yr)', fontsize=9, fontweight='bold', loc='left')
    axA.tick_params(axis='x', labelsize=6.5, rotation=20)

    axB = fig.add_subplot(gs[0, 1]); axB.axis('off')
    cycles = d['total_dis'] / e_kwh if e_kwh else 0
    cost_per_kwh = d['ann_capex'] / d['total_dis'] if d['total_dis'] else float('nan')
    kpi_text = (f"E* = {d['E_kwh']:.0f} kWh   P* = {d['P_kw']:.0f} kW\n"
                f"Install decision: {'YES' if d.get('y_installed', 1) else 'NO'}\n"
                f"Self-sufficiency: {d['self_suff']}%\n"
                f"Annual savings: €{d['annual_savings']:,.0f}\n"
                f"Cycles/year: {cycles:.1f}\n"
                f"Cost/kWh stored: €{cost_per_kwh:.2f}\n"
                f"Neg-price hours: {d['neg_price_hours']}")
    axB.text(0.02, 0.95, "B. Key Result Metrics", fontsize=9, fontweight='bold', va='top', transform=axB.transAxes)
    axB.text(0.02, 0.78, kpi_text, fontsize=8, va='top', transform=axB.transAxes, family='monospace')

    axC = fig.add_subplot(gs[0, 2])
    axC.pie([d['solar_pct'], d['battery_pct'], d['grid_pct']], labels=['Solar', 'Battery', 'Grid'],
            autopct='%1.0f%%', colors=[C_SOLAR, C_DIS, C_IMPORT], textprops={'fontsize': 7.5},
            wedgeprops={'width': 0.4})
    axC.set_title('C. Load Coverage Mix', fontsize=9, fontweight='bold', loc='left')

    axD = fig.add_subplot(gs[1, 0:2])
    w = _week_slice(df, 'summer')
    axD.plot(w['hour'], w['load'], color=C_LOAD, lw=1, label='Load')
    axD.plot(w['hour'], w['solar'], color=C_SOLAR, lw=1, label='Solar')
    axD.plot(w['hour'], w['grid_import'], color=C_IMPORT, lw=0.9, label='Import')
    axD.plot(w['hour'], w['discharge'], color=C_DIS, lw=0.9, label='Discharge')
    axD.plot(w['hour'], w['solar_export'], color=C_EXPORT, lw=0.9, label='Export')
    axD.set_title('D. Summer Week Dispatch', fontsize=9, fontweight='bold', loc='left')
    axD.set_ylabel('kW', fontsize=8)
    axD.legend(fontsize=6.5, ncol=5, loc='upper center', bbox_to_anchor=(0.5, -0.18), frameon=False)

    axE = fig.add_subplot(gs[1, 2])
    w2 = _week_slice(df, 'winter')
    axE.plot(w2['hour'], w2['load'], color=C_LOAD, lw=0.9)
    axE.plot(w2['hour'], w2['solar'], color=C_SOLAR, lw=0.9)
    axE.plot(w2['hour'], w2['grid_import'], color=C_IMPORT, lw=0.8)
    axE.plot(w2['hour'], w2['discharge'], color=C_DIS, lw=0.8)
    axE.set_title('E. Winter Week Dispatch', fontsize=9, fontweight='bold', loc='left')
    axE.set_ylabel('kW', fontsize=8)
    axE.tick_params(labelsize=6.5)

    axF = fig.add_subplot(gs[2, 0:2])
    dfc = df.copy(); dfc['day'] = dfc['hour']//24
    daily = dfc.groupby('day')['soc'].agg(['mean', 'min', 'max'])
    axF.fill_between(daily.index, daily['min'], daily['max'], color=C_SOC, alpha=0.18)
    axF.plot(daily.index, daily['mean'], color=C_SOC, lw=1.1)
    axF.set_title('F. Full-Year State of Charge', fontsize=9, fontweight='bold', loc='left')
    axF.set_xlabel('Day of year', fontsize=8)
    axF.set_ylabel('kWh', fontsize=8)

    axG = fig.add_subplot(gs[2, 2])
    neg = df[df['export_price'] < 0]
    if len(neg):
        target = int(neg['hour'].values[len(neg)//2])
        zoom_start = max(0, target-24)
        w3 = df.iloc[zoom_start:zoom_start+72]
        axG2 = axG.twinx()
        axG.plot(w3['hour'], w3['solar_export'], color=C_EXPORT, lw=1.1)
        axG.plot(w3['hour'], w3['ch_solar'], color=C_DIS, lw=1.1)
        axG2.plot(w3['hour'], w3['export_price'], color=C_PRICE, lw=0.9, ls='--')
        axG2.axhline(0, color=C_PRICE, lw=0.5, alpha=0.5)
        axG2.tick_params(labelsize=6.5)
    else:
        axG.text(0.5, 0.5, 'No negative\nprice hours', ha='center', va='center', fontsize=8, transform=axG.transAxes)
    axG.set_title('G. Negative-Price Window', fontsize=9, fontweight='bold', loc='left')
    axG.tick_params(labelsize=6.5)

    fig.suptitle('BESS Optimization — Combined Results Dashboard', fontsize=13, fontweight='bold', y=0.995)
    return _buf(fig)


# ────────────────────────────────────────────────────────────────
# Validation checks (per the project's Validation & Sensitivity Guide)
# ────────────────────────────────────────────────────────────────

def compute_validation(d, df):
    rows = []
    e_kwh = d['E_kwh']

    bal_err = (df['solar'] + df['discharge'] + df['grid_import']
               - df['load'] - df['ch_solar'] - df['ch_grid'] - df['solar_export']).abs().max()
    rows.append(("1.1 Power balance (every hour)", f"max error {bal_err:.2e} kW",
                 "PASS" if bal_err < 0.01 else "FAIL"))

    viol_cd = ((df['ch_solar'] + df['ch_grid'] > 0.1) & (df['discharge'] > 0.1)).sum()
    rows.append(("1.3 No simultaneous charge + discharge", f"{viol_cd} violations",
                 "PASS" if viol_cd == 0 else "FAIL"))

    viol_ie = ((df['grid_import'] > 0.1) & (df['solar_export'] > 0.1)).sum()
    rows.append(("1.4 No simultaneous import + export", f"{viol_ie} violations",
                 "PASS" if viol_ie == 0 else "FAIL"))

    soc_min, soc_max = df['soc'].min(), df['soc'].max()
    rows.append(("1.5 SOC stays within observed bounds", f"{soc_min:.0f}\u2013{soc_max:.0f} kWh", "INFO"))

    cycles = d['total_dis'] / e_kwh if e_kwh else 0
    rows.append(("2.2 Cycles/year in plausible range (50\u20131000)", f"{cycles:.1f} cycles/yr",
                 "PASS" if 50 <= cycles <= 1000 else "FAIL"))

    neg = df[df['export_price'] < 0]
    max_neg_exp = neg['solar_export'].max() if len(neg) else 0.0
    rows.append(("2.5 Export \u2248 0 during negative prices", f"max {max_neg_exp:.1f} kW exported",
                 "PASS" if max_neg_exp < 1.0 else "FAIL"))

    if d['annual_savings'] > 0:
        payback = (d.get('battery_cost', float('nan')) * e_kwh) / d['annual_savings'] if 'battery_cost' in d else float('nan')
        payback_txt = f"{payback:.1f} yrs" if payback == payback else "n/a (battery_cost not provided)"
        payback_ok = "PASS" if (payback == payback and 3 <= payback <= 20) else "FAIL"
    else:
        payback_txt = "no payback \u2014 savings are negative"
        payback_ok = "FAIL"
    rows.append(("3.1 Simple payback in reasonable range (5\u201315 yrs)", payback_txt, payback_ok))

    cost_per_kwh = d['ann_capex'] / d['total_dis'] if d['total_dis'] else float('nan')
    rows.append(("3.2 Cost per kWh stored (plausible \u20ac0.08\u20130.25)", f"\u20ac{cost_per_kwh:.2f}/kWh",
                 "PASS" if 0.08 <= cost_per_kwh <= 0.25 else "FAIL"))

    return rows


# ────────────────────────────────────────────────────────────────
# Word document assembly
# ────────────────────────────────────────────────────────────────

def _set_cell_shading(cell, hex_color):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement('w:shd')
    shd.set(qn('w:val'), 'clear')
    shd.set(qn('w:color'), 'auto')
    shd.set(qn('w:fill'), hex_color)
    tcPr.append(shd)


def _style_table_header(row, bg="1F2937"):
    for cell in row.cells:
        _set_cell_shading(cell, bg)
        for p in cell.paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                r.font.size = Pt(9.5)


def build_report_docx(result_json: dict) -> bytes:
    d = result_json
    df = pd.DataFrame(d['hourly'])

    doc = Document()
    section = doc.sections[0]
    section.page_height, section.page_width = Cm(29.7), Cm(21.0)  # A4
    section.top_margin = Cm(2.5); section.bottom_margin = Cm(2.5)
    section.left_margin = Cm(2.5); section.right_margin = Cm(3.0)

    style = doc.styles['Normal']
    style.font.name = 'Calibri'
    style.font.size = Pt(11)

    title = doc.add_heading('BESS Optimization — Results Report', level=1)
    title.runs[0].font.size = Pt(20)

    intro = doc.add_paragraph()
    intro.add_run(
        f"Battery size: E* = {d['E_kwh']:.0f} kWh, P* = {d['P_kw']:.0f} kW. "
        f"Dataset: {d['N_HOURS']} hourly rows. "
        f"Annual savings: €{d['annual_savings']:,.0f} "
        f"({'POSITIVE' if d['annual_savings'] > 0 else 'NEGATIVE'} investment under these inputs)."
    ).font.size = Pt(10.5)

    # KPI table
    doc.add_heading('Key Results', level=2)
    kpi_rows = [
        ("E* / P*", f"{d['E_kwh']:.0f} kWh / {d['P_kw']:.0f} kW"),
        ("Annualized CAPEX", f"€{d['ann_capex']:,.0f} / yr"),
        ("Net annual cost", f"€{d['ann_net_cost']:,.0f} / yr"),
        ("Baseline cost (no battery)", f"€{d['base_cost']:,.0f} / yr"),
        ("Annual savings", f"€{d['annual_savings']:,.0f} / yr"),
        ("Self-sufficiency", f"{d['self_suff']}%"),
        ("Solar / Battery / Grid share", f"{d['solar_pct']}% / {d['battery_pct']}% / {d['grid_pct']}%"),
        ("Negative-price hours", f"{d['neg_price_hours']}"),
    ]
    t = doc.add_table(rows=1, cols=2)
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    t.style = 'Light Grid Accent 1'
    hdr = t.rows[0].cells
    hdr[0].text, hdr[1].text = 'Metric', 'Value'
    _style_table_header(t.rows[0])
    for m, v in kpi_rows:
        row = t.add_row().cells
        row[0].text, row[1].text = m, v

    doc.add_paragraph()

    # Combined dashboard
    doc.add_heading('Combined Results Dashboard', level=2)
    doc.add_picture(chart_combined_dashboard(d, df), width=Cm(16))
    cap = doc.add_paragraph()
    cap.add_run("Figure 1. ").bold = True
    cap.add_run("Combined dashboard — cost breakdown, KPIs, load-coverage mix, summer/winter week dispatch, "
                "full-year SOC, and negative-price window behaviour.").font.size = Pt(9)

    doc.add_page_break()

    # Individual charts
    doc.add_heading('Individual Result Graphs', level=2)
    figs = [
        (chart_cost_breakdown(d), "Annual cost breakdown by component."),
        (chart_week_dispatch(df, 'summer', 'Summer Week Dispatch (high solar)'),
         "Representative high-solar week: dispatch of load, solar, import, discharge and export, with SOC and export price below."),
        (chart_week_dispatch(df, 'winter', 'Winter Week Dispatch (low solar)'),
         "Representative low-solar week: dispatch relies more heavily on grid import."),
        (chart_soc_annual(df, d['E_kwh'], df['soc'].min(), df['soc'].max()),
         "Full-year state of charge, daily mean and range."),
        (chart_negative_price_zoom(df),
         "Dispatch behaviour around a negative export-price window: export should drop toward zero while charging increases."),
    ]
    for i, (buf, caption) in enumerate(figs, start=2):
        doc.add_picture(buf, width=Cm(15))
        cap = doc.add_paragraph()
        cap.add_run(f"Figure {i}. ").bold = True
        cap.add_run(caption).font.size = Pt(9)

    # Validation table
    doc.add_heading('Validation Checks', level=2)
    checks = compute_validation(d, df)
    t2 = doc.add_table(rows=1, cols=3)
    t2.style = 'Light Grid Accent 1'
    hdr2 = t2.rows[0].cells
    hdr2[0].text, hdr2[1].text, hdr2[2].text = 'Check', 'Observed', 'Result'
    _style_table_header(t2.rows[0])
    for name, obs, res in checks:
        row = t2.add_row().cells
        row[0].text, row[1].text, row[2].text = name, obs, res
        color = RGBColor(0x16, 0x65, 0x34) if res == 'PASS' else (
            RGBColor(0x99, 0x1b, 0x1b) if res == 'FAIL' else RGBColor(0x37, 0x41, 0x51))
        for p in row[2].paragraphs:
            for r in p.runs:
                r.font.bold = True
                r.font.color.rgb = color

    # Verdict
    doc.add_heading('Investment Verdict', level=2)
    verdict = doc.add_paragraph()
    is_positive = d['annual_savings'] > 0
    lead = verdict.add_run(f"{'POSITIVE' if is_positive else 'NEGATIVE'} INVESTMENT. ")
    lead.bold = True
    lead.font.color.rgb = RGBColor(0x16, 0x65, 0x34) if is_positive else RGBColor(0x99, 0x1b, 0x1b)
    lead.font.size = Pt(11)
    detail = (
        f"Annual savings are €{d['annual_savings']:,.0f}. "
        + ("The battery pays for itself and reduces total site energy cost under these inputs."
           if is_positive else
           "The battery costs more than it saves under these inputs — reducing battery cost, "
           "increasing import price, or re-running with a different site profile would be needed "
           "to make this a positive investment.")
    )
    verdict.add_run(detail).font.size = Pt(10.5)

    buf = io.BytesIO()
    doc.save(buf)
    return buf.getvalue()
