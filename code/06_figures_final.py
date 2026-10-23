"""Publication figures 2-4 in one consistent style (JCSM: TIFF 600 dpi line / 300 dpi halftone, ≤ 8 display items).

Fig. 2  participant flow
Fig. 3  convergent validity: SRT vs SRI (A) and SRT vs SD midpoint (B), hex-binned density + weighted fit
Fig. 4  forest plot: primary models, SD-midpoint models, extraction-rule sensitivity

Palette (validated, CVD-safe): SRT #1b6f9b, SD midpoint #c86f2a, SRI #7a4fb5; ink #222, muted #6b6b6b.
"""
import json, numpy as np, pandas as pd, matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch
from matplotlib.ticker import FixedLocator, NullFormatter, FuncFormatter

C = dict(SRT='#1b6f9b', SD='#c86f2a', SRI='#7a4fb5', ink='#222222', muted='#6b6b6b', grid='#e3e3e3', band='#f4f4f2')
plt.rcParams.update({'font.family': 'Liberation Sans', 'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9,
                     'xtick.labelsize': 8.5, 'ytick.labelsize': 8.5, 'axes.edgecolor': C['muted'], 'axes.linewidth': 0.8,
                     'xtick.color': C['muted'], 'ytick.color': C['muted'], 'axes.labelcolor': C['ink'], 'text.color': C['ink'],
                     'xtick.major.width': 0.8, 'ytick.major.width': 0.8, 'xtick.major.size': 3, 'ytick.major.size': 3,
                     'pdf.fonttype': 42, 'ps.fonttype': 42})

def save(fig, name):
    fig.savefig(f'out/{name}.png', dpi=300, bbox_inches='tight', facecolor='white')
    fig.savefig(f'out/{name}.tiff', dpi=600, bbox_inches='tight', facecolor='white', pil_kwargs={'compression': 'tiff_lzw'})
    plt.close(fig); print('saved', name)

def despine(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)

# ------------------------------------------------------------------ Fig. 2 flow
def fig_flow():
    steps = [('NHANES 2011–2014 participants\nwith wrist accelerometer data', 14693),
             ('7 consecutive valid days\n(individualized day boundary;\n≥ 960 worn min and ≥ 1 sleep episode per day)', 10238),
             ('Adults aged ≥ 20 years', 6796),
             ('Complete PHQ-9 (all 9 items)', 6271),
             ('Complete covariates', 5701),
             ('Analytic sample\n(mean daily sleep 3–15 h)', 5679)]
    excl = ['< 7 consecutive valid days (n = 4,455)', 'Aged < 20 years (n = 3,442)', 'PHQ-9 missing or incomplete (n = 525)',
            'Missing poverty–income ratio (n = 439); BMI, smoking,\nalcohol or chronic conditions (n = 131)', 'Implausible mean sleep duration (n = 22)']
    fig, ax = plt.subplots(figsize=(6.0, 7.2)); ax.set_xlim(0, 10); ax.set_ylim(0, 10.4); ax.axis('off')
    xL, w, h = 0.3, 5.0, 1.08
    ys = np.linspace(9.2, 0.6, len(steps))
    for i, ((lab, n), y) in enumerate(zip(steps, ys)):
        last = i == len(steps) - 1
        box = FancyBboxPatch((xL, y - h / 2), w, h, boxstyle='round,pad=0,rounding_size=0.12',
                             fc=C['SRT'] if last else 'white', ec=C['SRT'] if last else C['muted'], lw=0.9)
        ax.add_patch(box)
        ax.text(xL + w / 2, y + 0.16, lab, ha='center', va='center', fontsize=7.6, color='white' if last else C['ink'], linespacing=1.25)
        ax.text(xL + w / 2, y - 0.36, f'n = {n:,}', ha='center', va='center', fontsize=8.6, fontweight='bold', color='white' if last else C['ink'])
        if i < len(steps) - 1:
            y2 = ys[i + 1] + h / 2
            ax.annotate('', xy=(xL + w / 2, y2 + 0.02), xytext=(xL + w / 2, y - h / 2 - 0.02),
                        arrowprops=dict(arrowstyle='-|>', color=C['muted'], lw=0.9, shrinkA=0, shrinkB=0, mutation_scale=9))
            ym = (y - h / 2 + y2) / 2
            ax.plot([xL + w / 2, 5.9], [ym, ym], color=C['muted'], lw=0.8)
            ax.text(6.05, ym, 'Excluded: ' + excl[i], ha='left', va='center', fontsize=7.4, color=C['muted'], linespacing=1.25)
    save(fig, 'Fig2_flow')

# ------------------------------------------------------------------ Fig. 3 scatter
def fig_scatter():
    d = pd.read_csv('out/analysis_indiv_gap30.csv')
    d = d[(d.age_y >= 20)].dropna(subset=['phq9', 'SRT', 'SRI', 'SD_midpoint', 'race', 'educ', 'pir', 'bmi', 'smoking', 'alcohol', 'chronic_cat'])
    d = d[d.implausible_sleep == 0]
    r = json.load(open('out/results_indiv_gap30.json'))['convergent']
    cmap = LinearSegmentedColormap.from_list('srt', ['#ffffff', '#d9e7ef', '#8fb9d1', '#4a8fb3', C['SRT'], '#0f3f5a'])
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.4))
    for ax, (x, xl, key, pan) in zip(axes, [('SRI', 'Sleep Regularity Index (SRI)', 'SRT~SRI', 'A'),
                                            ('SD_midpoint', 'SD of sleep midpoint (min)', 'SRT~SD_midpoint', 'B')]):
        hb = ax.hexbin(d[x], d.SRT, C=d.wt / 1000, reduce_C_function=np.sum, gridsize=38, cmap=cmap, mincnt=1, linewidths=0.15, edgecolors='white')
        # weighted linear fit
        w = d.wt.values; xv = d[x].values; yv = d.SRT.values
        b1 = np.cov(xv, yv, aweights=w)[0, 1] / np.cov(xv, xv, aweights=w)[0, 1]; b0 = np.average(yv, weights=w) - b1 * np.average(xv, weights=w)
        xs = np.linspace(np.percentile(xv, 0.5), np.percentile(xv, 99.5), 50)
        ax.plot(xs, b0 + b1 * xs, color=C['ink'], lw=1.4)
        rv = r[key]['r_w']
        ax.text(0.03, 0.96, f'weighted r = {rv:.2f}'.replace('-', '\u2212'), transform=ax.transAxes, ha='left', va='top', fontsize=9)
        ax.text(-0.14, 1.04, pan, transform=ax.transAxes, fontsize=12, fontweight='bold')
        ax.set_xlabel(xl); ax.set_ylabel('Sleep Regularity Test (SRT)'); ax.set_ylim(15, 100); despine(ax)
        ax.grid(True, color=C['grid'], lw=0.6); ax.set_axisbelow(True)
    axes[0].set_xlim(-30, 100); axes[1].set_xlim(0, 420)
    cb = fig.colorbar(hb, ax=axes, fraction=0.025, pad=0.02); cb.set_label('Weighted participants per bin (thousands)', fontsize=8); cb.outline.set_visible(False); cb.ax.tick_params(labelsize=7.5)
    save(fig, 'Fig3_scatter')

# ------------------------------------------------------------------ Fig. 4 forest
def fig_forest():
    R = {t: json.load(open(f'out/results_{t}.json')) for t in ['indiv_gap30', 'indiv_gap60', 'noon_gap30', 'noon_gap60']}
    def g(tag, model, term):
        c = R[tag]['models'][model]['coef'][term]; return c['OR'], c['lo'], c['hi']
    rows = [('Primary models', None, None),
            ('SRI, without SRT (Model 2)', g('indiv_gap30', 'M2', 'zSRI'), 'SRI'),
            ('SRI, with SRT (Model 3)', g('indiv_gap30', 'M3', 'zSRI'), 'SRI'),
            ('SRT, with SRI (Model 3)', g('indiv_gap30', 'M3', 'zSRT'), 'SRT'),
            ('SD-midpoint models', None, None),
            ('SRT, with SRI and SD midpoint', g('indiv_gap30', 'S1', 'zSRT'), 'SRT'),
            ('SD midpoint, with SRI and SRT', g('indiv_gap30', 'S1', 'zSD_midpoint'), 'SD'),
            ('SD midpoint, alone', g('indiv_gap30', 'S4', 'zSD_midpoint'), 'SD'),
            ('SD midpoint, with SRT', g('indiv_gap30', 'S5', 'zSD_midpoint'), 'SD'),
            ('SRT, with SD midpoint', g('indiv_gap30', 'S5', 'zSRT'), 'SRT'),
            ('Alternative extraction rules (SRT, Model 3)', None, None),
            ('Gap threshold 60 min', g('indiv_gap60', 'M3', 'zSRT'), 'SRT'),
            ('Fixed noon boundary, gap 30 min', g('noon_gap30', 'M3', 'zSRT'), 'SRT'),
            ('Fixed noon boundary, gap 60 min', g('noon_gap60', 'M3', 'zSRT'), 'SRT')]
    mk = {'SRT': ('o', C['SRT']), 'SRI': ('s', C['SRI']), 'SD': ('^', C['SD'])}
    n = len(rows); fig, ax = plt.subplots(figsize=(7.2, 0.33 * n + 0.9))
    ys = np.arange(n)[::-1]
    for y, (lab, est, kind) in zip(ys, rows):
        if est is None:
            ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8.6, fontweight='bold')
            continue
        if int(y) % 2 == 0: ax.axhspan(y - 0.5, y + 0.5, color=C['band'], lw=0, zorder=0)
        o, lo, hi = est; m, col = mk[kind]
        ax.plot([lo, hi], [y, y], color=col, lw=1.3, solid_capstyle='butt', zorder=2)
        ax.plot(o, y, marker=m, ms=6.5, mfc=col, mec='white', mew=0.8, zorder=3, ls='none')
        ax.text(-0.02, y, lab, transform=ax.get_yaxis_transform(), ha='right', va='center', fontsize=8.4)
        ax.text(1.02, y, f'{o:.2f} ({lo:.2f}–{hi:.2f})', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8.2)
    ax.axvline(1, color=C['muted'], lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.set_xscale('log'); ax.set_xlim(0.5, 1.6); ax.set_ylim(-0.6, n - 0.4)
    ticks = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.4]
    ax.xaxis.set_major_locator(FixedLocator(ticks)); ax.xaxis.set_major_formatter(FuncFormatter(lambda v, p: f'{v:g}'))
    ax.xaxis.set_minor_locator(FixedLocator([])); ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks([]); ax.spines['left'].set_visible(False); despine(ax)
    ax.set_xlabel('Odds ratio (95% CI) for depressive symptoms per 1 SD')
    ax.text(1.02, n - 0.2, 'OR (95% CI)', transform=ax.get_yaxis_transform(), ha='left', va='center', fontsize=8.4, fontweight='bold')
    # legend
    for i, (k, lab) in enumerate([('SRT', 'SRT'), ('SRI', 'SRI'), ('SD', 'SD midpoint')]):
        m, col = mk[k]
        ax.plot([], [], marker=m, color=col, ls='none', ms=6, mec='white', mew=0.8, label=lab)
    ax.legend(loc='lower left', frameon=False, fontsize=8, ncol=3, handletextpad=0.3, columnspacing=1.2, bbox_to_anchor=(0, -0.22))
    save(fig, 'Fig4_forest')

if __name__ == '__main__':
    fig_flow(); fig_scatter(); fig_forest()
