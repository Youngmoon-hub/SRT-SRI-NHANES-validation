"""
Generate publication-quality figures for the SRT-SRI convergent validity study.

Manuscript: Kwon (2026). Convergent Validity of the Sleep Regularity Test
            with the Sleep Regularity Index in a US National Sample.
            Sleep Health (under review).

Author: Youngmoon Kwon, Ph.D. (Inha University)
Email:  ym1980@gmail.com

==============================================================================
  USAGE
==============================================================================
  Run this script after nhanes_srt_sri_analysis.py has produced
  nhanes_metrics.csv. The script generates 4 figures in both
  TIFF (600 DPI, journal submission) and PNG (300 DPI, preview) formats.

  python nhanes_figures.py

==============================================================================
  REQUIREMENTS
==============================================================================
  Python 3.10+
  numpy, pandas, scipy, matplotlib

  Install:  pip install numpy pandas scipy matplotlib
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.patches import Patch
from scipy import stats
from scipy.stats import gaussian_kde


# =============================================================================
# CONFIGURATION
# =============================================================================
INPUT_CSV = 'nhanes_metrics.csv'
OUT_DIR   = 'figures'
os.makedirs(OUT_DIR, exist_ok=True)

# Figure styling: Sleep Health journal aesthetic
rcParams['font.family']         = 'DejaVu Sans'
rcParams['font.size']           = 9
rcParams['axes.linewidth']      = 0.8
rcParams['axes.spines.top']     = False
rcParams['axes.spines.right']   = False
rcParams['xtick.major.width']   = 0.8
rcParams['ytick.major.width']   = 0.8
rcParams['xtick.direction']     = 'out'
rcParams['ytick.direction']     = 'out'
rcParams['savefig.dpi']         = 600
rcParams['savefig.bbox']        = 'tight'

# Color palette
COL_SRT     = '#D85A4B'   # warm red-orange
COL_SRI     = '#3D6B9B'   # deep blue
COL_MALE    = '#4A7BA6'
COL_FEMALE  = '#C2734D'
COL_NEUTRAL = '#444444'


# =============================================================================
# DATA
# =============================================================================
df = pd.read_csv(INPUT_CSV)
adults = df[df['age'] >= 20].dropna(subset=['SRI', 'SRT']).copy()


# =============================================================================
# FIGURE 1: Computation principles (illustrative diagram)
# =============================================================================
def figure_1():
    fig, axes = plt.subplots(2, 1, figsize=(7.5, 6.5))

    # ── Panel A: SRT illustration ─────────────────────────────────────────
    ax = axes[0]
    ax.set_title('A. Sleep Regularity Test (SRT)',
                 loc='left', fontweight='bold', fontsize=10, pad=24)

    # Generate 7 hypothetical days
    np.random.seed(42)
    day_data = []
    for d in range(7):
        onset_h  = 11 + np.random.uniform(-0.8, 1.2)
        offset_h = onset_h + 7.5 + np.random.uniform(-1, 1)
        nap_present = (d in [2, 5])
        if nap_present:
            nap_start = 2 + np.random.uniform(-0.5, 0.5)
            nap_end   = nap_start + 1
        else:
            nap_start, nap_end = None, None
        day_data.append((onset_h, offset_h, nap_start, nap_end))

    # Draw days
    for d, (on, off, ns, ne) in enumerate(day_data):
        y = 6 - d
        ax.axhspan(y - 0.35, y + 0.35, xmin=0, xmax=1,
                   color='#F5F5F5', alpha=0.4)
        ax.barh(y, off - on, left=on, height=0.5, color=COL_SRT,
                edgecolor='white', linewidth=0.5, zorder=3)
        if ns is not None:
            ax.barh(y, ne - ns, left=ns, height=0.5, color='#E8A574',
                    edgecolor='white', linewidth=0.5, zorder=3)
        ax.text(-0.5, y, f'Day {d+1}', ha='right', va='center', fontsize=8.5)

    # TSR union envelope
    union_start = min(d[0] for d in day_data) - 0.1
    union_end   = max(d[1] for d in day_data) + 0.1
    nap_starts = [d[2] for d in day_data if d[2] is not None]
    nap_ends   = [d[3] for d in day_data if d[3] is not None]

    y_union = -0.7
    ax.barh(y_union, union_end - union_start, left=union_start, height=0.32,
            color='#FFD89A', edgecolor=COL_NEUTRAL, linewidth=1, zorder=3)
    if nap_starts:
        ax.barh(y_union, max(nap_ends) - min(nap_starts),
                left=min(nap_starts), height=0.32,
                color='#FFD89A', edgecolor=COL_NEUTRAL, linewidth=1, zorder=3)
    ax.text(-0.5, y_union, 'TSR (union)', ha='right', va='center',
            fontsize=8.5, fontweight='bold')

    # Time axis
    hours_label = ['12:00', '15:00', '18:00', '21:00', '00:00',
                   '03:00', '06:00', '09:00', '12:00']
    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels(hours_label, fontsize=8)
    ax.set_xlim(-2.5, 25)
    ax.set_ylim(-1.4, 6.6)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time of day (noon-to-noon window)', fontsize=9)

    legend_elements = [
        Patch(facecolor=COL_SRT, label='Main sleep episode'),
        Patch(facecolor='#E8A574', label='Additional sleep episode'),
        Patch(facecolor='#FFD89A', edgecolor=COL_NEUTRAL,
              linewidth=0.8, label='TSR (union of sleep minutes)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=7.5,
              frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.20))

    ax.text(12.5, -1.25, r'SRT = (M̄ / TSR) × 100',
            ha='center', va='center', fontsize=9.5, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#FFF8E0',
                      edgecolor='#888', linewidth=0.6))

    # ── Panel B: SRI illustration ─────────────────────────────────────────
    ax = axes[1]
    ax.set_title('B. Sleep Regularity Index (SRI)',
                 loc='left', fontweight='bold', fontsize=10, pad=24)

    # 2 consecutive days
    day1_states, day2_states = [], []
    for h in np.arange(0, 24, 0.5):
        if h >= 23 or h < 7:
            s1 = 1
        elif 14 <= h < 15:
            s1 = 1
        else:
            s1 = 0
        if h >= 23.5 or h < 7.5:
            s2 = 1
        else:
            s2 = 0
        day1_states.append(s1)
        day2_states.append(s2)

    x = np.arange(0, 24, 0.5)
    day1_arr = np.array(day1_states)
    day2_arr = np.array(day2_states)

    y1 = 1.5
    for xi, s in zip(x, day1_arr):
        color = COL_SRI if s == 1 else '#E8E8E8'
        ax.barh(y1, 0.5, left=xi, height=0.55, color=color,
                edgecolor='white', linewidth=0.2)
    ax.text(-0.5, y1, 'Day d', ha='right', va='center', fontsize=8.5)

    y2 = 0.5
    for xi, s in zip(x, day2_arr):
        color = COL_SRI if s == 1 else '#E8E8E8'
        ax.barh(y2, 0.5, left=xi, height=0.55, color=color,
                edgecolor='white', linewidth=0.2)
    ax.text(-0.5, y2, 'Day d+1', ha='right', va='center', fontsize=8.5)

    y_match = -0.6
    for xi, s1, s2 in zip(x, day1_arr, day2_arr):
        if s1 == s2 == 1:
            color = '#2D5F8F'
        elif s1 == s2 == 0:
            color = '#A8C8E0'
        else:
            color = '#E89C7B'
        ax.barh(y_match, 0.5, left=xi, height=0.32, color=color,
                edgecolor='white', linewidth=0.2)
    ax.text(-0.5, y_match, 'State match', ha='right', va='center',
            fontsize=8.5, fontweight='bold')

    ax.set_xticks(np.arange(0, 25, 3))
    ax.set_xticklabels(['00:00', '03:00', '06:00', '09:00', '12:00',
                        '15:00', '18:00', '21:00', '24:00'], fontsize=8)
    ax.set_xlim(-2.5, 25)
    ax.set_ylim(-1.2, 2.4)
    ax.set_yticks([])
    ax.spines['left'].set_visible(False)
    ax.set_xlabel('Time of day (24-hour clock)', fontsize=9)

    legend_elements2 = [
        Patch(facecolor='#2D5F8F', label='Both sleep'),
        Patch(facecolor='#A8C8E0', label='Both wake'),
        Patch(facecolor='#E89C7B', label='Mismatch'),
    ]
    ax.legend(handles=legend_elements2, loc='upper right', fontsize=7.5,
              frameon=False, ncol=3, bbox_to_anchor=(1.0, 1.20))

    ax.text(12, -1.05, r'SRI = [match% × 2 − 100]',
            ha='center', va='center', fontsize=9.5, style='italic',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#E8F0F7',
                      edgecolor='#888', linewidth=0.6))

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Figure1.tiff', dpi=600, format='tiff')
    plt.savefig(f'{OUT_DIR}/Figure1.png',  dpi=300)
    plt.close()
    print('Figure 1 saved')


# =============================================================================
# FIGURE 2: SRT vs SRI hexbin scatter
# =============================================================================
def figure_2():
    fig, ax = plt.subplots(figsize=(5.2, 4.6))

    hb = ax.hexbin(adults['SRI'], adults['SRT'], gridsize=45, cmap='Blues',
                   mincnt=1, edgecolors='none')

    mn = min(adults['SRI'].min(), adults['SRT'].min())
    mx = max(adults['SRI'].max(), adults['SRT'].max())
    ax.plot([mn, mx], [mn, mx], color='#888', linestyle='--',
            linewidth=1.2, label='Identity (SRT = SRI)', zorder=3)

    slope, intercept, _, _, _ = stats.linregress(adults['SRI'], adults['SRT'])
    xr = np.array([adults['SRI'].min(), adults['SRI'].max()])
    yr = slope * xr + intercept
    ax.plot(xr, yr, color=COL_SRT, linewidth=2.0,
            label='OLS regression', zorder=4)

    r_p, _ = stats.pearsonr(adults['SRI'], adults['SRT'])
    n = len(adults)
    z = 0.5 * np.log((1 + r_p) / (1 - r_p))
    se = 1 / np.sqrt(n - 3)
    ci_lo = float(np.tanh(z - 1.96 * se))
    ci_hi = float(np.tanh(z + 1.96 * se))

    ax.text(0.05, 0.95,
            f'r = {r_p:.3f}\n95% CI [{ci_lo:.3f}, {ci_hi:.3f}]\n'
            f'R² = {r_p**2 * 100:.1f}%\nn = {n:,}',
            transform=ax.transAxes, ha='left', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                      edgecolor='#AAA', linewidth=0.6))

    cb = plt.colorbar(hb, ax=ax, label='Number of observations',
                      fraction=0.04, pad=0.02)
    cb.outline.set_linewidth(0.5)

    ax.set_xlabel('Sleep Regularity Index (SRI)', fontsize=10)
    ax.set_ylabel('Sleep Regularity Test (SRT)', fontsize=10)
    ax.set_xlim(adults['SRI'].min() - 3, adults['SRI'].max() + 3)
    ax.set_ylim(adults['SRT'].min() - 2, adults['SRT'].max() + 2)
    ax.legend(loc='lower right', fontsize=8, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Figure2.tiff', dpi=600, format='tiff')
    plt.savefig(f'{OUT_DIR}/Figure2.png',  dpi=300)
    plt.close()
    print('Figure 2 saved')


# =============================================================================
# FIGURE 3: Age group patterns
# =============================================================================
def figure_3():
    fig, axes = plt.subplots(2, 1, figsize=(6.0, 6.0),
                             gridspec_kw={'hspace': 0.30})

    groups = [(6, 12, '6–11'),   (12, 18, '12–17'),
              (18, 30, '18–29'), (30, 40, '30–39'),
              (40, 50, '40–49'), (50, 60, '50–59'),
              (60, 70, '60–69'), (70, 90, '≥70')]

    age_data = []
    for lo, hi, lab in groups:
        s = df[(df['age'] >= lo) & (df['age'] < hi)].dropna(
            subset=['SRI', 'SRT'])
        if len(s) < 10:
            continue
        r2, _ = stats.pearsonr(s['SRI'], s['SRT'])
        age_data.append({
            'label':  lab,
            'n':      len(s),
            'sri_m':  s['SRI'].mean(),
            'sri_se': s['SRI'].std() / np.sqrt(len(s)),
            'srt_m':  s['SRT'].mean(),
            'srt_se': s['SRT'].std() / np.sqrt(len(s)),
            'r':      r2,
        })

    x = np.arange(len(age_data))
    labels = [d['label'] for d in age_data]

    # ── Top panel: means with gap shading ────────────────────────────────
    ax = axes[0]
    sri_means = [d['sri_m']  for d in age_data]
    sri_se    = [d['sri_se'] for d in age_data]
    srt_means = [d['srt_m']  for d in age_data]
    srt_se    = [d['srt_se'] for d in age_data]

    ax.fill_between(x, sri_means, srt_means, color='#FFE0DC',
                    alpha=0.7, zorder=1, label='SRT − SRI gap')
    ax.errorbar(x, sri_means, yerr=sri_se, fmt='s', color=COL_SRI,
                markersize=7, linewidth=1.5, capsize=3, label='SRI',
                zorder=3, markeredgecolor='white', markeredgewidth=0.5)
    ax.errorbar(x, srt_means, yerr=srt_se, fmt='o', color=COL_SRT,
                markersize=7, linewidth=1.5, capsize=3, label='SRT',
                zorder=3, markeredgecolor='white', markeredgewidth=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Age group (years)', fontsize=9.5)
    ax.set_ylabel('Score (mean ± SE)', fontsize=9.5)
    ax.legend(loc='lower left', fontsize=8.5, frameon=False)
    ax.set_ylim(50, 82)
    ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

    # ── Bottom panel: r values ───────────────────────────────────────────
    ax = axes[1]
    rs = [d['r'] for d in age_data]
    ax.plot(x, rs, 'o-', color='#5B7C99', markersize=8, linewidth=1.8,
            markeredgecolor='white', markeredgewidth=0.6, zorder=3)
    ax.axhline(y=0.7, color='#AAA', linestyle=':', linewidth=0.8, zorder=1)
    ax.axhline(y=0.8, color='#AAA', linestyle=':', linewidth=0.8, zorder=1)
    ax.text(7.3, 0.803, 'r = 0.80',
            fontsize=7.5, color='#888', ha='right', va='bottom')

    for xi, r2 in zip(x, rs):
        ax.annotate(f'{r2:.3f}', (xi, r2), textcoords='offset points',
                    xytext=(0, 9), ha='center', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_xlabel('Age group (years)', fontsize=9.5)
    ax.set_ylabel('Pearson r (SRT vs SRI)', fontsize=9.5)
    ax.set_ylim(0.78, 0.89)
    ax.grid(axis='y', linestyle=':', alpha=0.3, zorder=0)

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Figure3.tiff', dpi=600, format='tiff')
    plt.savefig(f'{OUT_DIR}/Figure3.png',  dpi=300)
    plt.close()
    print('Figure 3 saved')


# =============================================================================
# FIGURE 4: Distributions by sex
# =============================================================================
def figure_4():
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))

    male   = adults[adults['sex'] == 'Male']
    female = adults[adults['sex'] == 'Female']

    for ax, metric, title in [(axes[0], 'SRI', 'SRI'),
                              (axes[1], 'SRT', 'SRT')]:
        bins = np.linspace(adults[metric].min(),
                           adults[metric].max(), 45)
        ax.hist(male[metric],   bins=bins, alpha=0.55, color=COL_MALE,
                label=f'Male (n = {len(male):,})',
                edgecolor='white', linewidth=0.3)
        ax.hist(female[metric], bins=bins, alpha=0.55, color=COL_FEMALE,
                label=f'Female (n = {len(female):,})',
                edgecolor='white', linewidth=0.3)

        xx = np.linspace(adults[metric].min(),
                         adults[metric].max(), 200)
        ax2 = ax.twinx()
        ax2.plot(xx, gaussian_kde(male[metric])(xx),
                 color=COL_MALE,   linewidth=1.8)
        ax2.plot(xx, gaussian_kde(female[metric])(xx),
                 color=COL_FEMALE, linewidth=1.8)
        ax2.set_yticks([])
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        ax2.spines['left'].set_visible(False)

        t_, _ = stats.ttest_ind(female[metric], male[metric])
        sp = np.sqrt(((len(male) - 1)   * male[metric].std()**2
                      + (len(female) - 1) * female[metric].std()**2)
                     / (len(male) + len(female) - 2))
        d_ = (female[metric].mean() - male[metric].mean()) / sp
        ax.text(0.03, 0.95, f't = {t_:.2f}, d = {d_:.3f}',
                transform=ax.transAxes, fontsize=9, va='top',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                          edgecolor='#AAA', linewidth=0.5))

        ax.set_xlabel(metric, fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title(title, fontsize=10, fontweight='bold', loc='left')
        ax.legend(loc='upper left', fontsize=8.5, frameon=False,
                  bbox_to_anchor=(0.02, 0.85))

    plt.tight_layout()
    plt.savefig(f'{OUT_DIR}/Figure4.tiff', dpi=600, format='tiff')
    plt.savefig(f'{OUT_DIR}/Figure4.png',  dpi=300)
    plt.close()
    print('Figure 4 saved')


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    figure_1()
    figure_2()
    figure_3()
    figure_4()
    print(f'\nAll figures saved to {OUT_DIR}/')
