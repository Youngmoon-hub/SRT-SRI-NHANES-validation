"""
Convergent validity of the Sleep Regularity Test (SRT) with the
Sleep Regularity Index (SRI) using NHANES 2011-2014 wrist actigraphy.

Manuscript: Kwon (2026). Convergent Validity of the Sleep Regularity Test
            with the Sleep Regularity Index in a US National Sample.
            Sleep Health (under review).

Author: Youngmoon Kwon, Ph.D. (Inha University)
Email:  ym1980@gmail.com
ORCID:  0009-0005-5187-0481

==============================================================================
  WHAT THIS SCRIPT DOES
==============================================================================
  1. Loads minute-level NHANES PAM data (PAXPREDM, 2011-2014) from PhysioNet
  2. Identifies sleep episodes (>=60 minutes continuous sleep)
  3. Designates main vs additional sleep episodes per noon-to-noon window
  4. Computes:
        SRT = (M_bar / TSR) * 100, where
              M_bar = mean daily total sleep duration (minutes)
              TSR   = union of sleep minutes across the 7-day window
        SRI = Phillips et al. (2017) state-comparison formula
  5. Outputs nhanes_metrics.csv with one row per participant
  6. Prints all manuscript-relevant statistics

==============================================================================
  INPUT DATA
==============================================================================
  Required files (publicly available from PhysioNet):
    - subject-info.csv               (demographic info)
    - nhanes_1440_PAXPREDM.csv       (minute-level sleep/wake predictions)

  Source: https://physionet.org/content/minute-level-step-count-nhanes/1.0.1/
  Original data: NHANES 2011-2014, https://www.cdc.gov/nchs/nhanes/

==============================================================================
  REQUIREMENTS
==============================================================================
  Python 3.10+
  numpy >= 1.25
  pandas >= 2.0
  scipy >= 1.11

  Install:  pip install numpy pandas scipy

==============================================================================
  USAGE
==============================================================================
  1. Place subject-info.csv and nhanes_1440_PAXPREDM.csv in the same
     directory as this script (or update DATA_PATH and SUBJECT_PATH below).
  2. Run:  python nhanes_srt_sri_analysis.py
  3. Output: nhanes_metrics.csv (one row per participant) plus
     all manuscript statistics printed to console.

  Estimated runtime: ~1 minute on a typical laptop (16 GB RAM recommended).
"""

import numpy as np
import pandas as pd
from scipy import stats
import time
import gc
import sys
import os

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_PATH    = 'nhanes_1440_PAXPREDM.csv'   # adjust path if needed
SUBJECT_PATH = 'subject-info.csv'
OUT_CSV      = 'nhanes_metrics.csv'

MIN_EPISODE_DURATION = 60   # minutes; conservative threshold to exclude
                            # transient sleep-wake transitions
MIN_VALID_DAYS       = 3    # minimum number of valid wear days per person
MIN_VALID_MINUTES    = 960  # minimum valid (worn) minutes per day
MAX_DAY              = 7    # use only PAXDAYM 1-7 (NHANES PAM standard)

# PAXPREDM coding:
#   1 = sleep (worn)
#   2 = wake (worn)
#   3 = not worn
#   4 = missing
SLEEP_CODE  = 1
WAKE_CODE   = 2
NOTWORN     = 3
MISSING     = 4
INVALID_FILL = 4   # NaN treated as missing

# =============================================================================
# 1. LOAD SUBJECT INFO
# =============================================================================
def load_subjects(path):
    """Load demographic information."""
    print(f"[1/6] Loading subject info: {path}")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)
    subj = pd.read_csv(path)
    subj_dict = {
        int(r['SEQN']): {
            'sex': r['gender'],
            'age': int(r['age_in_years_at_screening']),
        }
        for _, r in subj.iterrows()
    }
    print(f"      {len(subj_dict)} subjects loaded")
    return subj_dict


# =============================================================================
# 2. LOAD MINUTE-LEVEL ACTIGRAPHY
# =============================================================================
def load_paxpredm(path, max_day=MAX_DAY):
    """
    Load minute-level PAXPREDM data, filter to days 1-max_day,
    and return arrays optimized for memory.
    """
    print(f"[2/6] Loading PAXPREDM: {path}")
    if not os.path.exists(path):
        print(f"ERROR: {path} not found", file=sys.stderr)
        sys.exit(1)

    # Read with optimized dtypes
    header = pd.read_csv(path, nrows=0)
    min_cols = [c for c in header.columns if c.startswith('min_')]
    dtype_map = {c: 'float32' for c in min_cols}
    dtype_map['SEQN']     = 'int32'
    dtype_map['PAXDAYM']  = 'int8'

    df = pd.read_csv(path, dtype=dtype_map,
                     usecols=['SEQN', 'PAXDAYM'] + min_cols)
    print(f"      raw shape: {df.shape}")

    # Filter to days 1-max_day (NHANES PAM standard 7-day window)
    df = df[df['PAXDAYM'] <= max_day]
    print(f"      after day 1-{max_day} filter: {df.shape}")

    # Convert to int8 to save memory
    df[min_cols] = df[min_cols].fillna(INVALID_FILL).astype(np.int8)
    gc.collect()

    M       = df[min_cols].values            # (n_rows, 1440)
    seqn    = df['SEQN'].values
    day     = df['PAXDAYM'].values
    del df
    gc.collect()
    print(f"      memory after load: ok")

    return M, seqn, day, min_cols


# =============================================================================
# 3. EPISODE EXTRACTION
# =============================================================================
def find_sleep_episodes(noon_to_noon_minutes, min_duration=MIN_EPISODE_DURATION):
    """
    Find continuous sleep runs >= min_duration in a 1440-minute
    noon-to-noon window. Returns list of (start_min, end_min, duration).
    """
    is_sleep = (noon_to_noon_minutes == SLEEP_CODE)
    episodes = []
    in_run, start = False, 0
    for i in range(len(is_sleep)):
        if is_sleep[i] and not in_run:
            in_run, start = True, i
        elif not is_sleep[i] and in_run:
            in_run = False
            duration = i - start
            if duration >= min_duration:
                episodes.append((start, i, duration))
    if in_run:
        duration = len(is_sleep) - start
        if duration >= min_duration:
            episodes.append((start, len(is_sleep), duration))
    return episodes


# =============================================================================
# 4. PER-PERSON COMPUTATION
# =============================================================================
def process_person(seqn, person_M, person_days, subj_dict):
    """
    Compute SRT and SRI for one participant.

    Args:
        seqn:        participant ID
        person_M:    array (n_days, 1440), midnight-to-midnight states
        person_days: array of PAXDAYM values, sorted
        subj_dict:   demographic dictionary

    Returns:
        dict with SRT, SRI, M_bar, TSR, demographics, etc.
        None if participant fails inclusion criteria.
    """
    info = subj_dict.get(int(seqn))
    if info is None or len(person_days) < 2:
        return None

    # Inclusion: each valid day must have >= MIN_VALID_MINUTES of worn time
    valid_per_day = np.array([
        np.isin(person_M[i], [SLEEP_CODE, WAKE_CODE]).sum()
        for i in range(len(person_days))
    ])
    valid_mask = valid_per_day >= MIN_VALID_MINUTES
    n_valid_days = int(valid_mask.sum())
    if n_valid_days < MIN_VALID_DAYS:
        return None

    # ── M_bar: mean daily total sleep duration ──────────────────────────────
    daily_total_sleep = []
    for d in range(len(person_days)):
        if not valid_mask[d]:
            continue
        daily_total_sleep.append((person_M[d] == SLEEP_CODE).sum())

    # ── TSR: union of sleep minutes across all valid days ───────────────────
    # "total minutes classified as sleep on at least one day"
    sleep_union = np.zeros(1440, dtype=bool)
    for d in range(len(person_days)):
        if not valid_mask[d]:
            continue
        sleep_union |= (person_M[d] == SLEEP_CODE)

    # ── Episodes (for inclusion >= MIN_VALID_DAYS main episodes) ────────────
    main_episode_durations = []
    additional_episodes_count = 0
    for k in range(len(person_days) - 1):
        if not (valid_mask[k] and valid_mask[k + 1]):
            continue
        # Noon-to-noon: 12:00 of day k to 12:00 of day k+1
        n2n = np.concatenate([person_M[k][720:], person_M[k + 1][:720]])
        eps = find_sleep_episodes(n2n)
        if not eps:
            continue
        eps_sorted = sorted(eps, key=lambda e: -e[2])
        main_episode_durations.append(eps_sorted[0][2])
        additional_episodes_count += len(eps_sorted) - 1

    if len(main_episode_durations) < MIN_VALID_DAYS:
        return None

    # ── SRT ─────────────────────────────────────────────────────────────────
    M_bar = float(np.mean(daily_total_sleep))
    TSR   = int(sleep_union.sum())
    SRT   = (M_bar / TSR) * 100.0 if TSR > 0 else np.nan

    # ── SRI (Phillips et al. 2017) ──────────────────────────────────────────
    # SRI = (1/((N-1)*1440)) * sum_d sum_t delta[s(d,t) == s(d+1,t)] * 200 - 100
    # Computed only on minutes where both consecutive days are valid.
    sri_match = 0
    sri_total = 0
    for d in range(len(person_days) - 1):
        if not (valid_mask[d] and valid_mask[d + 1]):
            continue
        s1, s2 = person_M[d], person_M[d + 1]
        both_valid = (np.isin(s1, [SLEEP_CODE, WAKE_CODE])
                      & np.isin(s2, [SLEEP_CODE, WAKE_CODE]))
        if both_valid.sum() < 60:
            continue
        sri_match += int((s1[both_valid] == s2[both_valid]).sum())
        sri_total += int(both_valid.sum())
    SRI = (sri_match / sri_total) * 200.0 - 100.0 if sri_total > 0 else np.nan

    return {
        'SEQN':           int(seqn),
        'age':            info['age'],
        'sex':            info['sex'],
        'n_valid_days':   n_valid_days,
        'n_main_eps':     len(main_episode_durations),
        'n_additional_eps': additional_episodes_count,
        'M_bar':          M_bar,
        'TSR':            TSR,
        'SRT':            SRT if not np.isnan(SRT) else None,
        'SRI':            SRI if not np.isnan(SRI) else None,
    }


# =============================================================================
# 5. RUN ANALYSIS
# =============================================================================
def run_analysis():
    t0 = time.time()

    subj_dict = load_subjects(SUBJECT_PATH)
    M, seqn_arr, day_arr, _ = load_paxpredm(DATA_PATH)

    print(f"[3/6] Processing per person...")
    unique_seqns = np.unique(seqn_arr)
    print(f"      {len(unique_seqns)} unique participants")

    results = []
    n_skipped = 0
    for i, sq in enumerate(unique_seqns):
        if i % 2000 == 0 and i > 0:
            print(f"      {i}/{len(unique_seqns)} processed "
                  f"({time.time() - t0:.0f}s elapsed)")
        mask  = (seqn_arr == sq)
        pdays = day_arr[mask]
        pM    = M[mask]
        sort_idx = np.argsort(pdays)
        res = process_person(sq, pM[sort_idx], pdays[sort_idx], subj_dict)
        if res is None:
            n_skipped += 1
        else:
            results.append(res)

    print(f"[4/6] Processed {len(results)} valid; {n_skipped} excluded")

    # ── Save ────────────────────────────────────────────────────────────────
    out = pd.DataFrame(results)
    out.to_csv(OUT_CSV, index=False)
    print(f"[5/6] Saved: {OUT_CSV}")

    # ── Manuscript-relevant statistics ──────────────────────────────────────
    print_manuscript_stats(out)

    print(f"\nTotal runtime: {time.time() - t0:.0f} seconds")
    return out


# =============================================================================
# 6. MANUSCRIPT-RELEVANT STATISTICS
# =============================================================================
def print_manuscript_stats(df):
    """Print all statistics that appear in the manuscript."""
    print(f"\n[6/6] MANUSCRIPT-RELEVANT STATISTICS")
    print("=" * 70)

    print(f"\n--- Full sample (age >= 6) ---")
    print(f"N = {len(df)}")
    print(f"  Age: M = {df['age'].mean():.1f}, SD = {df['age'].std():.1f}")
    print(f"  Female: {(df['sex']=='Female').mean()*100:.1f}%")
    print(f"  Mean valid wear days: {df['n_valid_days'].mean():.2f} "
          f"(SD = {df['n_valid_days'].std():.2f})")

    adults = df[df['age'] >= 20].dropna(subset=['SRI', 'SRT']).copy()
    print(f"\n--- Adult sample (age >= 20) ---")
    print(f"N = {len(adults)}")
    print(f"  Age: M = {adults['age'].mean():.1f}, "
          f"SD = {adults['age'].std():.1f}")
    print(f"  Female: {(adults['sex']=='Female').mean()*100:.1f}%")
    print(f"  Mean M_bar: {adults['M_bar'].mean():.1f} min")
    print(f"  Mean TSR:   {adults['TSR'].mean():.1f} min")

    print(f"\n--- Table 1: Descriptives (Adults) ---")
    for m in ['SRI', 'SRT']:
        v = adults[m].dropna()
        print(f"  {m}: M = {v.mean():.2f}, SD = {v.std():.2f}, "
              f"Med = {v.median():.2f}, "
              f"range [{v.min():.1f}, {v.max():.1f}]")

    print(f"\n--- Convergent validity (Adults) ---")
    r_p, p_p = stats.pearsonr(adults['SRI'], adults['SRT'])
    r_s, p_s = stats.spearmanr(adults['SRI'], adults['SRT'])
    n = len(adults)
    z_fish = 0.5 * np.log((1 + r_p) / (1 - r_p))
    se = 1 / np.sqrt(n - 3)
    ci_lo = float(np.tanh(z_fish - 1.96 * se))
    ci_hi = float(np.tanh(z_fish + 1.96 * se))
    slope, intercept, _, _, _ = stats.linregress(adults['SRI'], adults['SRT'])
    print(f"  Pearson r  = {r_p:.4f}, 95% CI [{ci_lo:.4f}, {ci_hi:.4f}], "
          f"p = {p_p:.2e}")
    print(f"  Spearman ρ = {r_s:.4f}, p = {p_s:.2e}")
    print(f"  R² = {r_p**2 * 100:.2f}%")
    print(f"  OLS slope = {slope:.4f}, intercept = {intercept:.4f}")

    print(f"\n--- Table 3: Sex stratification (Adults) ---")
    male   = adults[adults['sex'] == 'Male']
    female = adults[adults['sex'] == 'Female']
    for sex_label, s in [('Male', male), ('Female', female)]:
        rs, _ = stats.pearsonr(s['SRI'], s['SRT'])
        print(f"  {sex_label}: n = {len(s)}, "
              f"SRI = {s['SRI'].mean():.2f} ± {s['SRI'].std():.2f}, "
              f"SRT = {s['SRT'].mean():.2f} ± {s['SRT'].std():.2f}, "
              f"r = {rs:.4f}")

    t_sri, p_sri = stats.ttest_ind(female['SRI'], male['SRI'])
    t_srt, p_srt = stats.ttest_ind(female['SRT'], male['SRT'])
    sp_sri = np.sqrt(((len(male) - 1) * male['SRI'].std()**2
                      + (len(female) - 1) * female['SRI'].std()**2)
                     / (len(male) + len(female) - 2))
    sp_srt = np.sqrt(((len(male) - 1) * male['SRT'].std()**2
                      + (len(female) - 1) * female['SRT'].std()**2)
                     / (len(male) + len(female) - 2))
    d_sri = (female['SRI'].mean() - male['SRI'].mean()) / sp_sri
    d_srt = (female['SRT'].mean() - male['SRT'].mean()) / sp_srt
    print(f"  Sex t-test SRI: t = {t_sri:.3f}, p = {p_sri:.2e}, "
          f"d = {d_sri:.4f}")
    print(f"  Sex t-test SRT: t = {t_srt:.3f}, p = {p_srt:.2e}, "
          f"d = {d_srt:.4f}")

    # Fisher z for sex difference in r
    r_m, _ = stats.pearsonr(male['SRI'], male['SRT'])
    r_f, _ = stats.pearsonr(female['SRI'], female['SRT'])
    z_m  = 0.5 * np.log((1 + r_m) / (1 - r_m))
    z_f  = 0.5 * np.log((1 + r_f) / (1 - r_f))
    se_d = np.sqrt(1 / (len(male) - 3) + 1 / (len(female) - 3))
    z_t  = (z_m - z_f) / se_d
    p_z  = 2 * (1 - stats.norm.cdf(abs(z_t)))
    print(f"  Fisher z (sex diff in r): z = {z_t:.4f}, p = {p_z:.4f}")

    print(f"\n--- Table 2: Age groups (Full sample) ---")
    groups = [(6, 12, '6-11'),   (12, 18, '12-17'),
              (18, 30, '18-29'), (30, 40, '30-39'),
              (40, 50, '40-49'), (50, 60, '50-59'),
              (60, 70, '60-69'), (70, 90, '>=70')]
    print(f"  {'Age':>6}  {'n':>5}  "
          f"{'SRI (M ± SD)':>16}  {'SRT (M ± SD)':>16}  "
          f"{'Gap':>6}  {'r':>8}  {'paired t':>10}")
    for lo, hi, lab in groups:
        s = df[(df['age'] >= lo) & (df['age'] < hi)].dropna(
            subset=['SRI', 'SRT'])
        if len(s) < 10:
            continue
        r2, _ = stats.pearsonr(s['SRI'], s['SRT'])
        gap   = s['SRT'].mean() - s['SRI'].mean()
        t_, _ = stats.ttest_rel(s['SRT'], s['SRI'])
        print(f"  {lab:>6}  {len(s):>5}  "
              f"{s['SRI'].mean():>5.1f} ± {s['SRI'].std():>5.1f}  "
              f"{s['SRT'].mean():>5.1f} ± {s['SRT'].std():>5.1f}  "
              f"{gap:>+5.1f}  {r2:>8.4f}  {t_:>10.2f}")

    print("=" * 70)


# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    run_analysis()
