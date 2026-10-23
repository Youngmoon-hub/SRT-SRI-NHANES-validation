"""
SRT / SRI / SD-midpoint computation for the SBR manuscript.

Definitions (per manuscript draft, 2026-09-26):
  - Day = noon-to-noon window (12:00 day k -> 12:00 day k+1)
  - Valid window: >= MIN_VALID_MINUTES worn (wake or sleep) minutes
  - Sample: participants with 7 CONSECUTIVE valid noon-to-noon windows
    (first such run is used; windows built from PAXDAYM 1..9)
  - Sleep episode: >= 60 consecutive minutes coded Sleep wear (PAXPREDM == 2)
    Main sleep and naps are NOT distinguished: all episodes count.
  - M   = 7-day mean of the daily sum of episode minutes
  - TSR = number of clock minutes (0..1439, noon-based) occupied by at least
          one episode across the 7 windows (union)
  - SRT = M / TSR * 100
  - SRI = Phillips et al. (2017), same 7 windows, worn minutes only
  - SD midpoint = SD (minutes) of main-episode (longest per window) midpoint
          across the 7 windows; midpoints measured as minutes since noon,
          so midnight crossing is not an issue.

PAXPREDM coding (NHANES documentation, PAXMIN_G):
  1 = Wake wear, 2 = Sleep wear, 3 = Non-wear, 4 = Unknown
"""
import numpy as np, pandas as pd, time, sys, os

DATA   = 'data/nhanes_1440_PAXPREDM.csv'
SUBJ   = 'data/subject-info.csv'
OUT    = os.environ.get('OUT', 'out/metrics_7day.csv')

WAKE, SLEEP, NONWEAR, UNKNOWN = 1, 2, 3, 4
MIN_EP   = 60
GAP_MERGE = int(os.environ.get('GAP_MERGE', 60))   # minutes
MIN_VALID_MINUTES = 960     # per noon-to-noon window
N_WIN    = 7

def episodes(x, min_dur=MIN_EP, gap=None):
    """Sleep episodes as (start, end_exclusive, span).
    Sleep bouts (runs of SLEEP) separated by gaps (wake/unknown/non-wear) shorter
    than GAP_MERGE minutes are merged into one episode (cf. HDCZA, van Hees 2018).
    Episode span = first to last sleep minute of the merged block; spans < min_dur
    are discarded."""
    if gap is None: gap = GAP_MERGE
    s = (x == SLEEP).astype(np.int8)
    d = np.diff(np.concatenate([[0], s, [0]]))
    starts = np.where(d == 1)[0]; ends = np.where(d == -1)[0]
    if len(starts) == 0: return []
    merged = [[starts[0], ends[0]]]
    for a, b in zip(starts[1:], ends[1:]):
        if a - merged[-1][1] < gap: merged[-1][1] = b
        else: merged.append([a, b])
    return [(a, b, b - a) for a, b in merged if b - a >= min_dur]

def process(pM, pdays):
    # pM: (n_days,1440) midnight-based rows sorted by PAXDAYM (1..9, possibly gaps)
    day_index = {d: i for i, d in enumerate(pdays)}
    wins = []   # list of (dayk, window_vector)
    for k in range(1, 9):
        if k in day_index and k + 1 in day_index:
            v = np.concatenate([pM[day_index[k]][720:], pM[day_index[k + 1]][:720]])
            worn = np.isin(v, [WAKE, SLEEP]).sum()
            wins.append((k, v, worn >= MIN_VALID_MINUTES))
        else:
            wins.append((k, None, False))
    # find first run of 7 consecutive valid windows
    valid = [w[2] for w in wins]
    start = None
    for i in range(len(valid) - N_WIN + 1):
        if all(valid[i:i + N_WIN]):
            start = i; break
    if start is None:
        return None
    W = [wins[i][1] for i in range(start, start + N_WIN)]

    daily = []; union = np.zeros(1440, bool); mids = []; n_eps = 0; n_nap = 0
    for v in W:
        eps = episodes(v)
        daily.append(sum(e[2] for e in eps))
        for a, b, _ in eps:
            union[a:b] = True
        n_eps += len(eps)
        if eps:
            main = max(eps, key=lambda e: e[2])
            mids.append((main[0] + main[1]) / 2.0)
            n_nap += len(eps) - 1
    M = float(np.mean(daily)); TSR = int(union.sum())
    SRT = M / TSR * 100 if TSR > 0 else np.nan
    sd_mid = float(np.std(mids, ddof=1)) if len(mids) >= 2 else np.nan
    mean_mid = float(np.mean(mids)) if mids else np.nan

    match = tot = 0
    for i in range(N_WIN - 1):
        a, b = W[i], W[i + 1]
        bv = np.isin(a, [WAKE, SLEEP]) & np.isin(b, [WAKE, SLEEP])
        match += int((a[bv] == b[bv]).sum()); tot += int(bv.sum())
    SRI = match / tot * 200 - 100 if tot > 0 else np.nan

    # Raw sleep-minute versions for reference
    raw_daily = [(v == SLEEP).sum() for v in W]
    return dict(first_window_day=wins[start][0], M=M, TSR=TSR, SRT=SRT, SRI=SRI,
                SD_midpoint=sd_mid, mean_midpoint_min_since_noon=mean_mid,
                n_episodes=n_eps, n_naps=n_nap, n_nights_with_episode=len(mids),
                raw_sleep_min_per_day=float(np.mean(raw_daily)))

def main():
    t0 = time.time()
    subj = pd.read_csv(SUBJ).set_index('SEQN')
    hdr = pd.read_csv(DATA, nrows=0); mc = [c for c in hdr.columns if c.startswith('min_')]
    dt = {c: 'float32' for c in mc}; dt['SEQN'] = 'int32'; dt['PAXDAYM'] = 'int8'
    df = pd.read_csv(DATA, usecols=['SEQN', 'PAXDAYM'] + mc, dtype=dt)
    print('rows', df.shape, f'{time.time()-t0:.0f}s')
    M = df[mc].fillna(UNKNOWN).to_numpy(dtype=np.int8)
    seqn = df['SEQN'].to_numpy(); day = df['PAXDAYM'].to_numpy()
    del df
    order = np.lexsort((day, seqn)); M, seqn, day = M[order], seqn[order], day[order]
    bounds = np.flatnonzero(np.diff(seqn)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(seqn)]])
    rows = []; n_total = len(starts)
    for i, (a, b) in enumerate(zip(starts, ends)):
        sq = int(seqn[a])
        r = process(M[a:b], list(day[a:b]))
        if r is None: continue
        r['SEQN'] = sq
        if sq in subj.index:
            r['age'] = int(subj.at[sq, 'age_in_years_at_screening']); r['sex'] = subj.at[sq, 'gender']
        rows.append(r)
        if i % 3000 == 0: print(i, '/', n_total, f'{time.time()-t0:.0f}s')
    out = pd.DataFrame(rows)
    os.makedirs('out', exist_ok=True); out.to_csv(OUT, index=False)
    print('participants with PAM data:', n_total, ' with 7 consecutive valid windows:', len(out))
    ad = out[out.age >= 20]
    print('adults >=20:', len(ad))
    print(ad[['M', 'TSR', 'SRT', 'SRI', 'SD_midpoint', 'n_naps', 'raw_sleep_min_per_day']].describe().round(2))
    print('r(SRT,SRI)=', ad[['SRT', 'SRI']].corr().iloc[0, 1].round(3),
          ' r(SRT,SDmid)=', ad[['SRT', 'SD_midpoint']].corr().iloc[0, 1].round(3),
          ' r(SRI,SDmid)=', ad[['SRI', 'SD_midpoint']].corr().iloc[0, 1].round(3))
    print(f'done {time.time()-t0:.0f}s')

if __name__ == '__main__':
    main()
