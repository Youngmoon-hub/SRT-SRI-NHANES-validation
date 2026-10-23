"""
SRT / SRI / SD-midpoint with INDIVIDUALISED day boundaries (main-sleep based days).

Differences from 01_compute_metrics.py (noon-to-noon):
  - Episodes are extracted on the participant's CONTINUOUS timeline (PAXDAYM 1..9
    concatenated), so no episode is cut at a clock boundary.
  - Day boundary = the clock minute at which the participant is LEAST often asleep
    across the whole recording (smoothed over +-60 min). Night sleepers get a
    daytime boundary (~noon), day sleepers a night-time boundary. Each day then
    contains one main sleep: "7 days" = 7 consecutive main sleeps, with the
    recording extending beyond both ends (NHANES provides up to 9 days).
  - Episode = sleep bouts (PAXPREDM == 2) merged when the gap < 30 min;
    span (onset..offset) >= 60 min; naps included. Assigned to the day that
    contains its midpoint.
  - Valid day: >= 960 worn minutes, >= 1 episode. First run of 7 consecutive
    valid days is used.
  - M, TSR, SRT, SRI, SD midpoint as before, computed on those 7 days.
"""
import numpy as np, pandas as pd, time, os, sys

DATA = 'data/nhanes_1440_PAXPREDM.csv'; SUBJ = 'data/subject-info.csv'
OUT  = os.environ.get('OUT', 'out/metrics_7day_indiv_gap30.csv')
WAKE, SLEEP = 1, 2
MIN_EP = 60; GAP = int(os.environ.get('GAP_MERGE', 30)); MIN_VALID = 960; N_WIN = 7

def episodes_cont(x):
    s = (x == SLEEP).astype(np.int8)
    d = np.diff(np.concatenate([[0], s, [0]]))
    st = np.where(d == 1)[0]; en = np.where(d == -1)[0]
    if len(st) == 0: return []
    m = [[st[0], en[0]]]
    for a, b in zip(st[1:], en[1:]):
        if a - m[-1][1] < GAP: m[-1][1] = b
        else: m.append([a, b])
    return [(a, b) for a, b in m if b - a >= MIN_EP]

def boundary_minute(x):
    """Clock minute (0..1439, midnight-based) of minimum sleep probability."""
    n_days = len(x) // 1440
    prob = (x.reshape(n_days, 1440) == SLEEP).mean(axis=0)
    k = np.ones(121) / 121
    sm = np.convolve(np.concatenate([prob[-60:], prob, prob[:60]]), k, mode='valid')
    # all minutes within 0.01 of the minimum; take the centre of the longest circular run
    low = sm <= sm.min() + 0.01
    idx = np.where(low)[0]
    if len(idx) == 1440: return 720
    # circular runs
    runs = []; cur = [idx[0]]
    for i in idx[1:]:
        if i == cur[-1] + 1: cur.append(i)
        else: runs.append(cur); cur = [i]
    runs.append(cur)
    if len(runs) > 1 and runs[0][0] == 0 and runs[-1][-1] == 1439:   # wrap-around
        runs[0] = runs[-1] + runs[0]; runs.pop()
    best = max(runs, key=len)
    return int(best[len(best) // 2] % 1440)

def process(pM, pdays):
    # build continuous timeline over days 1..9 (missing days filled with code 4)
    full = np.full((9, 1440), 4, dtype=np.int8)
    for d, row in zip(pdays, pM):
        if 1 <= d <= 9: full[d - 1] = row
    x = full.reshape(-1)
    eps = episodes_cont(x)
    if len(eps) < N_WIN: return None
    b = boundary_minute(x)
    # day k = [b + 1440k, b + 1440(k+1)), k = 0..7
    days = [(b + 1440 * k, b + 1440 * (k + 1)) for k in range(8)]
    info = []
    for (lo, hi) in days:
        seg = x[lo:hi]
        worn = np.isin(seg, [WAKE, SLEEP]).sum()
        de = [(a, e) for a, e in eps if lo <= (a + e) / 2 < hi]
        info.append(dict(lo=lo, hi=hi, valid=(worn >= MIN_VALID and len(de) > 0), eps=de))
    start = None
    for i in range(len(info) - N_WIN + 1):
        if all(info[j]['valid'] for j in range(i, i + N_WIN)): start = i; break
    if start is None: return None
    W = info[start:start + N_WIN]
    daily, mids, nnap = [], [], 0
    union = np.zeros(1440, bool)
    for w in W:
        daily.append(sum(e - a for a, e in w['eps']))
        for a, e in w['eps']:
            for t in range(a, e): union[t % 1440] = True
        main = max(w['eps'], key=lambda p: p[1] - p[0]); nnap += len(w['eps']) - 1
        mids.append(((main[0] + main[1]) / 2 - w['lo']))   # minutes since day boundary
    M = float(np.mean(daily)); TSR = int(union.sum()); SRT = M / TSR * 100
    sd_mid = float(np.std(mids, ddof=1))
    match = tot = 0
    for i in range(N_WIN - 1):
        a = x[W[i]['lo']:W[i]['hi']]; bb = x[W[i + 1]['lo']:W[i + 1]['hi']]
        bv = np.isin(a, [WAKE, SLEEP]) & np.isin(bb, [WAKE, SLEEP])
        match += int((a[bv] == bb[bv]).sum()); tot += int(bv.sum())
    SRI = match / tot * 200 - 100 if tot else np.nan
    return dict(boundary_min=b, boundary_clock=f'{b//60:02d}:{b%60:02d}', first_day=start + 1,
                M=M, TSR=TSR, SRT=SRT, SRI=SRI, SD_midpoint=sd_mid, n_naps=nnap,
                raw_sleep_min_per_day=float(np.mean([(x[w['lo']:w['hi']] == SLEEP).sum() for w in W])))

def main():
    t0 = time.time()
    subj = pd.read_csv(SUBJ).set_index('SEQN')
    hdr = pd.read_csv(DATA, nrows=0); mc = [c for c in hdr.columns if c.startswith('min_')]
    dt = {c: 'float32' for c in mc}; dt['SEQN'] = 'int32'; dt['PAXDAYM'] = 'int8'
    df = pd.read_csv(DATA, usecols=['SEQN', 'PAXDAYM'] + mc, dtype=dt)
    M = df[mc].fillna(4).to_numpy(dtype=np.int8); seqn = df['SEQN'].to_numpy(); day = df['PAXDAYM'].to_numpy(); del df
    order = np.lexsort((day, seqn)); M, seqn, day = M[order], seqn[order], day[order]
    bounds = np.flatnonzero(np.diff(seqn)) + 1
    starts = np.concatenate([[0], bounds]); ends = np.concatenate([bounds, [len(seqn)]])
    rows = []
    for a, b in zip(starts, ends):
        sq = int(seqn[a]); r = process(M[a:b], list(day[a:b]))
        if r is None: continue
        r['SEQN'] = sq
        if sq in subj.index:
            r['age'] = int(subj.at[sq, 'age_in_years_at_screening']); r['sex'] = subj.at[sq, 'gender']
        rows.append(r)
    out = pd.DataFrame(rows); out.to_csv(OUT, index=False)
    ad = out[out.age >= 20]
    print('participants:', len(starts), ' with 7 consecutive valid days:', len(out), ' adults:', len(ad))
    print(ad[['M', 'TSR', 'SRT', 'SRI', 'SD_midpoint', 'n_naps', 'raw_sleep_min_per_day']].describe().round(2).loc[['mean', 'std', '50%']])
    print('boundary clock distribution (adults):'); print((ad.boundary_min // 60).value_counts().sort_index().to_dict())
    print('r(SRT,SRI)=', ad[['SRT', 'SRI']].corr().iloc[0, 1].round(3), ' r(SRT,SDmid)=', ad[['SRT', 'SD_midpoint']].corr().iloc[0, 1].round(3))
    print(f'{time.time()-t0:.0f}s')

if __name__ == '__main__': main()
