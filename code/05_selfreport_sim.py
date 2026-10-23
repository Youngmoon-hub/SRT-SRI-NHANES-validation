"""
Measurement-error sensitivity analysis: how robust are the SRT results to
self-report (sleep-diary) levels of input error?

SRT needs only bedtimes and wake times. Starting from the accelerometer-derived
episodes used in the primary analysis, we perturb them as a diary would:
  (a) random timing error on each onset and offset  (SD = 15, 30, 45 min)
  (b) naps omitted with probability p              (0, 0.5, 1.0)
  (c) times rounded to the nearest 15 min          (diary rounding)
and recompute SRT, then (i) weighted r with SRI, (ii) weighted r with the
unperturbed SRT, (iii) mean bias, (iv) Model-3 OR per SD (original SD units)
for PHQ-9 >= 10, with the same covariates and survey design as the main analysis.
Replicates: R per condition; median and 2.5-97.5 percentile reported.
"""
import numpy as np, pandas as pd, json, sys, os, time, importlib.util
from scipy import stats

os.environ.setdefault('GAP_MERGE', '30')
spec = importlib.util.spec_from_file_location('m01b', 'code/01b_compute_metrics_indiv.py'); m01b = importlib.util.module_from_spec(spec); spec.loader.exec_module(m01b)
WAKE, SLEEP, MIN_VALID, N_WIN = m01b.WAKE, m01b.SLEEP, m01b.MIN_VALID, m01b.N_WIN

ANALYSIS = 'out/analysis_indiv_gap30.csv'; DATA = 'data/nhanes_1440_PAXPREDM.csv'
R = int(os.environ.get('R', 200)); SEED = 20261022

# ---------------------------------------------------------------- analytic sample (identical to 03_analysis.py)
d = pd.read_csv(ANALYSIS)
COVS_EXTRA = ['bmi', 'smoking', 'alcohol', 'chronic_cat']
d = d[(d.age_y >= 20)].dropna(subset=['phq9', 'SRT', 'SRI', 'SD_midpoint', 'race', 'educ', 'pir'] + COVS_EXTRA).copy()
d = d[d.implausible_sleep == 0].copy().set_index('SEQN')
d['dep'] = (d.phq9 >= 10).astype(float); d['age10'] = d.age_y / 10
SD_SRT, SD_SRI = d.SRT.std(), d.SRI.std()
d['zSRI'] = (d.SRI - d.SRI.mean()) / SD_SRI
print('analytic n =', len(d))

# ---------------------------------------------------------------- recover the 7-day episode windows
def windows(pM, pdays):
    full = np.full((9, 1440), 4, dtype=np.int8)
    for dd, row in zip(pdays, pM):
        if 1 <= dd <= 9: full[dd - 1] = row
    x = full.reshape(-1); eps = m01b.episodes_cont(x)
    if len(eps) < N_WIN: return None
    b = m01b.boundary_minute(x)
    days = [(b + 1440 * k, b + 1440 * (k + 1)) for k in range(8)]
    info = []
    for (lo, hi) in days:
        seg = x[lo:hi]; worn = np.isin(seg, [WAKE, SLEEP]).sum()
        de = [(a, e) for a, e in eps if lo <= (a + e) / 2 < hi]
        info.append(dict(lo=lo, hi=hi, valid=(worn >= MIN_VALID and len(de) > 0), eps=de))
    for i in range(len(info) - N_WIN + 1):
        if all(info[j]['valid'] for j in range(i, i + N_WIN)): return info[i:i + N_WIN]
    return None

t0 = time.time()
CACHE = 'out/episodes_7day_indiv_gap30.pkl'
if os.path.exists(CACHE):
    EP = pd.read_pickle(CACHE)
else:
  hdr = pd.read_csv(DATA, nrows=0); mc = [c for c in hdr.columns if c.startswith("min_")]
  df = pd.read_csv(DATA, usecols=['SEQN', 'PAXDAYM'] + mc, dtype={**{c: 'float32' for c in mc}, 'SEQN': 'int32', 'PAXDAYM': 'int8'})
  df = df[df.SEQN.isin(d.index)]
  EP = {}   # SEQN -> list of (onset, offset, is_main) over the 7 days (continuous minutes)
  for sq, g in df.groupby('SEQN'):
      g = g.sort_values('PAXDAYM'); W = windows(g[mc].fillna(4).to_numpy(dtype=np.int8), list(g.PAXDAYM))
      if W is None: continue
      rows = []
      for w in W:
          main = max(w['eps'], key=lambda p: p[1] - p[0])
          rows += [(a, e, (a, e) == main) for a, e in w['eps']]
      EP[int(sq)] = np.array(rows, dtype=float)
  del df
  pd.to_pickle(EP, CACHE)
d = d.loc[sorted(EP)]
print('episodes recovered for', len(EP), f'({time.time()-t0:.0f}s)')
# check: unperturbed recomputation reproduces SRT
def srt_from(eps):
    daily_total = eps[:, 1].sum() - eps[:, 0].sum()           # sum of episode lengths over 7 days
    M = daily_total / N_WIN
    union = np.zeros(1440, bool)
    for a, e in eps[:, :2].astype(int):
        idx = np.arange(a, e) % 1440; union[idx] = True
    return M / union.sum() * 100
chk = np.array([srt_from(EP[s]) for s in d.index]); print('max |recomputed - stored SRT| =', np.abs(chk - d.SRT.values).max().round(4))

# ---------------------------------------------------------------- design helpers (as in 03_analysis.py)
Wt, S, P = d.wt, d.SDMVSTRA, d.SDMVPSU
def svy_total_var(z, strata, psu):
    z = np.atleast_2d(z.T).T; k = z.shape[1]; V = np.zeros((k, k))
    df_ = pd.DataFrame(z); df_['_s'] = strata.values; df_['_p'] = psu.values
    for s, g in df_.groupby('_s'):
        zp = g.groupby('_p')[list(range(k))].sum().values; nh = zp.shape[0]
        if nh < 2: continue
        dev = zp - zp.mean(axis=0); V += nh / (nh - 1) * dev.T @ dev
    return V
DDF = len(set(zip(S, P))) - S.nunique()
def wcorr(x, y):
    w = Wt.values; x = np.asarray(x, float); y = np.asarray(y, float); sw = w.sum()
    mx = (w*x).sum()/sw; my = (w*y).sum()/sw
    sxx = (w*(x-mx)**2).sum()/sw; syy = (w*(y-my)**2).sum()/sw; sxy = (w*(x-mx)*(y-my)).sum()/sw
    return sxy/np.sqrt(sxx*syy)
def svy_logit(y, X, w, maxit=50):
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = 1/(1+np.exp(-X@b)); H = X.T @ (X*(w*p*(1-p))[:, None]); g = X.T @ (w*(y-p))
        step = np.linalg.solve(H, g); b += step
        if np.max(np.abs(step)) < 1e-9: break
    p = 1/(1+np.exp(-X@b)); Hi = np.linalg.inv(X.T @ (X*(w*p*(1-p))[:, None]))
    V = Hi @ svy_total_var(X*(w*(y-p))[:, None], S, P) @ Hi
    return b, V
def design_matrix(df_, terms):
    cols = {'const': np.ones(len(df_))}
    for t in terms:
        if t in ('race', 'educ', 'smoking', 'alcohol', 'chronic_cat'):
            ref = {'race': 'Non-Hispanic White', 'educ': 'College graduate+', 'smoking': 'Never', 'alcohol': 'Never', 'chronic_cat': '0'}[t]
            for lev in sorted(df_[t].unique()):
                if lev != ref: cols[f'{t}[{lev}]'] = (df_[t] == lev).astype(float).values
        else: cols[t] = df_[t].values.astype(float)
    return pd.DataFrame(cols)
base = ['age10', 'female', 'race', 'educ', 'pir', 'bmi', 'smoking', 'alcohol', 'chronic_cat', 'sleep_h']
XA = design_matrix(d, base); XB = design_matrix(d, base + ['zSRI']); y = d.dep.values; w = Wt.values

def _or(XBASE, srt):
    z = (srt - srt.mean()) / SD_SRT            # original SD units -> OR comparable with the main analysis
    X = np.column_stack([XBASE.values, z]); b, V = svy_logit(y, X, w); iz = X.shape[1] - 1
    se = np.sqrt(V[iz, iz]); t = b[iz]/se
    return np.exp(b[iz]), 2*stats.t.sf(abs(t), DDF)
def model3(srt): return _or(XB, srt)      # with SRI (incremental validity, as Model 3)
def model_noSRI(srt): return _or(XA, srt) # without SRI (the self-report use case: SRI unavailable)

# ---------------------------------------------------------------- simulation
rng = np.random.default_rng(SEED)
def perturb(eps, sd, p_nap, round15):
    keep = np.ones(len(eps), bool)
    if p_nap > 0:                                          # naps (non-main episodes) omitted with prob p_nap
        keep = (eps[:, 2] == 1) | (rng.random(len(eps)) >= p_nap)
    e = eps[keep, :2].copy()
    if sd > 0: e += rng.normal(0, sd, e.shape)
    if round15: e = np.round(e / 15) * 15
    e[:, 1] = np.maximum(e[:, 1], e[:, 0] + 15)         # keep offset after onset
    return e

conds = [dict(sd=0, p_nap=0, round15=False)]
for sd in (15, 30, 45):
    for p_nap in (0, 0.5, 1.0):
        conds.append(dict(sd=sd, p_nap=p_nap, round15=True))
conds.append(dict(sd=0, p_nap=1.0, round15=False))   # naps omitted only
conds.append(dict(sd=0, p_nap=0, round15=True))      # rounding only

SRT0 = d.SRT.values; SRI = d.SRI.values
results = []
for c in conds:
    t1 = time.time(); reps = []
    nrep = 1 if (c['sd'] == 0 and c['p_nap'] == 0 and not c['round15']) else R
    for r in range(nrep):
        srt = np.array([srt_from(perturb(EP[s], c['sd'], c['p_nap'], c['round15'])) for s in d.index])
        OR, pv = model3(srt); ORa, pva = model_noSRI(srt)
        reps.append(dict(r_sri=wcorr(srt, SRI), r_srt0=wcorr(srt, SRT0), bias=float(np.average(srt - SRT0, weights=w)),
                         OR=OR, p=pv, ORa=ORa, pa=pva))
    reps = pd.DataFrame(reps)
    q = lambda col, qq: float(np.percentile(reps[col], qq))
    row = dict(**c, n_rep=nrep,
               r_sri=q('r_sri', 50), r_sri_lo=q('r_sri', 2.5), r_sri_hi=q('r_sri', 97.5),
               r_srt0=q('r_srt0', 50), bias=q('bias', 50),
               OR=q('OR', 50), OR_lo=q('OR', 2.5), OR_hi=q('OR', 97.5),
               p_median=q('p', 50), frac_p05=float((reps.p < .05).mean()),
               ORa=q('ORa', 50), ORa_lo=q('ORa', 2.5), ORa_hi=q('ORa', 97.5), pa_median=q('pa', 50), frac_pa05=float((reps.pa < .05).mean()))
    results.append(row)
    print(f"sd={c['sd']:2d} p_nap={c['p_nap']:.1f} round={int(c['round15'])}: r(SRI)={row['r_sri']:.3f} [{row['r_sri_lo']:.3f},{row['r_sri_hi']:.3f}] "
          f"r(SRT0)={row['r_srt0']:.3f} bias={row['bias']:+.1f} OR={row['OR']:.3f} [{row['OR_lo']:.3f},{row['OR_hi']:.3f}] "
          f"p_med={row['p_median']:.4f} P(p<.05)={row['frac_p05']:.2f} | noSRI OR={row['ORa']:.3f} [{row['ORa_lo']:.3f},{row['ORa_hi']:.3f}] P(p<.05)={row['frac_pa05']:.2f}  ({time.time()-t1:.0f}s)", flush=True)

res = pd.DataFrame(results); res.to_csv('out/selfreport_sim.csv', index=False)
json.dump(dict(R=R, seed=SEED, n=len(d), sd_srt=float(SD_SRT), rows=results), open('out/selfreport_sim.json', 'w'), indent=1)
print('saved out/selfreport_sim.csv/.json')
