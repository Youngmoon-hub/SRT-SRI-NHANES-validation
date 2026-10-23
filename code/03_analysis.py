"""
Design-based analyses for the SBR manuscript.

- Survey design: strata = SDMVSTRA, PSU = SDMVPSU, weight = WTMEC2YR/2
- Weighted descriptives (Table 1), weighted Pearson correlations with
  design-based SEs (convergent validity), hierarchical survey-weighted
  logistic regression (Models 1-3) with Taylor-linearised (sandwich)
  variance, design-based Wald tests, VIF, sensitivity models.

Implementation note: no 'survey' package in Python, so the standard
Taylor linearisation estimator (as in R survey::svyglm, Stata svy) is coded
explicitly in `svy_var`.
"""
import numpy as np, pandas as pd, sys, json
from scipy import stats

INP = sys.argv[1] if len(sys.argv) > 1 else 'out/analysis_gap30.csv'
TAG = sys.argv[2] if len(sys.argv) > 2 else 'gap30'

d = pd.read_csv(INP)
COVS_EXTRA = ['bmi', 'smoking', 'alcohol', 'chronic_cat']
d = d[(d.age_y >= 20)].dropna(subset=['phq9', 'SRT', 'SRI', 'SD_midpoint', 'race', 'educ', 'pir'] + COVS_EXTRA).copy()
d = d[d.implausible_sleep == 0].copy()
d['dep'] = (d.phq9 >= 10).astype(float)
N = len(d)
print(f'[{TAG}] analytic sample n = {N}')

# ---------------------------------------------------------------- design helpers
def svy_total_var(z, strata, psu):
    """Taylor linearisation variance of a total of per-observation scores z (n x k)."""
    z = np.atleast_2d(z.T).T
    k = z.shape[1]; V = np.zeros((k, k))
    df = pd.DataFrame(z); df['_s'] = strata.values; df['_p'] = psu.values
    for s, g in df.groupby('_s'):
        zp = g.groupby('_p')[list(range(k))].sum().values     # PSU totals
        nh = zp.shape[0]
        if nh < 2: continue
        dev = zp - zp.mean(axis=0)
        V += nh / (nh - 1) * dev.T @ dev
    return V

def design_df(strata, psu):
    return len(set(zip(strata, psu))) - strata.nunique()

W, S, P = d.wt, d.SDMVSTRA, d.SDMVPSU
DDF = design_df(S, P)
print('design df (PSUs - strata) =', DDF)

def wmean(x):
    return float(np.sum(W * x) / W.sum())

def wmean_se(x):
    m = wmean(x); z = W * (x - m) / W.sum()
    return m, float(np.sqrt(svy_total_var(z.values, S, P)[0, 0]))

def wcorr(x, y):
    """Weighted Pearson r with linearised SE (delta method on the 5 totals)."""
    w = W.values; x = x.values.astype(float); y = y.values.astype(float)
    sw = w.sum(); mx = (w*x).sum()/sw; my = (w*y).sum()/sw
    sxx = (w*(x-mx)**2).sum()/sw; syy = (w*(y-my)**2).sum()/sw; sxy = (w*(x-mx)*(y-my)).sum()/sw
    r = sxy/np.sqrt(sxx*syy)
    # influence function of r
    u = (x-mx)/np.sqrt(sxx); v = (y-my)/np.sqrt(syy)
    infl = w*(u*v - 0.5*r*(u**2 + v**2))/sw
    se = float(np.sqrt(svy_total_var(infl, S, P)[0, 0]))
    return float(r), se

# ---------------------------------------------------------------- Table 1
out = {'n': N, 'design_df': DDF}
print('\n--- Table 1: weighted descriptives ---')
tab1 = {}
for v in ['age_y', 'female', 'pir', 'bmi', 'sleep_h', 'SRT', 'SRI', 'SD_midpoint', 'TSR', 'phq9', 'dep', 'n_naps']:
    m, se = wmean_se(d[v]); sd = float(np.sqrt(wmean((d[v]-m)**2)))
    tab1[v] = dict(mean=m, se=se, sd=sd, unweighted_mean=float(d[v].mean()), unweighted_sd=float(d[v].std()))
    print(f'  {v:12s} weighted M = {m:8.2f} (SD {sd:6.2f}, SE {se:5.2f}) | unweighted M = {d[v].mean():8.2f} (SD {d[v].std():.2f})')
for v in ['race', 'educ', 'smoking', 'alcohol', 'chronic_cat']:
    tab1[v] = {k: wmean((d[v] == k).astype(float)) * 100 for k in d[v].unique()}
    print(f'  {v}: ' + ', '.join(f'{k} {p:.1f}%' for k, p in tab1[v].items()))
tab1['dep_by_group'] = {}
# design-based group comparison (Wald t on weighted mean difference via indicator regression)
def group_test(v):
    x = d[v].astype(float).values; g = d.dep.values; w = W.values
    X = np.c_[np.ones(len(x)), g]
    b = np.linalg.solve(X.T@(X*w[:,None]), X.T@(w*x)); r = x - X@b
    Hi = np.linalg.inv(X.T@(X*w[:,None])); V = Hi @ svy_total_var(X*(w*r)[:,None], S, P) @ Hi
    tt = b[1]/np.sqrt(V[1,1]); return float(2*stats.t.sf(abs(tt), DDF))
tab1['dep_group_p'] = {v: group_test(v) for v in ['age_y', 'female', 'pir', 'bmi', 'sleep_h', 'SRT', 'SRI', 'SD_midpoint', 'TSR']}
print('  group-difference p (dep vs no dep):', {k: round(v, 4) for k, v in tab1['dep_group_p'].items()})
for g, sub in d.groupby('dep'):
    ww = sub.wt
    tab1['dep_by_group'][int(g)] = {v: float((ww * sub[v]).sum() / ww.sum()) for v in ['age_y', 'female', 'pir', 'bmi', 'sleep_h', 'SRT', 'SRI', 'SD_midpoint', 'TSR']}
    for cv in ['race', 'educ', 'smoking', 'alcohol', 'chronic_cat']:
        tab1['dep_by_group'][int(g)][cv] = {k: float((ww * (sub[cv] == k)).sum() / ww.sum() * 100) for k in d[cv].unique()}
    tab1['dep_by_group'][int(g)]['n'] = int(len(sub))
print('  by depressive symptoms:', json.dumps(tab1['dep_by_group'], indent=None))
out['table1'] = tab1

# ---------------------------------------------------------------- Convergent validity
print('\n--- Convergent validity (weighted Pearson r, design SE) ---')
cv = {}
for a, b in [('SRT', 'SRI'), ('SRT', 'SD_midpoint'), ('SRI', 'SD_midpoint'), ('SRT', 'sleep_h'), ('SRI', 'sleep_h')]:
    r, se = wcorr(d[a], d[b]); lo, hi = r - stats.t.ppf(.975, DDF)*se, r + stats.t.ppf(.975, DDF)*se
    ru = float(d[[a, b]].corr().iloc[0, 1]); rs = float(d[[a, b]].corr(method='spearman').iloc[0, 1])
    cv[f'{a}~{b}'] = dict(r_w=r, se=se, ci=[lo, hi], r_unweighted=ru, rho_unweighted=rs)
    print(f'  {a}~{b}: weighted r = {r:.3f} [{lo:.3f}, {hi:.3f}]  (unweighted r = {ru:.3f}, rho = {rs:.3f})')
out['convergent'] = cv

# ---------------------------------------------------------------- Survey logistic regression
def svy_logit(y, X, w, strata, psu, maxit=50):
    """Pseudo-ML weighted logistic regression + Taylor linearised covariance."""
    X = np.asarray(X, float); y = np.asarray(y, float); w = np.asarray(w, float)
    b = np.zeros(X.shape[1])
    for _ in range(maxit):
        p = 1/(1+np.exp(-X@b)); Wd = w*p*(1-p)
        H = X.T @ (X*Wd[:, None]); g = X.T @ (w*(y-p))
        step = np.linalg.solve(H, g); b += step
        if np.max(np.abs(step)) < 1e-9: break
    p = 1/(1+np.exp(-X@b))
    H = X.T @ (X*(w*p*(1-p))[:, None]); Hi = np.linalg.inv(H)
    scores = X*(w*(y-p))[:, None]
    V = Hi @ svy_total_var(scores, strata, psu) @ Hi
    ll = np.sum(w*(y*np.log(p+1e-300)+(1-y)*np.log(1-p+1e-300)))
    return b, V, p, ll

def design_matrix(df, terms):
    cols = {'const': np.ones(len(df))}
    for t in terms:
        if t in ('race', 'educ', 'smoking', 'alcohol', 'chronic_cat'):
            ref = {'race': 'Non-Hispanic White', 'educ': 'College graduate+', 'smoking': 'Never', 'alcohol': 'Never', 'chronic_cat': '0'}[t]
            for lev in sorted(df[t].unique()):
                if lev != ref: cols[f'{t}[{lev}]'] = (df[t] == lev).astype(float).values
        else:
            cols[t] = df[t].values.astype(float)
    return pd.DataFrame(cols)

# standardise regularity indices (unweighted SD of analytic sample)
for v in ['SRT', 'SRI', 'SD_midpoint']:
    d[f'z{v}'] = (d[v] - d[v].mean()) / d[v].std()
    out.setdefault('sd_units', {})[v] = float(d[v].std())
d['age10'] = d.age_y / 10

base = ['age10', 'female', 'race', 'educ', 'pir', 'bmi', 'smoking', 'alcohol', 'chronic_cat', 'sleep_h']
y = d.dep.values

def fit(terms, label):
    X = design_matrix(d, terms); b, V, p, ll = svy_logit(y, X.values, W.values, S, P)
    se = np.sqrt(np.diag(V)); t = b/se; pv = 2*stats.t.sf(np.abs(t), DDF)
    tcrit = stats.t.ppf(.975, DDF)
    res = pd.DataFrame({'coef': b, 'se': se, 'OR': np.exp(b), 'lo': np.exp(b - tcrit*se), 'hi': np.exp(b + tcrit*se), 'p': pv}, index=X.columns)
    # AUC (weighted) and McFadden pseudo-R2 vs null
    from sklearn.metrics import roc_auc_score
    auc = roc_auc_score(y, p, sample_weight=W.values)
    p0 = np.sum(W*y)/W.sum(); ll0 = np.sum(W*(y*np.log(p0)+(1-y)*np.log(1-p0)))
    print(f'\n  [{label}] terms={terms}\n    weighted AUC = {auc:.4f}, McFadden R2 = {1-ll/ll0:.4f}')
    show = [i for i in res.index if i.startswith('z') or i == 'sleep_h']
    print(res.loc[show].round(4).to_string())
    return dict(res=res, auc=float(auc), r2=float(1-ll/ll0), ll=float(ll), X=X, V=V, b=b)

def wald(fitres, names):
    idx = [list(fitres['X'].columns).index(n) for n in names]
    bb = fitres['b'][idx]; VV = fitres['V'][np.ix_(idx, idx)]
    F = float(bb @ np.linalg.solve(VV, bb)) / len(idx)
    pv = float(stats.f.sf(F, len(idx), DDF - len(idx) + 1))
    return F, pv

print('\n--- Hierarchical survey logistic regression: PHQ-9 >= 10 ---')
m1 = fit(base, 'Model 1: covariates + sleep duration')
m2 = fit(base + ['zSRI'], 'Model 2: + SRI')
m3 = fit(base + ['zSRI', 'zSRT'], 'Model 3: + SRT')
F2, p2 = wald(m2, ['zSRI']); F3, p3 = wald(m3, ['zSRT'])
print(f'\n  Design-based Wald test, SRI added (M1->M2): F(1,{DDF}) = {F2:.2f}, p = {p2:.4g}')
print(f'  Design-based Wald test, SRT added (M2->M3): F(1,{DDF}) = {F3:.2f}, p = {p3:.4g}')

# Clinically interpretable unit: OR per 10-point SRT / 10-point SRI (raw scale) in model 3
d['SRT10'] = d.SRT / 10; d['SRI10'] = d.SRI / 10
m3raw = fit(base + ['SRI10', 'SRT10'], 'Model 3 (raw units: per 10 points)')
out_raw = {k: m3raw['res'].loc[k, ['OR', 'lo', 'hi', 'p']].to_dict() for k in ['SRI10', 'SRT10']}
print('  per-10-point ORs:', {k: {kk: round(vv, 3) for kk, vv in v.items()} for k, v in out_raw.items()})
# VIF (unweighted, on model 3 design matrix without constant)
X3 = m3['X'].drop(columns='const')
vif = {}
for c in X3.columns:
    others = X3.drop(columns=c).values; others = np.c_[np.ones(len(others)), others]
    beta, *_ = np.linalg.lstsq(others, X3[c].values, rcond=None)
    r2 = 1 - np.sum((X3[c].values - others@beta)**2)/np.sum((X3[c].values - X3[c].mean())**2)
    vif[c] = 1/(1-r2)
print('  VIF (model 3):', {k: round(v, 2) for k, v in vif.items() if k.startswith('z') or k == 'sleep_h'}, ' max =', round(max(vif.values()), 2))

# Sensitivity analyses
print('\n--- Sensitivity ---')
s1 = fit(base + ['zSRI', 'zSD_midpoint', 'zSRT'], 'S1: + SD midpoint in model')
s2 = fit(base + ['zSRT'], 'S2: SRT first (without SRI)')
s3 = fit(base + ['zSRT', 'zSRI'], 'S3: SRI added after SRT')
Fs3, ps3 = wald(s3, ['zSRI']); print(f'    Wald SRI added after SRT: F = {Fs3:.2f}, p = {ps3:.4g}')
s4 = fit(base + ['zSD_midpoint'], 'S4: SD midpoint alone')
s5 = fit(base + ['zSD_midpoint', 'zSRT'], 'S5: SRT added after SD midpoint')

# Continuous PHQ-9 (weighted linear regression, design SE)
def svy_lin(yv, X):
    X = np.asarray(X, float); w = W.values
    b = np.linalg.solve(X.T@(X*w[:, None]), X.T@(w*yv)); r = yv - X@b
    Hi = np.linalg.inv(X.T@(X*w[:, None])); V = Hi @ svy_total_var(X*(w*r)[:, None], S, P) @ Hi
    return b, np.sqrt(np.diag(V))
Xl = design_matrix(d, base + ['zSRI', 'zSRT']); bl, sel = svy_lin(d.phq9.values, Xl.values)
i = list(Xl.columns).index('zSRT'); j = list(Xl.columns).index('zSRI')
print(f'\n  PHQ-9 total (linear, M3 terms): b_SRT = {bl[i]:.3f} (SE {sel[i]:.3f}, p = {2*stats.t.sf(abs(bl[i]/sel[i]), DDF):.4g}); '
      f'b_SRI = {bl[j]:.3f} (SE {sel[j]:.3f}, p = {2*stats.t.sf(abs(bl[j]/sel[j]), DDF):.4g})')

# Save
def pack(m): return {'auc': m['auc'], 'r2': m['r2'], 'coef': m['res'].round(5).to_dict(orient='index')}
out['models'] = {'M1': pack(m1), 'M2': pack(m2), 'M3': pack(m3), 'S1': pack(s1), 'S2': pack(s2), 'S3': pack(s3), 'S4': pack(s4), 'S5': pack(s5)}
out['wald'] = {'SRI_M1toM2': [F2, p2], 'SRT_M2toM3': [F3, p3], 'SRI_afterSRT': [Fs3, ps3]}
out['vif'] = vif
out['per10'] = out_raw
out['phq9_linear'] = {'b_SRT': float(bl[i]), 'se_SRT': float(sel[i]), 'b_SRI': float(bl[j]), 'se_SRI': float(sel[j])}
json.dump(out, open(f'out/results_{TAG}.json', 'w'), indent=1, default=float)
print(f'\nsaved out/results_{TAG}.json')
