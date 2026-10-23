"""Merge sleep metrics with NHANES DEMO (covariates, design variables) and DPQ (PHQ-9)."""
import pandas as pd, numpy as np, pyreadstat, sys

METRICS = sys.argv[1] if len(sys.argv) > 1 else 'out/metrics_7day_gap30.csv'
OUT     = sys.argv[2] if len(sys.argv) > 2 else 'out/analysis_gap30.csv'

m = pd.read_csv(METRICS)

demo = pd.concat([pyreadstat.read_xport(f'data/DEMO_{c}.xpt')[0] for c in 'GH'], ignore_index=True)
demo = demo[['SEQN', 'SDDSRVYR', 'RIAGENDR', 'RIDAGEYR', 'RIDRETH3', 'DMDEDUC2', 'INDFMPIR',
             'WTMEC2YR', 'SDMVPSU', 'SDMVSTRA']]

dpq = pd.concat([pyreadstat.read_xport(f'data/DPQ_{c}.xpt')[0] for c in 'GH'], ignore_index=True)
items = [f'DPQ0{i}0' for i in range(1, 10)]
dpq[items] = dpq[items].where(dpq[items].isin([0, 1, 2, 3]))      # 7 / 9 -> missing
dpq['phq9_n_items'] = dpq[items].notna().sum(axis=1)
dpq['phq9'] = np.where(dpq['phq9_n_items'] == 9, dpq[items].sum(axis=1), np.nan)
dpq = dpq[['SEQN', 'phq9', 'phq9_n_items']]

# --- Additional covariates (BMX, SMQ, ALQ, MCQ) ---
def loadx(pref):
    return pd.concat([pyreadstat.read_xport(f'data/{pref}_{c}.xpt')[0] for c in 'GH'], ignore_index=True)
bmx = loadx('BMX')[['SEQN', 'BMXBMI']]
smq = loadx('SMQ')[['SEQN', 'SMQ020', 'SMQ040']]
alq = loadx('ALQ')[['SEQN', 'ALQ101', 'ALQ110']]
mcq_cols = ['MCQ160A', 'MCQ160B', 'MCQ160C', 'MCQ160D', 'MCQ160E', 'MCQ160F', 'MCQ160K', 'MCQ160L', 'MCQ160M', 'MCQ160N', 'MCQ220']
mcq = loadx('MCQ')[['SEQN'] + mcq_cols]

d = (m.merge(demo, on='SEQN', how='left').merge(dpq, on='SEQN', how='left')
     .merge(bmx, on='SEQN', how='left').merge(smq, on='SEQN', how='left')
     .merge(alq, on='SEQN', how='left').merge(mcq, on='SEQN', how='left'))
d['SEQN'] = d['SEQN'].astype(int)

# Derived variables
d['female']   = (d['RIAGENDR'] == 2).astype(int)
d['age_y']    = d['RIDAGEYR']
race_map = {1: 'Mexican American', 2: 'Other Hispanic', 3: 'Non-Hispanic White',
            4: 'Non-Hispanic Black', 6: 'Non-Hispanic Asian', 7: 'Other/multiracial'}
d['race']     = d['RIDRETH3'].map(race_map)
edu_map  = {1: '<High school', 2: '<High school', 3: 'High school/GED',
            4: 'Some college', 5: 'College graduate+'}
d['educ']     = d['DMDEDUC2'].map(edu_map)              # 7/9 -> NaN
d['pir']      = d['INDFMPIR']
d['wt']       = d['WTMEC2YR'] / 2.0                      # two cycles combined
d['dep']      = np.where(d['phq9'].isna(), np.nan, (d['phq9'] >= 10).astype(float))
d['sleep_h']  = d['M'] / 60.0                            # mean daily episode minutes -> hours
d['bmi']      = d['BMXBMI']
# smoking: never / former / current
d['smoking']  = np.select([d.SMQ020 == 2, (d.SMQ020 == 1) & (d.SMQ040 == 3), (d.SMQ020 == 1) & d.SMQ040.isin([1, 2])],
                          ['Never', 'Former', 'Current'], default=None)
# alcohol: never (<12 drinks lifetime) / former (>=12 lifetime, <12 past year) / current (>=12 past year)
d['alcohol']  = np.select([(d.ALQ101 == 2) & (d.ALQ110 == 2), (d.ALQ101 == 2) & (d.ALQ110 == 1), d.ALQ101 == 1],
                          ['Never', 'Former', 'Current'], default=None)
# chronic conditions: count of physician-diagnosed conditions (arthritis, CHF, CHD, angina, MI, stroke,
# chronic bronchitis, liver disease, thyroid disease, gout, cancer); 7/9 -> missing
mc = d[mcq_cols].where(d[mcq_cols].isin([1, 2]))
d['n_chronic'] = np.where(mc.isna().any(axis=1), np.nan, (mc == 1).sum(axis=1))
d['chronic_cat'] = pd.cut(d['n_chronic'], [-1, 0, 1, 99], labels=['0', '1', '2+']).astype(object)
# plausibility flag: mean daily episode minutes outside 3-15 h
d['implausible_sleep'] = ((d['M'] < 180) | (d['M'] > 900)).astype(int)

d.to_csv(OUT, index=False)

# Flow numbers
print('7-day sample (all ages):', len(d))
a = d[d.age_y >= 20]; print('age>=20:', len(a))
b = a[a.phq9.notna()];  print('  + PHQ-9 complete:', len(b))
c = b.dropna(subset=['SRT', 'SRI', 'SD_midpoint']); print('  + all 3 regularity indices:', len(c))
e = c.dropna(subset=['race', 'educ', 'pir']); print('  + demographic covariates complete:', len(e))
f = e.dropna(subset=['bmi', 'smoking', 'alcohol', 'chronic_cat']); print('  + BMI/smoking/alcohol/chronic complete:', len(f))
g = f[f.implausible_sleep == 0]; print('  + plausible mean sleep (3-15 h):', len(g))
print('implausible among demographic-complete:', int(e.implausible_sleep.sum()))
e = g
print('depressive symptoms (PHQ-9>=10), unweighted %:', round(e.dep.mean() * 100, 2))
print('missing by variable among age>=20:'); print(a[['phq9', 'SRT', 'SRI', 'SD_midpoint', 'race', 'educ', 'pir', 'bmi', 'smoking', 'alcohol', 'chronic_cat']].isna().sum())
