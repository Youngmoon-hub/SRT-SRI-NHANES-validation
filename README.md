# The Sleep Regularity Test (SRT): validity against the Sleep Regularity Index and depressive symptoms in NHANES 2011–2014

Analysis code for: Kwon Y. *The Sleep Regularity Test: validity of a sleep regularity index derived from bedtimes and wake times in US adults.* (submitted to the Journal of Clinical Sleep Medicine, 2026)

## What the SRT is
SRT = M / TSR × 100, where **M** is the 7-day mean of daily sleep duration (all episodes, naps included; episode length = onset to offset) and **TSR** (total sleep range) is the number of clock minutes occupied by sleep on at least one of the 7 days. Days are delimited by main sleep (7 consecutive main sleep periods), not by clock time.

## Data (not redistributed here)
- Minute-level NHANES 2011–2014 accelerometer classifications (PAXPREDM) as curated on PhysioNet: Koffman & Muschelli 2025, https://doi.org/10.13026/9n0r-tv02 → `data/nhanes_1440_PAXPREDM.csv`, `data/subject-info.csv`
- NHANES questionnaire/exam files (cycles G and H): DEMO, DPQ, BMX, SMQ, ALQ, MCQ (`.xpt`) from https://www.cdc.gov/nchs/nhanes/ → `data/`

**PAXPREDM coding (NHANES documentation): 1 = wake wear, 2 = sleep wear, 3 = nonwear, 4 = unknown.** An earlier version of this repository treated code 1 as sleep; that version is superseded by the code here.

## Pipeline
```
python code/01b_compute_metrics_indiv.py                       # primary: individualized day boundary, 30-min merge (GAP_MERGE=60 for sensitivity)
python code/02_merge.py out/metrics_7day_indiv_gap30.csv out/analysis_indiv_gap30.csv
python code/03_analysis.py out/analysis_indiv_gap30.csv indiv_gap30
python code/01_compute_metrics.py                              # sensitivity: fixed noon boundary
python code/05_selfreport_sim.py                               # simulated diary-level input error (Table S2)
python code/04_fig1_concept.py; python code/06_figures_final.py # figures
```
Survey design (strata SDMVSTRA, PSU SDMVPSU, weight WTMEC2YR/2) with Taylor-linearized variances is implemented directly in `03_analysis.py` (no R dependency). Python 3.13; NumPy, pandas, SciPy, scikit-learn (AUC), matplotlib, pyreadstat.

## Outputs
`results/results_<rule>.json` (four extraction-rule combinations), `results/selfreport_sim.csv`, `figures/`.

## License
MIT for code. NHANES data are public domain; PhysioNet data under their license.
