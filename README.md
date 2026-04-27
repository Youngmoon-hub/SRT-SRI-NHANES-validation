# SRT–SRI Convergent Validity in NHANES 2011–2014

Analysis code for the manuscript:

> **Convergent Validity of the Sleep Regularity Test with the Sleep Regularity
> Index in a US National Sample: A Diary-Compatible Measure for Population
> Sleep Health**
>
> Youngmoon Kwon, Ph.D. (Inha University)
> Submitted to *Sleep Health: Journal of the National Sleep Foundation*, 2026.

- **Author**: Youngmoon Kwon, Department of Physical Education, Inha University, Incheon, Republic of Korea
- **Email**: ym1980@gmail.com
- **ORCID**: [0009-0005-5187-0481](https://orcid.org/0009-0005-5187-0481)

## Overview

This repository contains the complete analysis pipeline used in the manuscript.
The pipeline computes the **Sleep Regularity Test (SRT)** and the
**Sleep Regularity Index (SRI)** from NHANES 2011–2014 minute-level wrist
actigraphy data, evaluates their convergent validity, and generates all figures
and tables in the manuscript.

## Repository structure

```
.
├── README.md                       # This file
├── LICENSE                         # MIT license
├── requirements.txt                # Python dependencies
├── nhanes_srt_sri_analysis.py      # Main analysis (SRT, SRI, statistics)
└── nhanes_figures.py               # Figure generation (Figures 1–4)
```

## Data source

The analysis uses publicly available data from PhysioNet, which redistributes
the NHANES 2011–2014 Physical Activity Monitor data in a per-person/per-day
format:

- PhysioNet dataset: <https://physionet.org/content/minute-level-step-count-nhanes/1.0.1/>
- Original NHANES files: <https://www.cdc.gov/nchs/nhanes/>

Required input files (place in the same directory as the scripts):

| File | Source | Approx. size |
|---|---|---|
| `subject-info.csv` | PhysioNet (root level) | 0.9 MB |
| `nhanes_1440_PAXPREDM.csv` | PhysioNet `csv/` folder, decompress `.csv.xz` | ~380 MB |

Data are redistributed under their original public domain status; this
repository does **not** include the data files themselves.

## Requirements

- Python 3.10 or higher
- 16 GB RAM recommended (analysis loads the full minute-level array)

Python packages (see `requirements.txt`):

```
numpy>=1.25
pandas>=2.0
scipy>=1.11
matplotlib>=3.7
```

Install with:

```bash
pip install -r requirements.txt
```

## How to reproduce the analysis

1. **Obtain the data** from PhysioNet (link above). Decompress the
   `nhanes_1440_PAXPREDM.csv.xz` file.

2. **Place the two data files** in the repository root next to the scripts.

3. **Run the analysis**:

   ```bash
   python nhanes_srt_sri_analysis.py
   ```

   This produces `nhanes_metrics.csv` (one row per participant) and prints
   all manuscript-relevant statistics. Runtime is approximately 1 minute.

4. **Generate the figures**:

   ```bash
   python nhanes_figures.py
   ```

   This produces `figures/Figure1–4.tiff` (600 DPI) and `Figure1–4.png`
   (300 DPI).

## Methods summary

### Sample inclusion

- Age ≥ 6 years
- ≥ 3 valid wear days, where each valid day has ≥ 960 valid (worn) minutes
- ≥ 3 main sleep episodes
- Days 1–7 only (NHANES PAM 7-day window standard)

### Sleep episode extraction

- Continuous runs of `PAXPREDM = 1` (sleep) lasting **≥ 60 minutes** are
  classified as sleep episodes. The 60-minute threshold conservatively
  excludes transient wake-sleep transitions misclassified by the wrist-modified
  Cole–Kripke algorithm.
- Episodes are extracted per **noon-to-noon window** to encompass typical
  nocturnal sleep within a single 24-hour interval.
- The longest qualifying episode in each window is the **main sleep episode**;
  any others are **additional sleep episodes** (typically naps, but may include
  secondary sleep periods in shift workers or older adults).

### SRT (Sleep Regularity Test)

```
SRT = (M̄ / TSR) × 100
```

where:

- **M̄** = mean daily total sleep duration (minutes), summed across all
  main and additional sleep episodes
- **TSR** = union of sleep minutes across the 7-day window (i.e., total
  minutes classified as sleep on at least one day)

SRT ranges from 0 to 100; higher values indicate greater temporal concentration
of sleep within a consistent time window.

### SRI (Sleep Regularity Index)

Following Phillips et al. (2017):

```
SRI = (1 / ((N − 1) × 1440)) × Σ Σ δ[s(d, t) = s(d+1, t)] × 200 − 100
```

Where N is the number of valid days, s(d, t) is the sleep/wake state on day
d at minute t, and δ is the indicator function. SRI is computed only on
minutes where both consecutive days have valid (worn) data.

## Output

`nhanes_metrics.csv` contains one row per participant with the following columns:

| Column | Description |
|---|---|
| `SEQN` | NHANES participant ID |
| `age` | Age in years at screening |
| `sex` | "Male" / "Female" |
| `n_valid_days` | Number of days with ≥ 960 valid minutes |
| `n_main_eps` | Number of main sleep episodes |
| `n_additional_eps` | Number of additional sleep episodes |
| `M_bar` | Mean daily total sleep duration (minutes) |
| `TSR` | Union of sleep minutes across the window (minutes) |
| `SRT` | Sleep Regularity Test, 0–100 |
| `SRI` | Sleep Regularity Index, −100 to +100 |

## Citation

If you use this code, please cite the manuscript (publication forthcoming) and
the underlying NHANES and PhysioNet sources.

## License

Released under the [MIT License](LICENSE). The NHANES data themselves are in the
US public domain.

## Disclosure

The author is the developer of the Sleep Regularity Test (SRT). To support
transparency, the complete analysis pipeline is open and the underlying data
are public. The author used generative AI tools to support code refactoring
and language editing; all scientific content, analyses, and conclusions are
the author's sole responsibility.
