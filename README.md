# A/B Test: Loan Screening AI Model — Performance Analysis

> **Warwick Business School** Advanced Data Analysis (IB98D0)  
> Based on Group 6's original report (2025.02) — extended with behavioral analysis and business impact quantification.

---

## Overview

This project evaluates whether an AI-assisted loan screening model significantly improves loan officer decision accuracy compared to the standard process, using a 10-day A/B test with 38 loan officers (Control: 10, Treatment: 28).

Rather than simply replicating the original Warwick report, this analysis addresses its methodological gaps and adds two entirely new analytical layers.

---

## Key Results

| OEC | Control | Treatment | Δ | Cohen's d |
|-----|---------|-----------|---|-----------|
| **F1** (Primary) | 43.69% | 61.52% | **+17.83pp** | 3.73 *(Huge)* |
| PrecisionAp | 74.49% | 86.46% | +11.97pp | 4.56 *(Huge)* |
| PrecisionRej | 34.26% | 55.01% | +20.75pp | 3.70 *(Huge)* |

All three metrics significant after Bonferroni correction (p ≈ 0, α = 0.017).  
**Recommendation: Discontinue experiment → Full deployment.**

**Financial Impact (per officer, 10 days):** $107K ~ $187K in loss reduction  
(consistent across $20K / $27.5K / $35K loan scenarios)

---

## Improvements over Original Report

| Item | Warwick Original | This Analysis |
|------|-----------------|---------------|
| Normality test | Skewness/Kurtosis only | + Shapiro-Wilk |
| Multiple comparison | None | Bonferroni correction |
| Cohen's d reference | Incorrect citation | Sawilowsky (2009) |
| CI interpretation | Table only | Actionable sentences |
| Experiment integrity | Partial | Day-1 Sanity Check |
| Behavioral analysis | None | Confidence, Automation Bias, Skill-based |
| Business impact | None | Dollar conversion + Sensitivity analysis |

---

## Behavioral Analysis Highlights

- **Confidence change:** Treatment +6.74 vs Control +4.73 (p = 0.525, not significant)
- **Automation Bias:** 7/28 Treatment officers (25%) show over-reliance on AI recommendations (≥75th pct threshold)
- **Skill-based effect:** Experienced officers benefit more (+20.56pp) than less experienced (+15.31pp) → Deploy to experienced officers first

---

## Analysis Structure

```
1단계 — Data Cleaning & Sanity Check
2단계 — OEC Calculation (F1, PrecisionAp, PrecisionRej)
3단계 — Normality & Variance Tests (Shapiro-Wilk, Levene's)
4단계 — Hypothesis Testing (t-test, Bonferroni, Cohen's d, 95% CI)
5단계 — Behavioral Analysis
6단계 — Business Impact (Financial Impact + Sensitivity Analysis)
```

---

## Stack

`Python` · `pandas` · `numpy` · `scipy` · `statsmodels` · `seaborn` · `matplotlib`

---

## How to Run

1. Upload `ADAproject_2025_data.xlsx` to your Google Drive
2. Open `AB_test_loan_analysis.ipynb` in Google Colab
3. Update `FILE_PATH` in cell 1-1 to match your Drive path
4. **Runtime → Run all**

> Note: The dataset is not included in this repository (proprietary course data).

---

## Files

| File | Description |
|------|-------------|
| `AB_test_loan_analysis.ipynb` | Full analysis notebook (69 cells) |
| `REPORT.md` | Final report with executive summary & recommendations |
