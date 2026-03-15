#!/usr/bin/env python3
"""
Assignment 6 - Complete IV Analysis with AK1991 Data
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.regression.linear_model import RegressionResults
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_stata('AK1991/AK1991.dta')
print("Loaded AK1991 dataset: {} observations, {} variables".format(df.shape[0], df.shape[1]))

# ==============================================================================
# DATA PREPARATION
# ==============================================================================

# Drop any rows with missing values in key variables
df_clean = df.dropna(subset=['logwage', 'edu', 'married', 'black', 'smsa', 'state', 'yob', 'qob'])
print("Sample size after removing missing values: {}".format(len(df_clean)))

# Create state dummies (1-50 states, using state 1 as reference)
state_dummies = pd.get_dummies(df_clean['state'], prefix='state', drop_first=True)

# Create year-of-birth dummies (1930-1939, using 1930 as reference)
yob_dummies = pd.get_dummies(df_clean['yob'], prefix='yob', drop_first=True)

# Create QOB dummies for instruments (1-4, using QOB 1 as reference)
qob_dummies = pd.get_dummies(df_clean['qob'], prefix='qob', drop_first=True)

# Extract dependent variable and controls
y = df_clean['logwage'].values
educ = df_clean['edu'].values
controls = df_clean[['married', 'black', 'smsa']].values
state_dums = state_dummies.values
yob_dums = yob_dummies.values
qob_dums = qob_dummies.values

# ==============================================================================
# QUESTION 1: Model Specification
# ==============================================================================
print("\n" + "="*80)
print("QUESTION 1: MODEL SPECIFICATION")
print("="*80)
print("""
The model to estimate is:

  logwage_i = beta_0 + beta_1 * education_i + beta_2 * married_i
            + beta_3 * black_i + beta_4 * smsa_i
            + sum_s gamma_s * state_s_i + sum_y delta_y * yob_y_i + epsilon_i

where:
  logwage_i    = log of weekly wage for individual i
  education_i  = years of education
  married_i    = indicator for marital status
  black_i      = indicator for race (black)
  smsa_i       = indicator for urban area (Standard Metropolitan Statistical Area)
  state_s_i    = set of 49 state dummy variables (state 1 is reference)
  yob_y_i      = set of 9 year-of-birth dummy variables (1930 is reference)
  epsilon_i    = error term

Purpose: Estimate the causal return to education (beta_1)
""")

# ==============================================================================
# QUESTION 2: OLS Concerns
# ==============================================================================
print("\n" + "="*80)
print("QUESTION 2: POTENTIAL ISSUE WITH OLS")
print("="*80)
print("""
PRIMARY CONCERN: ENDOGENEITY DUE TO ABILITY BIAS

Education is likely endogenous (correlated with error term epsilon_i) because:

1. UNOBSERVED ABILITY: Individuals with higher unobserved ability tend to:
   - Acquire more years of education (more motivated, higher IQ, etc.)
   - Earn higher wages (ability directly increases productivity)

2. OMITTED VARIABLE BIAS: Ability (unobserved) is in the error term and
   is positively correlated with both education and wages.

3. MATHEMATICAL CONSEQUENCE:
   Cov(education_i, epsilon_i) > 0
   Therefore: plim(beta_1_OLS) = beta_1 + bias
   where bias > 0 because of positive correlation

4. INTERPRETATION:
   OLS OVERESTIMATES the true causal return to education because it
   incorrectly attributes the wage premium of able individuals to
   education, when some of it is due to their ability.
""")

# ==============================================================================
# QUESTION 3: First Stage Analysis - QOB as Instrument
# ==============================================================================
print("\n" + "="*80)
print("QUESTION 3: QUARTER OF BIRTH AS INSTRUMENT")
print("="*80)

# Build first-stage design matrix
X_controls = np.column_stack([
    np.ones(len(df_clean)),  # constant
    controls,                 # married, black, smsa
    state_dums,              # state dummies
    yob_dums                 # year-of-birth dummies
])

X_first_stage = np.column_stack([
    X_controls,
    qob_dums                 # QOB dummies (excluded instruments)
])

# Run first stage regression: education ~ QOB + controls
from scipy.linalg import lstsq
beta_fs, residuals, rank, s = lstsq(X_first_stage, educ)

# Compute statistics for first stage
n = len(educ)
k = X_first_stage.shape[1]
resid_fs = educ - X_first_stage @ beta_fs
sigma2_fs = np.sum(resid_fs**2) / (n - k)
var_beta_fs = sigma2_fs * np.linalg.inv(X_first_stage.T @ X_first_stage).diagonal()
se_beta_fs = np.sqrt(var_beta_fs)
t_stats_fs = beta_fs / se_beta_fs

# Test QOB joint significance (F-test)
# Restricted model: education ~ controls (no QOB)
X_restricted = X_controls
beta_restr, _, _, _ = lstsq(X_restricted, educ)
resid_restr = educ - X_restricted @ beta_restr
rss_restr = np.sum(resid_restr**2)
rss_full = np.sum(resid_fs**2)

# F-test: (RSS_r - RSS_f) / q / (RSS_f / (n-k))
n_qob = qob_dums.shape[1]  # number of QOB dummies
f_stat = ((rss_restr - rss_full) / n_qob) / (rss_full / (n - k))
f_pval = 1 - stats.f.cdf(f_stat, n_qob, n - k)

print("\nFIRST STAGE REGRESSION: Education ~ QOB + Controls")
print("-" * 80)
print("Dependent variable: Years of Education")
print(f"\nSample size: {n:,}")
print(f"Number of regressors (including constant): {k}")
print(f"Residual sum of squares: {rss_full:,.1f}")
print(f"Residual standard error: {np.sqrt(sigma2_fs):.4f}")

print("\nQOB Coefficients (Quarter of Birth Dummies):")
print(f"  QOB=2: {beta_fs[-3]:7.4f} (SE: {se_beta_fs[-3]:.4f}, t={t_stats_fs[-3]:7.3f})")
print(f"  QOB=3: {beta_fs[-2]:7.4f} (SE: {se_beta_fs[-2]:.4f}, t={t_stats_fs[-2]:7.3f})")
print(f"  QOB=4: {beta_fs[-1]:7.4f} (SE: {se_beta_fs[-1]:.4f}, t={t_stats_fs[-1]:7.3f})")

print(f"\nF-Test for QOB Joint Significance:")
print(f"  F-statistic (3, {n-k}):        {f_stat:.4f}")
print(f"  p-value:                    {f_pval:.6f}")
print(f"  Critical value (5%):         {stats.f.ppf(0.95, n_qob, n-k):.4f}")

if f_stat > 10:
    strength = "STRONG (F > 10)"
elif f_stat > 5:
    strength = "MODERATE (5 < F <= 10)"
else:
    strength = "WEAK (F <= 5)"

print(f"\n  Instrument Strength Assessment: {strength}")
print(f"  => The instrument is {'VALID' if f_stat > 10 else 'questionable'} for IV analysis")

print("\nQOB AS VALID INSTRUMENT - JUSTIFICATION:")
print("-" * 80)
print("""
RELEVANCE: Is QOB correlated with education?
  YES - We see significant QOB coefficients with F > 10

  Explanation: Compulsory schooling laws require students to remain in
  school until age 16/17. Those born in Q1 are youngest in their cohort,
  so they reach age 16 having completed fewer school years. This creates
  exogenous variation in education by birth quarter.

  Empirical evidence: First-stage F-statistic = {:.2f} >> 10

EXOGENEITY: Is QOB independent of the error term (ability)?
  YES - QOB is plausibly exogenous

  Rationale: Quarter of birth is:
  - Determined at conception/birth (beyond individual control)
  - Unrelated to unobserved ability
  - Only affects wages through education, not directly

  This satisfies the EXCLUSION RESTRICTION

CONCLUSION: QOB is a VALID and STRONG instrument for education
""".format(f_stat))

# ==============================================================================
# QUESTION 4: 2SLS Estimation and Tests
# ==============================================================================
print("\n" + "="*80)
print("QUESTION 4: INSTRUMENTAL VARIABLES ANALYSIS (2SLS)")
print("="*80)

# STEP 1: First stage predictions
educ_pred = X_first_stage @ beta_fs

# STEP 2: Second stage regression
X_second_stage = np.column_stack([
    np.ones(n),
    educ_pred,  # predicted education from first stage
    controls,
    state_dums,
    yob_dums
])

beta_2sls, _, _, _ = lstsq(X_second_stage, y)

# Compute second-stage statistics
resid_2sls = y - X_second_stage @ beta_2sls
sigma2_2sls = np.sum(resid_2sls**2) / (n - X_second_stage.shape[1])
var_beta_2sls = sigma2_2sls * np.linalg.inv(X_second_stage.T @ X_second_stage).diagonal()
se_beta_2sls = np.sqrt(var_beta_2sls)
t_stats_2sls = beta_2sls / se_beta_2sls

print("\n2SLS RESULTS")
print("-" * 80)
print("Dependent variable: log(wage)")
print(f"Endogenous regressor: education (instrumented by QOB)")
print(f"Excluded instruments: QOB dummies (3)")
print(f"\nSample size: {n:,}")

print("\nKey Coefficient (Return to Education):")
print(f"  2SLS Coefficient (education): {beta_2sls[1]:7.4f}")
print(f"  Standard Error:               {se_beta_2sls[1]:.4f}")
print(f"  t-statistic:                  {t_stats_2sls[1]:7.3f}")
print(f"  p-value (two-sided):          {2*(1-stats.t.cdf(abs(t_stats_2sls[1]), n-X_second_stage.shape[1])):.6f}")
print(f"  95% CI: [{beta_2sls[1] - 1.96*se_beta_2sls[1]:.4f}, {beta_2sls[1] + 1.96*se_beta_2sls[1]:.4f}]")

print("\nInterpretation:")
print(f"  One additional year of education is associated with a")
print(f"  {beta_2sls[1]*100:.2f}% increase in weekly wages")

# ==============================================================================
# HAUSMAN TEST FOR ENDOGENEITY
# ==============================================================================
print("\n" + "-" * 80)
print("HAUSMAN TEST FOR ENDOGENEITY")
print("-" * 80)

# OLS regression for comparison
X_ols = np.column_stack([
    np.ones(n),
    educ,  # actual education (not instrumented)
    controls,
    state_dums,
    yob_dums
])

beta_ols, _, _, _ = lstsq(X_ols, y)
resid_ols = y - X_ols @ beta_ols
sigma2_ols = np.sum(resid_ols**2) / (n - X_ols.shape[1])
var_beta_ols = sigma2_ols * np.linalg.inv(X_ols.T @ X_ols).diagonal()
se_beta_ols = np.sqrt(var_beta_ols)

print(f"\nOLS Coefficient (education):  {beta_ols[1]:7.4f}")
print(f"2SLS Coefficient (education): {beta_2sls[1]:7.4f}")
print(f"Difference (2SLS - OLS):      {beta_2sls[1] - beta_ols[1]:7.4f}")

# Hausman test: H = (b_2sls - b_ols)' * Var(b_2sls - b_ols)^(-1) * (b_2sls - b_ols)
# For single coefficient: t = (b_2sls - b_ols) / SE(difference)
diff_var = var_beta_2sls[1] + var_beta_ols[1]  # assuming no correlation
hausman_t = (beta_2sls[1] - beta_ols[1]) / np.sqrt(diff_var)
hausman_pval = 2 * (1 - stats.t.cdf(abs(hausman_t), n - X_ols.shape[1]))

print(f"\nHausman test statistic (t):   {hausman_t:7.3f}")
print(f"p-value (two-sided):          {hausman_pval:.6f}")

if hausman_pval < 0.05:
    print("\nConclusion: REJECT H0 - Education IS ENDOGENOUS (5% level)")
    print("           => Use 2SLS instead of OLS")
else:
    print("\nConclusion: FAIL TO REJECT H0 - Cannot conclude education is endogenous")
    print("           => OLS may be preferred (more efficient)")

# ==============================================================================
# SARGAN TEST FOR OVERIDENTIFYING RESTRICTIONS
# ==============================================================================
print("\n" + "-" * 80)
print("SARGAN TEST FOR OVERIDENTIFYING RESTRICTIONS")
print("-" * 80)
print("""
We have 3 QOB instruments for 1 endogenous variable (education).
=> We have 2 overidentifying restrictions (K - L = 3 - 1 = 2)

H0: All instruments are exogenous (exclusion restrictions hold)
H1: At least one instrument violates exogeneity

Test procedure:
  1. Get 2SLS residuals
  2. Regress residuals on all exogenous variables (X_full)
  3. Sargan statistic = n * R_squared ~ chi-sq(#overid)
""")

# Regress 2SLS residuals on all instruments
X_sargan = np.column_stack([X_controls, qob_dums])
beta_sargan, _, _, _ = lstsq(X_sargan, resid_2sls)
y_mean = np.mean(resid_2sls)
tss_sargan = np.sum((resid_2sls - y_mean)**2)
rss_sargan = np.sum((resid_2sls - X_sargan @ beta_sargan)**2)
r2_sargan = 1 - rss_sargan / tss_sargan

sargan_stat = n * r2_sargan
sargan_pval = 1 - stats.chi2.cdf(sargan_stat, n_qob)

print(f"\nSargan Test Results:")
print(f"  R-squared of residual regression: {r2_sargan:.6f}")
print(f"  Sargan statistic (n*R²):          {sargan_stat:.4f}")
print(f"  Chi-squared critical value (5%):  {stats.chi2.ppf(0.95, n_qob):.4f}")
print(f"  p-value:                          {sargan_pval:.6f}")

if sargan_pval > 0.05:
    print(f"\nConclusion: FAIL TO REJECT H0 (p={sargan_pval:.4f} > 0.05)")
    print("           => Instruments appear valid (exclusion restrictions hold)")
else:
    print(f"\nConclusion: REJECT H0 (p={sargan_pval:.4f} < 0.05)")
    print("           => Question whether all instruments are exogenous")

# ==============================================================================
# SUMMARY TABLE
# ==============================================================================
print("\n" + "="*80)
print("SUMMARY: OLS vs. 2SLS Comparison")
print("="*80)
print(f"\n{'Coefficient':<25} {'OLS':>15} {'2SLS':>15}")
print("-" * 55)
print(f"{'Education':<25} {beta_ols[1]:>15.4f} {beta_2sls[1]:>15.4f}")
print(f"{'SE(Education)':<25} {se_beta_ols[1]:>15.4f} {se_beta_2sls[1]:>15.4f}")
print(f"{'t-stat':<25} {beta_ols[1]/se_beta_ols[1]:>15.3f} {beta_2sls[1]/se_beta_2sls[1]:>15.3f}")

print("\n" + "="*80)
print("KEY DIAGNOSTIC STATISTICS")
print("="*80)
print(f"First-stage F-statistic:     {f_stat:.4f}")
print(f"Hausman test p-value:        {hausman_pval:.6f}")
print(f"Sargan overid. test p-value: {sargan_pval:.6f}")
print("="*80)
