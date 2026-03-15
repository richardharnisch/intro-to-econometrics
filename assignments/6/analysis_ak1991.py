#!/usr/bin/env python3
"""
Assignment 6 - AK1991 Instrumental Variables Analysis
Quarter of Birth as an Instrument for Education
"""

import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.sandbox.regression.gmm import IV2SLS
from statsmodels.stats.outliers_influence import variance_inflation_factor

# Load the data
print("Loading AK1991 data...")
df = pd.read_stata('AK1991/AK1991.dta')

# Display basic info
print(f"Dataset shape: {df.shape}")
print(f"\nFirst few rows:\n{df.head()}")
print(f"\nVariable names:\n{df.columns.tolist()}")
print(f"\nData types:\n{df.dtypes}")
print(f"\nBasic statistics:\n{df.describe()}")

# ==============================================================================
# QUESTION 1-3: Model specification and diagnostics
# ==============================================================================
print("\n" + "="*80)
print("QUESTION 1-3: Model specification and IV diagnostics")
print("="*80)

# Create necessary variables
# QOB: quarter of birth (1, 2, 3, 4)
# YOBS: year of birth (need to check what this is in the data)
# State dummies, year of birth dummies

# Check variable names for key variables
print("\nSearching for key variables...")
print("Variables containing 'wage':", [v for v in df.columns if 'wage' in v.lower()])
print("Variables containing 'educ':", [v for v in df.columns if 'educ' in v.lower()])
print("Variables containing 'qob' or 'quarter':", [v for v in df.columns if 'qob' in v.lower() or 'quarter' in v.lower()])
print("Variables containing 'year' or 'birth':", [v for v in df.columns if 'year' in v.lower() or 'birth' in v.lower()])

# Look at a sample of data to understand structure
print("\nSample of data:")
print(df.iloc[0:5, 0:15])

# ==============================================================================
# Question 1: Model specification
# ==============================================================================
print("\n" + "-"*80)
print("QUESTION 1: Model Specification")
print("-"*80)
print("""
Model to estimate:
log(wage_i) = beta_0 + beta_1 * education_i + beta_2 * married_i
            + beta_3 * black_i + beta_4 * urban_i
            + sum(gamma_s * state_s_i) + sum(delta_y * yob_y_i) + epsilon_i

where:
- education_i: years of education
- married_i: indicator for married
- black_i: indicator for black
- urban_i: indicator for urban/SMSA
- state_s_i: full set of state dummies (excluding one for reference)
- yob_y_i: full set of year-of-birth dummies (excluding one for reference)
""")

# ==============================================================================
# Question 2: OLS concern
# ==============================================================================
print("\n" + "-"*80)
print("QUESTION 2: Potential issue with OLS")
print("-"*80)
print("""
ENDOGENEITY / ABILITY BIAS

Education is likely endogenous in the wage equation:
- Individuals with higher unobserved ability tend to:
  (1) Acquire more education
  (2) Earn higher wages

This means Cov(education_i, epsilon_i) != 0, violating OLS exogeneity.

Result: OLS coefficient on education is biased upward (ability bias).
The true causal return to education is less than what OLS estimates.
""")

# ==============================================================================
# Question 3: First Stage Analysis - QOB as instrument
# ==============================================================================
print("\n" + "-"*80)
print("QUESTION 3: Quarter of Birth as Instrument")
print("-"*80)

# Check what variables are available
print("\nData shape:", df.shape)
print("\nColumn names (first 30):")
for i, col in enumerate(df.columns[:30]):
    print(f"  {i}: {col}")

# Prepare data for analysis
df_clean = df.dropna(subset=['lwklywge', 'educ', 'married', 'black', 'smsa', 'yob', 'qob'])
print(f"\nSample size after dropping missing values: {len(df_clean)}")

# Create dummies for year of birth and quarter of birth
yob_dummies = pd.get_dummies(df_clean['yob'], prefix='yob', drop_first=True)
qob_dummies = pd.get_dummies(df_clean['qob'], prefix='qob', drop_first=True)

# Get state dummies if available
state_cols = [col for col in df_clean.columns if col.startswith('state')]
if state_cols:
    print(f"Found {len(state_cols)} state dummy variables")
    state_data = df_clean[state_cols].astype(int)
else:
    print("No state dummy variables found, checking for state codes...")
    if 'state' in df_clean.columns:
        state_data = pd.get_dummies(df_clean['state'], prefix='state', drop_first=True)
        print(f"Created {state_data.shape[1]} state dummies")
    else:
        state_data = pd.DataFrame(index=df_clean.index)
        print("No state information found")

# Combine all data
X_base = df_clean[['married', 'black', 'smsa']].astype(float)
X_all = pd.concat([X_base, yob_dummies, state_data], axis=1)

print(f"\nRegressor matrix size: {X_all.shape}")
print(f"Columns: {X_all.columns.tolist()[:20]}...")

# First stage: education on QOB and other variables
print("\n" + "="*80)
print("FIRST STAGE REGRESSION: Education on QOB and controls")
print("="*80)

X_first_stage = pd.concat([qob_dummies, X_all], axis=1)
X_first_stage = sm.add_constant(X_first_stage)

first_stage = sm.OLS(df_clean['educ'], X_first_stage).fit()
print(first_stage.summary())

# Extract QOB coefficients and test relevance
qob_cols = [col for col in X_first_stage.columns if col.startswith('qob')]
print(f"\nQOB variables: {qob_cols}")
print(f"QOB coefficients:\n{first_stage.params[qob_cols]}")

# F-test for QOB joint significance
qob_indices = [X_first_stage.columns.get_loc(col) for col in qob_cols]
f_stat = first_stage.f_test([['qob_2', 'qob_3', 'qob_4']])
print(f"\nF-test for QOB joint significance:")
print(f"  F-statistic: {f_stat.fvalue[0,0]:.4f}")
print(f"  p-value: {f_stat.pvalue:.6f}")

if f_stat.fvalue[0,0] > 10:
    print("  => STRONG instrument (F > 10)")
else:
    print("  => WEAK instrument (F <= 10)")

print("\n" + "-"*80)
print("ASSESSMENT OF QOB AS INSTRUMENT:")
print("-"*80)
print("""
RELEVANCE:
- QOB is significantly related to education (compulsory schooling laws)
- Children born in Q1 are youngest in cohort, reach age 16 with fewer school years
- This creates exogenous variation in education by birth quarter

EXOGENEITY:
- Season of birth is plausibly exogenous to ability and other determinants of wages
- QOB should only affect wages through its effect on education
- This satisfies the exclusion restriction

=> QOB is a VALID INSTRUMENT for education
""")
