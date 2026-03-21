# cpi.py
# =============================================================================
# Cheating Probability Index (CPI) Computation
# =============================================================================
# This module takes the raw anomaly scores from all four algorithms,
# normalizes them to a 0-1 scale, applies weights, and produces a single
# CPI value per student along with a human-readable label.
#
# WEIGHT RATIONALE (risk-based weighting):
#   SVM  0.35 — copy/paste is a direct, deliberate action (strongest signal)
#   TAB  0.25 — tab switching is behavioral but has innocent explanations
#   RT   0.25 — response time is meaningful but ambiguous (fast ≠ cheating)
#   HMM  0.15 — requires baseline data; lowest weight due to data dependency
#   Ref: Corrigan-Gibbs et al. (2015), Chandola et al. (2009)
#
# THRESHOLD RATIONALE (quartile division):
#   Unlikely     0-24%   Q1 — bottom quarter of probability space
#   Possible    25-49%   Q2 — below midpoint
#   Likely      50-74%   Q3 — above midpoint (flagged)
#   Highly Likely 75-100% Q4 — top quarter (flagged)
#   Ref: Chandola et al. (2009), Likert (1932)
#
# NORMALIZATION METHOD (sigmoid inversion):
#   Isolation Forest and OC-SVM output decision_function scores where
#   more negative = more anomalous. Sigmoid inversion maps these to 0-1
#   where 1 = maximally suspicious: norm = 1 / (1 + exp(score))
#   HMM outputs a drop value (higher = more suspicious), clipped to 0-50
#   and divided by 50 to give 0-1.
# =============================================================================

import numpy as np
from constants import (
    CPI_WEIGHT_SVM,
    CPI_WEIGHT_ISO_TAB,
    CPI_WEIGHT_ISO_RT,
    CPI_WEIGHT_HMM,
    CPI_THRESHOLD_LIKELY,
    CPI_THRESHOLD_HIGHLY_LIKELY,
    CPI_THRESHOLD_POSSIBLE
)


def _sigmoid_norm(score):
    """
    Converts a decision_function score (negative = anomalous) to 0-1.
    Uses sigmoid inversion: norm = 1 / (1 + exp(score))
    - A very negative score (e.g. -2.0) → close to 1.0 (suspicious)
    - A positive score  (e.g. +1.0) → close to 0.27 (normal)
    - Score of 0.0 → exactly 0.5 (boundary)
    """
    return float(1.0 / (1.0 + np.exp(score)))


def _hmm_norm(drop):
    """
    Converts an HMM drop value (higher = more suspicious) to 0-1.
    Clips to a max of 50 (empirically reasonable upper bound for log-likelihood drop).
    drop=0    → 0.0 (no change from baseline, normal)
    drop=50+  → 1.0 (maximally different from baseline, suspicious)
    """
    return float(min(max(drop, 0.0), 50.0) / 50.0)


def compute_cpi(row):
    """
    Computes the Cheating Probability Index for one student row.

    Expected columns in row:
      iso_tab_score, rt_score, svm_score, hmm_score

    Returns a dict with:
      cpi_raw      : float 0.0-1.0
      cpi_score    : float 0.0-100.0 (percentage)
      cpi_label    : str  (Unlikely / Possible / Likely / Highly Likely)
      is_flagged   : bool (True if cpi_score >= CPI_THRESHOLD_LIKELY)

    If a score column is missing or NaN, that algorithm contributes 0
    (neutral — does not penalize the student).
    """

    # Step 1: Normalize each raw score to 0-1
    iso_tab_raw = row.get('iso_tab_score', 0.0)
    rt_raw      = row.get('rt_score',      0.0)
    svm_raw     = row.get('svm_score',     0.0)
    hmm_raw     = row.get('hmm_score',     0.0)

    # Replace NaN with 0 (neutral)
    iso_tab_raw = 0.0 if (iso_tab_raw is None or np.isnan(iso_tab_raw)) else iso_tab_raw
    rt_raw      = 0.0 if (rt_raw      is None or np.isnan(rt_raw))      else rt_raw
    svm_raw     = 0.0 if (svm_raw     is None or np.isnan(svm_raw))     else svm_raw
    hmm_raw     = 0.0 if (hmm_raw     is None or np.isnan(hmm_raw))     else hmm_raw

    iso_tab_norm = _sigmoid_norm(iso_tab_raw)
    rt_norm      = _sigmoid_norm(rt_raw)
    svm_norm     = _sigmoid_norm(svm_raw)
    hmm_norm     = _hmm_norm(hmm_raw)

    # Step 2: Apply weighted formula
    # CPI = (w_svm × svm_norm) + (w_tab × tab_norm) + (w_rt × rt_norm) + (w_hmm × hmm_norm)
    cpi_raw = (
        (CPI_WEIGHT_SVM     * svm_norm)     +
        (CPI_WEIGHT_ISO_TAB * iso_tab_norm) +
        (CPI_WEIGHT_ISO_RT  * rt_norm)      +
        (CPI_WEIGHT_HMM     * hmm_norm)
    )

    # Step 3: Convert to percentage
    cpi_score = round(cpi_raw * 100, 2)

    # Step 4: Assign label based on quartile thresholds
    if cpi_score >= CPI_THRESHOLD_HIGHLY_LIKELY:
        cpi_label  = 'Highly Likely'
        is_flagged = True
    elif cpi_score >= CPI_THRESHOLD_LIKELY:
        cpi_label  = 'Likely'
        is_flagged = True
    elif cpi_score >= CPI_THRESHOLD_POSSIBLE:
        cpi_label  = 'Possible'
        is_flagged = False
    else:
        cpi_label  = 'Unlikely'
        is_flagged = False

    return {
        'cpi_raw':     cpi_raw,
        'cpi_score':   cpi_score,
        'cpi_label':   cpi_label,
        'is_flagged':  is_flagged
    }


def apply_cpi_to_results(scored_df):
    """
    Applies compute_cpi() to every student row in scored_df.
    Adds cpi_score, cpi_label, and is_flagged columns.
    Returns the updated DataFrame.
    """
    cpi_scores  = []
    cpi_labels  = []
    is_flagged  = []

    for _, row in scored_df.iterrows():
        result = compute_cpi(row)
        cpi_scores.append(result['cpi_score'])
        cpi_labels.append(result['cpi_label'])
        is_flagged.append(result['is_flagged'])

    scored_df['cpi_score']  = cpi_scores
    scored_df['cpi_label']  = cpi_labels
    scored_df['is_flagged'] = is_flagged

    flagged_count = sum(is_flagged)
    print(f'[CPI] Computed for {len(scored_df)} students. '
          f'Flagged: {flagged_count} (CPI >= {CPI_THRESHOLD_LIKELY}%)')

    return scored_df