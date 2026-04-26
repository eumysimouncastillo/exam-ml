# cpi.py
# =============================================================================
# Cheating Probability Index (CPI) Computation
# =============================================================================
# This module takes the raw anomaly scores from all four algorithms,
# normalizes them to a 0-1 scale, applies weights, and produces a single
# CPI value per student along with a human-readable label.
#
# WEIGHT RATIONALE (risk-based weighting):
#   SVM  0.40 — copy/paste is a direct, deliberate action (strongest signal)
#   TAB  0.30 — tab switching is behavioral but has innocent explanations
#   RT   0.15 — response time is meaningful but ambiguous (fast ≠ cheating)
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
# NORMALIZATION METHOD:
#   Two separate normalizers are used depending on the algorithm's output
#   characteristics:
#
#   OC-SVM → _sigmoid_norm (sigmoid inversion + boundary rescaling)
#     OC-SVM's decision_function has unbounded magnitude — a score of -7.6
#     means far more anomalous than -0.5. Sigmoid inversion maps this to 0-1,
#     then rescaled so the decision boundary (score=0) maps to exactly 0.0.
#     Formula: max(0, (sigmoid(score) - 0.5) * 2.0)
#
#   Isolation Forest (Tab, RT) → _iso_norm (linear mapping)
#     Isolation Forest's decision_function is bounded in a narrow range
#     regardless of how extreme the anomaly is — this is a documented
#     property of the algorithm (Liu et al., 2008). Sigmoid normalization
#     suppresses this signal. A linear mapping from the effective IF range
#     [-0.5, 0.0] to [1.0, 0.0] preserves relative anomaly magnitude and
#     correctly anchors normal behavior at zero contribution.
#     Formula: max(0, min(1, score / -0.5))
#
#   HMM → _hmm_norm (ratio-based drop, clipped to [0, 2])
#     HMM outputs a drop value (higher = more suspicious), clipped to 0-2
#     and divided by 2 to give 0-1.
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
    For OC-SVM only. OC-SVM decision_function has meaningful unbounded
    magnitude (e.g. -7.6 = very anomalous, -0.1 = mildly anomalous).

    Step 1 — Sigmoid inversion: sigmoid = 1 / (1 + exp(score))
      - Very negative score (e.g. -7.6) → close to 1.0 (suspicious)
      - Score of 0.0 (decision boundary)  → exactly 0.5
      - Positive score  (e.g. +1.0)       → below 0.5 (normal)

    Step 2 — Boundary rescaling: norm = max(0, (sigmoid - 0.5) * 2.0)
      - Shifts so the decision boundary (sigmoid=0.5) maps to 0.0.
      - A student with no anomalous SVM behavior contributes 0 to CPI.
      - Score of 0.0 (boundary) → 0.0 (no contribution)
      - Score of -3.0           → 0.90 (high contribution)
      - Score of -7.6           → ~1.0 (maximum contribution)
      - Positive scores         → 0.0  (clamped — normal behavior)
    """
    sigmoid = 1.0 / (1.0 + np.exp(score))
    return float(max(0.0, (sigmoid - 0.5) * 2.0))


def _iso_tab_severity(is_flagged, tab_switch_count, avg_duration_away, max_duration_away):
    """
    For Isolation Forest (Tab) only.
    IF-Tab detects whether behavior is anomalous (binary). Severity is then
    computed from raw feature exceedances above the normal boundary (0-3).
    Equal weighting across all three dimensions: frequency, typical duration,
    worst-case duration.
    Ceilings: 20 switches, 120s avg, 300s max → severity = 1.0
    Ref: Liu et al. (2008) — IF decision_function has bounded magnitude;
    severity grading from raw features preserves detection-scoring separation.
    """
    if not is_flagged:
        return 0.0
    count_excess = max(0.0, (tab_switch_count  - 3) / 17)
    avg_excess   = max(0.0, (avg_duration_away - 3) / 117)
    max_excess   = max(0.0, (max_duration_away - 3) / 297)
    severity     = (count_excess + avg_excess + max_excess) / 3.0
    return float(min(1.0, severity))


def _iso_norm(score):
    """
    For Isolation Forest (RT) only.
    Linear mapping from IF decision_function range [-0.5, 0.0] to [1.0, 0.0].
    score =  0.0  → 0.0 (boundary, no contribution)
    score = -0.5  → 1.0 (maximum anomaly)
    Ref: Liu et al. (2008).
    """
    return float(max(0.0, min(1.0, score / -0.5)))


def _hmm_norm(drop):
    """
    Converts the HMM combined drop score to 0-1.
    drop = max(per_step_llr_drop, ratio_shift)
    ratio_shift = (max/min of mean IKIs) - 1.0
    drop=0   → 0.0  (exam rhythm matches baseline)
    drop=2+  → 1.0  (maximally different — typing twice as fast/slow)
    Threshold for hmm_flagged is 0.8 (set in models_ml.py).
    """
    return float(min(max(drop, 0.0), 2.0) / 2.0)


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

    # IF-Tab and IF-RT use linear mapping (_iso_norm)
    # OC-SVM uses sigmoid inversion + boundary rescaling (_sigmoid_norm)
    # HMM uses ratio-based drop normalization (_hmm_norm)
    iso_tab_norm = _iso_tab_severity(
        row.get('iso_tab_flagged', False),
        row.get('tab_switch_count', 0.0),
        row.get('avg_duration_away', 0.0),
        row.get('max_duration_away', 0.0)
    )
    rt_norm      = _iso_norm(rt_raw)
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

    print(f"--- DEBUG CPI CALCULATION ---")
    print(f"Weights: SVM={CPI_WEIGHT_SVM}, TAB={CPI_WEIGHT_ISO_TAB}, RT={CPI_WEIGHT_ISO_RT}, HMM={CPI_WEIGHT_HMM}")
    print(f"Norms:   SVM={svm_norm:.2f}, TAB={iso_tab_norm:.2f}, RT={rt_norm:.2f}, HMM={hmm_norm:.2f}")
    print(f"Final Score: {cpi_score}%")
    print(f"-----------------------------")

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