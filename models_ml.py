# models_ml.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from hmmlearn import hmm
from constants import (
    MODELS_DIR,
    ISO_TAB_PATH, ISO_TAB_FEATURES,
    ISO_RT_PATH,  ISO_RT_FEATURES,
    OC_SVM_PATH,  OC_SVM_FEATURES
)

HMM_DIR = os.path.join(MODELS_DIR, 'hmm')


# ── Isolation Forest: Tab Switching ──────────────────────────────────────────
def train_isolation_forest_tab(features_df):
    """
    Trains on tab switching features.
    contamination=0.05: assumes up to 5% of training rows may be unusual.
    """
    X = features_df[ISO_TAB_FEATURES].values
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, ISO_TAB_PATH)
    print(f'[IsoForest-Tab] Trained on {len(X)} rows. Saved to {ISO_TAB_PATH}')
    return model


def score_isolation_forest_tab(features_df):
    """
    iso_tab_score:   more negative = more anomalous tab behavior
    iso_tab_flagged: True = anomaly detected
    """
    model = joblib.load(ISO_TAB_PATH)
    X = features_df[ISO_TAB_FEATURES].values
    features_df['iso_tab_score']   = model.decision_function(X)
    features_df['iso_tab_flagged'] = model.predict(X) == -1
    return features_df


# ── Isolation Forest: Response Time ──────────────────────────────────────────
def train_isolation_forest_rt(features_df):
    """
    Trains on response time features.
    Replaces Z-Score — this is a proper ML algorithm with a training phase,
    a saved model file, and a predict step.
    """
    X = features_df[ISO_RT_FEATURES].values
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, ISO_RT_PATH)
    print(f'[IsoForest-RT] Trained on {len(X)} rows. Saved to {ISO_RT_PATH}')
    return model


def score_isolation_forest_rt(features_df):
    """
    rt_score:   more negative = more anomalous response time
    rt_flagged: True = anomaly detected
    """
    model = joblib.load(ISO_RT_PATH)
    X = features_df[ISO_RT_FEATURES].values
    features_df['rt_score']   = model.decision_function(X)
    features_df['rt_flagged'] = model.predict(X) == -1
    return features_df


# ── One-Class SVM: Copy/Paste and Shortcuts ───────────────────────────────────
def train_oc_svm(features_df):
    """
    Trains on copy/paste features using raw (unscaled) counts.

    copy_count, paste_count, cut_count are 0 for normal students.
    shortcut_count has a small realistic range (0-3) for normal students.
    This gives the model enough variance to learn what normal looks like.

    Kernel: rbf with a fixed gamma value.
    gamma='scale' is avoided because it calculates gamma from feature variance,
    which is near-zero for these features and produces a degenerate boundary.
    A fixed gamma=1.0 gives the RBF kernel a stable, predictable radius.

    nu=0.05: up to 5% of training samples may be outside the boundary.
    """
    X = features_df[OC_SVM_FEATURES].values
    model = OneClassSVM(kernel='rbf', nu=0.02, gamma=1.0)
    model.fit(X)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, OC_SVM_PATH)
    print(f'[OC-SVM] Trained on {len(X)} rows. Saved to {OC_SVM_PATH}')
    return model


def score_oc_svm(features_df):
    """
    svm_score:   negative = outside the normal copy/paste boundary
    svm_flagged: True = anomaly detected
    """
    model = joblib.load(OC_SVM_PATH)
    X = features_df[OC_SVM_FEATURES].values
    features_df['svm_score']   = model.decision_function(X)
    features_df['svm_flagged'] = model.predict(X) == -1
    return features_df


# ── Hidden Markov Model: Keystroke Dynamics ───────────────────────────────────
def train_student_hmm(student_id, iki_sequence):
    if len(iki_sequence) < 50:
        print(f'[HMM] {student_id}: only {len(iki_sequence)} intervals, need 50+')
        return None
    X = np.array(iki_sequence).reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=3, covariance_type='diag',
                             n_iter=200, random_state=42)
    model.fit(X)
    baseline_score          = model.score(X)
    baseline_score_per_step = baseline_score / len(X)
    baseline_mean_iki       = float(np.mean(iki_sequence))   # ← NEW
    baseline_std_iki        = float(np.std(iki_sequence))    # ← NEW
    os.makedirs(HMM_DIR, exist_ok=True)
    path = os.path.join(HMM_DIR, f'{student_id}.pkl')
    joblib.dump({
        'model':                   model,
        'baseline_score':          baseline_score,
        'baseline_score_per_step': baseline_score_per_step,
        'baseline_iki_len':        len(iki_sequence),
        'baseline_mean_iki':       baseline_mean_iki,         # ← NEW
        'baseline_std_iki':        baseline_std_iki,          # ← NEW
    }, path)
    print(f'[HMM] Trained for {student_id}. '
          f'Baseline per-step: {baseline_score_per_step:.4f} | '
          f'Mean IKI: {baseline_mean_iki:.1f}ms | '
          f'Std IKI: {baseline_std_iki:.1f}ms | '
          f'IKI count: {len(iki_sequence)}')
    return model


def score_hmm(features_df):
    scores, flagged = [], []
    for _, row in features_df.iterrows():
        sid  = row['student_id']
        iki  = row['_iki_sequence']
        path = os.path.join(HMM_DIR, f'{sid}.pkl')

        if not os.path.exists(path) or len(iki) < 10:
            scores.append(0.0)
            flagged.append(False)
            continue

        saved = joblib.load(path)

        baseline_per_step = saved.get('baseline_score_per_step')
        baseline_mean_iki = saved.get('baseline_mean_iki')
        baseline_std_iki  = saved.get('baseline_std_iki')

        # Old .pkl missing new keys — student must retake baseline
        if baseline_per_step is None or baseline_mean_iki is None:
            print(f'[HMM] student={sid}: stale baseline — retake required')
            scores.append(0.0)
            flagged.append(False)
            continue

        try:
            iki_arr       = np.array(iki)
            exam_score    = saved['model'].score(iki_arr.reshape(-1, 1))
            exam_per_step = exam_score / len(iki)
            exam_mean_iki = float(np.mean(iki_arr))
            exam_std_iki  = float(np.std(iki_arr))

            # Signal 1: per-step log-likelihood deviation
            llr_drop = abs(baseline_per_step - exam_per_step)

            # Signal 2: ratio-based mean IKI shift
            # Measures speed drift independent of baseline variance
            ratio_shift = (max(exam_mean_iki, baseline_mean_iki) /
                           min(exam_mean_iki, baseline_mean_iki)) - 1.0

            # Use the stronger of the two signals
            drop = max(llr_drop, ratio_shift)

            print(f'[HMM Debug] student={sid} | '
                  f'baseline_per_step={baseline_per_step:.4f} | '
                  f'exam_per_step={exam_per_step:.4f} | '
                  f'llr_drop={llr_drop:.4f} | '
                  f'baseline_mean={baseline_mean_iki:.1f}ms | '
                  f'exam_mean={exam_mean_iki:.1f}ms | '
                  f'ratio_shift={ratio_shift:.4f} | '
                  f'final_drop={drop:.4f} | '
                  f'iki_len={len(iki)}')

            scores.append(float(drop))
            flagged.append(drop > 0.8)

        except Exception as e:
            print(f'[HMM] student={sid} scoring error: {e}')
            scores.append(0.0)
            flagged.append(False)

    features_df['hmm_score']   = scores
    features_df['hmm_flagged'] = flagged
    return features_df