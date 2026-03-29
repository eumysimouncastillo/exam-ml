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
    model = OneClassSVM(kernel='rbf', nu=0.01, gamma=1.0)
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
    """
    Trains a personal HMM for one student using their baseline typing.
    Call during exam registration (typing exercise at login).
    Requires at least 50 keystroke intervals to be reliable.
    """
    if len(iki_sequence) < 50:
        print(f'[HMM] {student_id}: only {len(iki_sequence)} intervals, need 50+')
        return None
    X = np.array(iki_sequence).reshape(-1, 1)
    model = hmm.GaussianHMM(n_components=3, covariance_type='diag',
                             n_iter=200, random_state=42)
    model.fit(X)
    baseline_score = model.score(X)
    os.makedirs(HMM_DIR, exist_ok=True)
    path = os.path.join(HMM_DIR, f'{student_id}.pkl')
    joblib.dump({'model': model, 'baseline_score': baseline_score}, path)
    print(f'[HMM] Trained for {student_id}. Baseline: {baseline_score:.2f}')
    return model


def score_hmm(features_df):
    """
    Compares exam typing rhythm to per-student baseline.
    Students without a saved baseline get a neutral score (not flagged).
    A large drop in log-likelihood indicates the typing pattern changed.
    """
    scores, flagged = [], []
    for _, row in features_df.iterrows():
        sid  = row['student_id']
        iki  = row['_iki_sequence']
        path = os.path.join(HMM_DIR, f'{sid}.pkl')
        if not os.path.exists(path) or len(iki) < 10:
            scores.append(0.0)
            flagged.append(False)
            continue
        saved    = joblib.load(path)
        baseline = saved['baseline_score']
        try:
            drop = baseline - saved['model'].score(np.array(iki).reshape(-1, 1))
            scores.append(float(drop))
            flagged.append(drop > 20)
        except Exception:
            scores.append(0.0)
            flagged.append(False)
    features_df['hmm_score']   = scores
    features_df['hmm_flagged'] = flagged
    return features_df