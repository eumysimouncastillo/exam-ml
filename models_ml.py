# models_ml.py
import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from hmmlearn import hmm
from constants import MODELS_DIR, ZSCORE_THRESHOLD, ZSCORE_MIN_STUDENTS

ISO_PATH     = os.path.join(MODELS_DIR, 'isolation_forest.pkl')
SVM_PATH     = os.path.join(MODELS_DIR, 'oc_svm.pkl')
HMM_DIR      = os.path.join(MODELS_DIR, 'hmm')
ISO_FEATURES = ['tab_switch_count', 'avg_duration_away', 'max_duration_away']
SVM_FEATURES = ['copy_count', 'paste_count', 'cut_count', 'shortcut_count']


# ── Isolation Forest ─────────────────────────────────────────────────────
def train_isolation_forest(features_df):
    '''
    Trains on tab switching features.
    Use only clean, non-suspicious sessions as training data.
    contamination=0.05 means: expect about 5% of training rows to be anomalies.
    Lower this if your training data is very clean.
    '''
    X = features_df[ISO_FEATURES].values
    model = IsolationForest(n_estimators=200, contamination=0.05, random_state=42)
    model.fit(X)
    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, ISO_PATH)
    print(f'[IsoForest] Trained on {len(X)} rows. Saved to {ISO_PATH}')
    return model


def score_isolation_forest(features_df):
    '''
    Scores each student.
    iso_score: more negative = more anomalous
    iso_flagged: True = anomaly detected
    '''
    model = joblib.load(ISO_PATH)
    X = features_df[ISO_FEATURES].values
    features_df['iso_score']   = model.decision_function(X)
    features_df['iso_flagged'] = model.predict(X) == -1
    return features_df


# ── One-Class SVM ────────────────────────────────────────────────────────
def train_oc_svm(features_df):
    '''
    Trains on copy/paste features.
    IMPORTANT: Training data must only contain sessions where copy/paste counts
    are zero. If you include sessions where students copied, the model will
    learn that is normal and will not flag it.
    nu=0.01 = very strict boundary (any copy/paste flags immediately).
    '''
    X = features_df[SVM_FEATURES].values
    model = OneClassSVM(kernel='rbf', nu=0.01, gamma='scale')
    model.fit(X)
    joblib.dump(model, SVM_PATH)
    print(f'[OC-SVM] Trained on {len(X)} rows. Saved to {SVM_PATH}')
    return model


def score_oc_svm(features_df):
    model = joblib.load(SVM_PATH)
    X = features_df[SVM_FEATURES].values
    features_df['svm_score']   = model.decision_function(X)
    features_df['svm_flagged'] = model.predict(X) == -1
    return features_df


# ── Z-Score ───────────────────────────────────────────────────────────────
def score_zscore(features_df):
    '''
    No trained model needed. Computes the mean and std of response times
    across all students in the same exam session, then flags outliers.
    Skips flagging if there are too few students for a reliable calculation.
    '''
    if len(features_df) < ZSCORE_MIN_STUDENTS:
        print(f'[Z-Score] Only {len(features_df)} students — below minimum {ZSCORE_MIN_STUDENTS}. Skipping.')
        features_df['z_score']   = 0.0
        features_df['z_flagged'] = False
        return features_df

    mean_rt = features_df['avg_rt'].mean()
    std_rt  = features_df['avg_rt'].std()

    if std_rt == 0 or pd.isna(std_rt):
        features_df['z_score']   = 0.0
        features_df['z_flagged'] = False
        return features_df

    features_df['z_score']   = (features_df['avg_rt'] - mean_rt) / std_rt
    features_df['z_flagged'] = features_df['z_score'].abs() > ZSCORE_THRESHOLD
    print(f'[Z-Score] Mean RT: {mean_rt:.1f}s  Std: {std_rt:.1f}s  Flagged: {features_df["z_flagged"].sum()}')
    return features_df


# ── Hidden Markov Model ───────────────────────────────────────────────────
def train_student_hmm(student_id, iki_sequence):
    '''
    Trains a personal HMM for one student using their baseline typing data.
    Call this during exam registration (e.g. typing exercise at login).
    Requires at least 50 keystroke intervals to be reliable.
    '''
    if len(iki_sequence) < 50:
        print(f'[HMM] {student_id}: Only {len(iki_sequence)} intervals — need 50+. Skipping.')
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
    '''
    Compares each student's exam typing rhythm to their personal baseline.
    Students without a saved baseline model are given a neutral score.
    A large drop in log-likelihood score indicates the typing pattern changed.
    '''
    hmm_scores  = []
    hmm_flagged = []

    for _, row in features_df.iterrows():
        sid = row['student_id']
        iki = row['_iki_sequence']
        path = os.path.join(HMM_DIR, f'{sid}.pkl')

        if not os.path.exists(path) or len(iki) < 10:
            hmm_scores.append(0.0)
            hmm_flagged.append(False)
            continue

        saved    = joblib.load(path)
        model    = saved['model']
        baseline = saved['baseline_score']
        try:
            exam_score = model.score(np.array(iki).reshape(-1, 1))
            drop = baseline - exam_score
            hmm_scores.append(float(drop))
            hmm_flagged.append(drop > 20)  # Calibrate this threshold with real data
        except Exception:
            hmm_scores.append(0.0)
            hmm_flagged.append(False)

    features_df['hmm_score']   = hmm_scores
    features_df['hmm_flagged'] = hmm_flagged
    return features_df
