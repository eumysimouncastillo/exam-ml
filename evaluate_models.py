# evaluate_models.py
# Measures accuracy of all four ML models using a labeled synthetic test set.
# Usage: python evaluate_models.py  (run AFTER train_models.py)

import numpy as np
import joblib
import os
from sklearn.metrics import accuracy_score
from synthetic_data import generate_test_data
from features import normalize_features
from constants import (
    ISO_TAB_PATH, ISO_TAB_FEATURES,
    ISO_RT_PATH,  ISO_RT_FEATURES,
    OC_SVM_PATH,  OC_SVM_FEATURES,
    MODELS_DIR
)

HMM_DIR = os.path.join(MODELS_DIR, 'hmm')


def evaluate_all():
    print('=== Evaluating model accuracy ===')
    print('Test set: 50 normal students + 50 suspicious students')
    print()

    # Generate labeled test data
    test_df, true_labels = generate_test_data(n_normal=50, n_suspicious=50)

    # Normalize using the SAVED scaler (not refitting)
    norm_df = normalize_features(test_df.copy(), is_training=False)

    # ── Isolation Forest: Tab ────────────────────────────────────────────
    model_tab = joblib.load(ISO_TAB_PATH)
    X_tab     = norm_df[ISO_TAB_FEATURES].values
    preds_tab = model_tab.predict(X_tab)   # 1 = normal, -1 = anomaly
    # Convert: model -1 (anomaly) → 1 (suspicious), model 1 (normal) → 0
    preds_tab_bin = [1 if p == -1 else 0 for p in preds_tab]
    acc_tab = accuracy_score(true_labels, preds_tab_bin)
    print(f'Isolation Forest (Tab Switching) Accuracy:  {acc_tab * 100:.1f}%')

    # ── Isolation Forest: RT ─────────────────────────────────────────────
    model_rt = joblib.load(ISO_RT_PATH)
    X_rt     = norm_df[ISO_RT_FEATURES].values
    preds_rt = model_rt.predict(X_rt)
    preds_rt_bin = [1 if p == -1 else 0 for p in preds_rt]
    acc_rt = accuracy_score(true_labels, preds_rt_bin)
    print(f'Isolation Forest (Response Time) Accuracy:  {acc_rt * 100:.1f}%')

    # ── One-Class SVM ────────────────────────────────────────────────────
    model_svm = joblib.load(OC_SVM_PATH)
    X_svm     = norm_df[OC_SVM_FEATURES].values
    preds_svm = model_svm.predict(X_svm)
    preds_svm_bin = [1 if p == -1 else 0 for p in preds_svm]
    acc_svm = accuracy_score(true_labels, preds_svm_bin)
    print(f'One-Class SVM (Copy/Paste)       Accuracy:  {acc_svm * 100:.1f}%')

    # ── HMM ──────────────────────────────────────────────────────────────
    # HMM needs per-student baseline models.
    # For synthetic evaluation: use the training students' IKI sequences
    # as 'baselines'. Test students' sequences are compared against them.
    # This simulates what happens when a student's typing rhythm changes.
    hmm_preds = []
    for _, row in test_df.iterrows():
        sid  = row['student_id']
        iki  = row['_iki_sequence']
        path = os.path.join(HMM_DIR, f'{sid}.pkl')
        # For test students without their own model, use a reference model
        # (TRAIN0000) as the baseline — this simulates a different typist
        if not os.path.exists(path):
            path = os.path.join(HMM_DIR, 'TRAIN0000.pkl')
        if not os.path.exists(path):
            hmm_preds.append(0); continue
        saved    = joblib.load(path)
        baseline = saved['baseline_score']
        try:
            score = saved['model'].score(np.array(iki).reshape(-1, 1))
            drop  = baseline - score
            hmm_preds.append(1 if drop > 20 else 0)
        except Exception:
            hmm_preds.append(0)
    acc_hmm = accuracy_score(true_labels, hmm_preds)
    print(f'Hidden Markov Model (Keystroke)  Accuracy:  {acc_hmm * 100:.1f}%')

    # ── Overall ──────────────────────────────────────────────────────────
    print()
    overall = (acc_tab + acc_rt + acc_svm + acc_hmm) / 4
    print(f'Average accuracy across all models: {overall * 100:.1f}%')
    print()
    print('NOTE: These scores are from synthetic data.')
    print('Retrain and re-evaluate with real exam data for final thesis results.')

if __name__ == '__main__':
    evaluate_all()
