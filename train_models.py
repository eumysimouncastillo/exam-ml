# train_models.py
# Run this script once to train all models on synthetic data.
# Usage: python train_models.py

import os
import numpy as np
from synthetic_data import generate_training_data
from features import normalize_features
from models_ml import (
    train_isolation_forest_tab,
    train_isolation_forest_rt,
    train_oc_svm,
    train_student_hmm
)
from constants import MODELS_DIR

def main():
    print('=== Training all models on synthetic data ===')
    print()

    # Step 1: Generate synthetic normal training data
    print('Step 1: Generating synthetic training data...')
    train_df = generate_training_data(n_students=200)

    # Step 2: Normalize features
    print()
    print('Step 2: Normalizing features...')
    norm_df = normalize_features(train_df.copy(), is_training=True)

    # Step 3: Train Isolation Forest — Tab Switching
    print()
    print('Step 3: Training Isolation Forest (Tab Switching)...')
    train_isolation_forest_tab(norm_df)

    # Step 4: Train Isolation Forest — Response Time
    print()
    print('Step 4: Training Isolation Forest (Response Time)...')
    train_isolation_forest_rt(norm_df)

    # Step 5: Train One-Class SVM
    print()
    print('Step 5: Training One-Class SVM (Copy/Paste)...')
    train_oc_svm(norm_df)

    # Step 6: Train HMM for each synthetic student using their IKI sequence
    print()
    print('Step 6: Training HMM models for each synthetic student...')
    hmm_trained = 0
    for _, row in train_df.iterrows():
        result = train_student_hmm(row['student_id'], row['_iki_sequence'])
        if result is not None:
            hmm_trained += 1
    print(f'[HMM] Trained {hmm_trained} student models')

    print()
    print('=== All models trained and saved to models/ folder ===')
    print(f'Files saved: {os.listdir(MODELS_DIR)}')

if __name__ == '__main__':
    main()
