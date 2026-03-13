# synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import random


def generate_training_data(n_students=200, seed=42):
    """
    Generates a feature DataFrame of n_students normal students.
    Used to train Isolation Forest (Tab), Isolation Forest (RT), and One-Class SVM.

    Normal behavior:
      - copy_count  = 0  (normal students do not copy during exams)
      - paste_count = 0  (normal students do not paste during exams)
      - cut_count   = 0  (normal students do not cut during exams)
      - shortcut_count = 0-3 (realistic: Ctrl+Z undo, Ctrl+A select all, etc.)

    Giving shortcut_count a small realistic range is important — it gives
    One-Class SVM the variance it needs to learn what normal behavior looks like.
    An all-zero training set gives the model no variance to learn from, which
    causes it to produce a degenerate boundary.
    """
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for i in range(n_students):
        rows.append({
            'student_id':        f'TRAIN{i:04d}',
            # Tab features — normal: 0-2 switches, short duration
            'tab_switch_count':  random.randint(0, 2),
            'avg_duration_away': random.uniform(0, 5),
            'max_duration_away': random.uniform(0, 10),
            # Copy/paste — normal: copy/paste/cut are always 0
            # shortcut_count has a small realistic range (Ctrl+Z, Ctrl+A, etc.)
            'copy_count':        0,
            'paste_count':       0,
            'cut_count':         0,
            'shortcut_count':    random.randint(0, 3),
            # Response time — normal: 30-180 seconds per question
            'avg_rt': random.uniform(30, 180),
            'min_rt': random.uniform(10, 30),
            'max_rt': random.uniform(180, 300),
            'std_rt': random.uniform(10, 40),
            # Keystroke — normal: 150-250ms average IKI
            'mean_iki': random.uniform(150, 250),
            'std_iki':  random.uniform(20, 60),
            '_iki_sequence': list(np.random.normal(200, 40, 100).clip(30, 5000))
        })
    df = pd.DataFrame(rows)
    print(f'[SyntheticData] Generated {len(df)} training rows (all normal)')
    return df


def generate_test_data(n_normal=50, n_suspicious=50, seed=99):
    """
    Generates a labeled test dataset.
    Returns: (features_df, true_labels)
      features_df : same structure as training data
      true_labels : list of 0 (normal) or 1 (suspicious) per student

    Used only for accuracy evaluation — NOT for training.
    """
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    labels = []

    # Normal students (label = 0) — same distribution as training
    for i in range(n_normal):
        rows.append({
            'student_id':        f'TEST_N{i:04d}',
            'tab_switch_count':  random.randint(0, 2),
            'avg_duration_away': random.uniform(0, 5),
            'max_duration_away': random.uniform(0, 10),
            'copy_count':        0,
            'paste_count':       0,
            'cut_count':         0,
            'shortcut_count':    random.randint(0, 3),
            'avg_rt': random.uniform(30, 180),
            'min_rt': random.uniform(10, 30),
            'max_rt': random.uniform(180, 300),
            'std_rt': random.uniform(10, 40),
            'mean_iki': random.uniform(150, 250),
            'std_iki':  random.uniform(20, 60),
            '_iki_sequence': list(np.random.normal(200, 40, 100).clip(30, 5000))
        })
        labels.append(0)

    # Suspicious students (label = 1) — clearly abnormal copy/paste behavior
    for i in range(n_suspicious):
        rows.append({
            'student_id':        f'TEST_S{i:04d}',
            'tab_switch_count':  random.randint(5, 20),
            'avg_duration_away': random.uniform(30, 120),
            'max_duration_away': random.uniform(120, 300),
            'copy_count':        random.randint(2, 8),
            'paste_count':       random.randint(2, 6),
            'cut_count':         random.randint(1, 4),
            'shortcut_count':    random.randint(5, 15),
            'avg_rt': random.uniform(2, 10),
            'min_rt': random.uniform(1, 3),
            'max_rt': random.uniform(10, 20),
            'std_rt': random.uniform(1, 5),
            'mean_iki': random.uniform(50, 100),
            'std_iki':  random.uniform(80, 150),
            '_iki_sequence': list(np.random.normal(75, 100, 100).clip(30, 5000))
        })
        labels.append(1)

    df = pd.DataFrame(rows)
    print(f'[SyntheticData] Generated test set: {n_normal} normal + {n_suspicious} suspicious')
    return df, labels