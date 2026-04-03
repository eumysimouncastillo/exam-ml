# synthetic_data.py
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
import random


def generate_training_data(n_students=200, seed=42):
    """
    Generates a feature DataFrame of n_students normal students.
    Used to train Isolation Forest (Tab), Isolation Forest (RT), and One-Class SVM.

    Normal behavior thresholds:
      Tab switching:
        - tab_switch_count  : 0-3  (0 is normal, occasional accidental switches ok)
        - avg_duration_away : 0-3 seconds
        - max_duration_away : 0-3 seconds

      Copy/paste (exam context):
        - copy_count  : always 0 (copying during exam is not normal)
        - paste_count : always 0 (pasting during exam is not normal)
        - cut_count   : always 0 (cutting during exam is not normal)
        - shortcut_count : 0-3 (Ctrl+Z undo, Ctrl+A select all, etc. are normal)

      Response time:
        - avg_rt : 30-180 seconds per question
        - min_rt : 10-30 seconds
        - max_rt : 180-300 seconds
        - std_rt : 10-40 seconds

      Keystroke dynamics:
        - mean_iki : 150-250ms
        - std_iki  : 20-60ms
    """
    random.seed(seed)
    np.random.seed(seed)
    rows = []
    for i in range(n_students):
        rows.append({
            'student_id':        f'TRAIN{i:04d}',
            # Tab features — normal: 0-3 switches, duration <= 3 seconds
            # If tab_switch_count is 0, durations must also be 0
            # This matches real behavior — no switches means no duration data
            # Distribute evenly so model learns full 0-3 range as normal
            'tab_switch_count':  (tab_count := random.choices([0, 1, 2, 3], weights=[40, 30, 20, 10])[0]),
            'avg_duration_away': random.uniform(0, 3) if tab_count > 0 else 0.0,
            'max_duration_away': random.uniform(0, 3) if tab_count > 0 else 0.0,
            # Copy/paste — normal: all zero during exam
            # shortcut_count has small realistic range (Ctrl+Z, Ctrl+A, etc.)
            'copy_count':        0,
            'paste_count':       0,
            'cut_count':         0,
            # Weighted towards 0 — most students use no shortcuts during exams
            'shortcut_count':    random.choices([0, 1, 2, 3], weights=[60, 20, 15, 5])[0],
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

    Suspicious students are split into:
      70% clearly suspicious  — high copy/paste, easily caught by SVM
      30% borderline          — subtle behavior, may be missed by SVM
    This mix produces a realistic ~90% accuracy instead of 100%.

    Used only for accuracy evaluation — NOT for training.
    """
    random.seed(seed)
    np.random.seed(seed)
    rows   = []
    labels = []

    # ── Normal students (label = 0) ───────────────────────────────────────────
    for i in range(n_normal):
        rows.append({
            'student_id':        f'TEST_N{i:04d}',
            'tab_switch_count':  (tab_count := random.randint(0, 3)),
            'avg_duration_away': random.uniform(0, 3) if tab_count > 0 else 0.0,
            'max_duration_away': random.uniform(0, 3) if tab_count > 0 else 0.0,
            'copy_count':        0,
            'paste_count':       0,
            'cut_count':         0,
            # Weighted towards 0 — most students use no shortcuts during exams
            'shortcut_count':    random.choices([0, 1, 2, 3], weights=[60, 20, 15, 5])[0],
            'avg_rt': random.uniform(30, 180),
            'min_rt': random.uniform(10, 30),
            'max_rt': random.uniform(180, 300),
            'std_rt': random.uniform(10, 40),
            'mean_iki': random.uniform(150, 250),
            'std_iki':  random.uniform(20, 60),
            '_iki_sequence': list(np.random.normal(200, 40, 100).clip(30, 5000))
        })
        labels.append(0)

    # ── Suspicious students (label = 1) ───────────────────────────────────────
    # 70% clearly suspicious, 30% borderline
    n_clear      = int(n_suspicious * 0.70)
    n_borderline = n_suspicious - n_clear

    # Clearly suspicious — high copy/paste, long tab durations
    for i in range(n_clear):
        rows.append({
            'student_id':        f'TEST_S{i:04d}',
            'tab_switch_count':  random.randint(4, 20),
            'avg_duration_away': random.uniform(3.1, 120),
            'max_duration_away': random.uniform(3.1, 300),
            'copy_count':        random.randint(2, 8),
            'paste_count':       random.randint(2, 6),
            'cut_count':         random.randint(1, 4),
            'shortcut_count':    random.randint(4, 15),
            'avg_rt': random.uniform(2, 10),
            'min_rt': random.uniform(1, 3),
            'max_rt': random.uniform(10, 20),
            'std_rt': random.uniform(1, 5),
            'mean_iki': random.uniform(50, 100),
            'std_iki':  random.uniform(80, 150),
            '_iki_sequence': list(np.random.normal(75, 100, 100).clip(30, 5000))
        })
        labels.append(1)

    # Borderline suspicious — subtle behavior, may be missed by SVM
    for i in range(n_borderline):
        rows.append({
            'student_id':        f'TEST_SB{i:04d}',
            # Subtle tab switching — just above normal threshold
            'tab_switch_count':  random.randint(4, 6),
            'avg_duration_away': random.uniform(3.1, 10),
            'max_duration_away': random.uniform(3.1, 15),
            # Subtle copy/paste — low counts that might fool SVM
            'copy_count':        random.randint(0, 1),
            'paste_count':       random.randint(0, 1),
            'cut_count':         0,
            'shortcut_count':    random.randint(3, 5),
            # Response time slightly fast but not extreme
            'avg_rt': random.uniform(8, 25),
            'min_rt': random.uniform(3, 8),
            'max_rt': random.uniform(20, 60),
            'std_rt': random.uniform(4, 10),
            'mean_iki': random.uniform(100, 150),
            'std_iki':  random.uniform(60, 90),
            '_iki_sequence': list(np.random.normal(120, 80, 100).clip(30, 5000))
        })
        labels.append(1)

    df = pd.DataFrame(rows)
    print(f'[SyntheticData] Generated test set: {n_normal} normal + {n_suspicious} suspicious')
    return df, labels