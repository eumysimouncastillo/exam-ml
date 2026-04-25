# constants.py
import os

VALID_ACTION_TYPES = {
    'tab_switch', 'tab_return', 'copy', 'paste', 'cut',
    'keystroke', 'shortcut', 'question_open', 'question_submit'
}

ACTION_TYPE_MAP = {
    'tab_switch': 0, 'tab_return': 1, 'copy': 2, 'paste': 3, 'cut': 4,
    'keystroke': 5, 'shortcut': 6, 'question_open': 7, 'question_submit': 8
}

# Updated to accept any positive integer — matches users.id (bigint) in sect_app
# Previously was r'^[0-9]{7,10}$' which expected student number format
STUDENT_ID_REGEX = r'^[0-9]+$'

IKI_MIN_MS  = 30
IKI_MAX_MS  = 5000
MODELS_DIR  = 'models/'

# Isolation Forest — Tab Switching
ISO_TAB_PATH     = os.path.join(MODELS_DIR, 'iso_tab.pkl')
ISO_TAB_FEATURES = ['tab_switch_count', 'avg_duration_away', 'max_duration_away']

# Isolation Forest — Response Time
ISO_RT_PATH      = os.path.join(MODELS_DIR, 'iso_rt.pkl')
ISO_RT_FEATURES  = ['avg_rt', 'min_rt', 'max_rt', 'std_rt']

# One-Class SVM — Copy/Paste
OC_SVM_PATH      = os.path.join(MODELS_DIR, 'oc_svm.pkl')
OC_SVM_FEATURES  = ['copy_count', 'paste_count', 'cut_count', 'shortcut_count']

# Scaler
SCALER_PATH      = os.path.join(MODELS_DIR, 'scaler.pkl')

# Features normalized by MinMaxScaler.
# copy_count, paste_count, cut_count, shortcut_count excluded —
# they are always 0 in normal training data so scaling destroys the signal.
SCALE_FEATURES = [
    'tab_switch_count', 'avg_duration_away', 'max_duration_away',
    'avg_rt', 'min_rt', 'max_rt', 'std_rt',
    'mean_iki', 'std_iki'
]

# =============================================================================
# CPI — Cheating Probability Index
# =============================================================================
# WEIGHTS (risk-based, sum = 1.0):
#   SVM  0.40 — copy/paste is direct and deliberate
#               Ref: Corrigan-Gibbs et al. (2015), Alessio et al. (2017)
#   TAB  0.30 — behavioral but has innocent explanations
#               Ref: Cluskey et al. (2011)
#   RT   0.15 — meaningful but ambiguous
#               Ref: Wollack and Fremer (2013) Handbook of Test Security
#   HMM  0.15 — requires baseline data, weighted lower
#               Ref: Shepherd et al. (2018) IEEE
CPI_WEIGHT_SVM     = 0.40
CPI_WEIGHT_ISO_TAB = 0.30
CPI_WEIGHT_ISO_RT  = 0.15
CPI_WEIGHT_HMM     = 0.15

# THRESHOLDS (uniform quartile division):
#   Unlikely       0  - 24%   is_flagged = 0
#   Possible      25  - 49%   is_flagged = 0
#   Likely        50  - 74%   is_flagged = 1
#   Highly Likely 75  - 100%  is_flagged = 1
#   Ref: Chandola et al. (2009), Likert (1932)
CPI_THRESHOLD_POSSIBLE      = 25
CPI_THRESHOLD_LIKELY        = 50
CPI_THRESHOLD_HIGHLY_LIKELY = 75