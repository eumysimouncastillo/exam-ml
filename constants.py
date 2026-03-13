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

STUDENT_ID_REGEX = r'^[0-9]{7,10}$'
IKI_MIN_MS        = 30
IKI_MAX_MS        = 5000
MODELS_DIR        = 'models/'

# Isolation Forest — Tab Switching
ISO_TAB_PATH     = os.path.join(MODELS_DIR, 'iso_tab.pkl')
ISO_TAB_FEATURES = ['tab_switch_count', 'avg_duration_away', 'max_duration_away']

# Isolation Forest — Response Time
ISO_RT_PATH      = os.path.join(MODELS_DIR, 'iso_rt.pkl')
ISO_RT_FEATURES  = ['avg_rt', 'min_rt', 'max_rt', 'std_rt']

# One-Class SVM — Copy/Paste
# These are intentionally NOT in SCALE_FEATURES below.
# Normal behavior is always 0 for these counts. Scaling them destroys the
# signal because MinMaxScaler has no range to work with (min=0, max=0 on
# training data), making every value — including suspicious ones — become 0.
# OC-SVM receives these as raw counts, which is the correct input.
OC_SVM_PATH      = os.path.join(MODELS_DIR, 'oc_svm.pkl')
OC_SVM_FEATURES  = ['copy_count', 'paste_count', 'cut_count', 'shortcut_count']

# Scaler
SCALER_PATH      = os.path.join(MODELS_DIR, 'scaler.pkl')

# Features that get normalized by MinMaxScaler.
# copy_count, paste_count, cut_count, shortcut_count are excluded because
# they are always 0 in normal data — scaling them removes the anomaly signal
# that One-Class SVM relies on to detect copy/paste behavior.
SCALE_FEATURES = [
    'tab_switch_count', 'avg_duration_away', 'max_duration_away',
    'avg_rt', 'min_rt', 'max_rt', 'std_rt',
    'mean_iki', 'std_iki'
]