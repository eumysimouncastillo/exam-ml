# constants.py

# Every valid action the browser extension can send.
# Any log with an action_type NOT in this set gets discarded during validation.
VALID_ACTION_TYPES = {
    'tab_switch',
    'tab_return',
    'copy',
    'paste',
    'cut',
    'keystroke',
    'shortcut',
    'question_open',
    'question_submit'
}

# Maps each action type to a number for ML algorithms
ACTION_TYPE_MAP = {
    'tab_switch':       0,
    'tab_return':       1,
    'copy':             2,
    'paste':            3,
    'cut':              4,
    'keystroke':        5,
    'shortcut':         6,
    'question_open':    7,
    'question_submit':  8
}

# Student ID must be 7 to 10 digits. Adjust to match your school's format.
STUDENT_ID_REGEX = r'^[0-9]{7,10}$'

# Keystroke interval limits in milliseconds
# Below 30ms = physically impossible (keyboard repeat event, not a real press)
# Above 5000ms = the student stopped typing — not part of typing rhythm
IKI_MIN_MS = 30
IKI_MAX_MS = 5000

# Z-Score: how many standard deviations from average before flagging
ZSCORE_THRESHOLD = 2.5

# Minimum students needed for Z-Score to be statistically meaningful
ZSCORE_MIN_STUDENTS = 10

# Where trained ML models are saved
MODELS_DIR = 'models/'
