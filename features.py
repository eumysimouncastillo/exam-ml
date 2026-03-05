# features.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from constants import IKI_MIN_MS, IKI_MAX_MS, MODELS_DIR

SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.pkl')
NUMERICAL_FEATURES = [
    'tab_switch_count', 'avg_duration_away', 'max_duration_away',
    'copy_count', 'paste_count', 'cut_count', 'shortcut_count',
    'avg_rt', 'min_rt', 'max_rt', 'std_rt', 'mean_iki', 'std_iki'
]


def extract_all_features(df):
    '''Main function. Returns one row of features per student.'''
    results = []
    for student_id, student_df in df.groupby('student_id'):
        tab = _tab_features(student_df)
        cp  = _copypaste_features(student_df)
        rt  = _response_time_features(student_df)
        ks  = _keystroke_features(student_df)
        row = {
            'student_id': student_id,
            **tab, **cp,
            'mean_iki':      ks['mean_iki'],
            'std_iki':       ks['std_iki'],
            '_iki_sequence': ks['iki_sequence'],
            **rt
        }
        results.append(row)
    features_df = pd.DataFrame(results)
    print(f'[Features] Extracted features for {len(features_df)} students')
    return features_df


def _tab_features(df):
    switches = df[df['action_type'] == 'tab_switch']
    returns  = df[df['action_type'] == 'tab_return']
    if switches.empty:
        return {'tab_switch_count': 0, 'avg_duration_away': 0.0, 'max_duration_away': 0.0}
    durations = []
    for sw_time in switches['timestamp']:
        after = returns[returns['timestamp'] > sw_time]
        if not after.empty:
            durations.append((after.iloc[0]['timestamp'] - sw_time).total_seconds())
    return {
        'tab_switch_count':  len(switches),
        'avg_duration_away': float(np.mean(durations))  if durations else 0.0,
        'max_duration_away': float(np.max(durations))   if durations else 0.0
    }


def _copypaste_features(df):
    counts = df['action_type'].value_counts()
    return {
        'copy_count':     int(counts.get('copy',     0)),
        'paste_count':    int(counts.get('paste',    0)),
        'cut_count':      int(counts.get('cut',      0)),
        'shortcut_count': int(counts.get('shortcut', 0))
    }


def _response_time_features(df):
    opens   = df[df['action_type'] == 'question_open'].set_index('question_id')
    submits = df[df['action_type'] == 'question_submit'].set_index('question_id')
    rts = []
    for q_id in opens.index:
        if q_id in submits.index:
            ot = opens.loc[q_id, 'timestamp']
            st = submits.loc[q_id, 'timestamp']
            if isinstance(ot, pd.Series): ot = ot.iloc[0]
            if isinstance(st, pd.Series): st = st.iloc[-1]
            d = (st - ot).total_seconds()
            if d > 0: rts.append(d)
    if not rts:
        return {'avg_rt': 0.0, 'min_rt': 0.0, 'max_rt': 0.0, 'std_rt': 0.0}
    rt = np.array(rts)
    return {'avg_rt': float(rt.mean()), 'min_rt': float(rt.min()),
            'max_rt': float(rt.max()), 'std_rt': float(rt.std())}


def _keystroke_features(df):
    ks = df[df['action_type'] == 'keystroke'].sort_values('timestamp')
    if len(ks) < 2:
        return {'mean_iki': 0.0, 'std_iki': 0.0, 'iki_sequence': []}
    iki = ks['timestamp'].diff().dt.total_seconds().dropna() * 1000
    iki = iki[(iki >= IKI_MIN_MS) & (iki <= IKI_MAX_MS)]
    if iki.empty:
        return {'mean_iki': 0.0, 'std_iki': 0.0, 'iki_sequence': []}
    return {'mean_iki': float(iki.mean()), 'std_iki': float(iki.std()),
            'iki_sequence': iki.tolist()}


def normalize_features(features_df, is_training=False):
    '''
    Scales numerical features to 0-1 range.
    is_training=True  → fit and save a new scaler (first run only)
    is_training=False → load and apply the saved scaler (every exam after that)
    '''
    if is_training:
        scaler = MinMaxScaler()
        features_df[NUMERICAL_FEATURES] = scaler.fit_transform(features_df[NUMERICAL_FEATURES])
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f'[Normalization] Scaler fitted and saved')
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError('Scaler not found. Run with is_training=True first.')
        scaler = joblib.load(SCALER_PATH)
        features_df[NUMERICAL_FEATURES] = scaler.transform(features_df[NUMERICAL_FEATURES])
        print('[Normalization] Features scaled using saved scaler')
    return features_df
