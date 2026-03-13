# features.py
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import MinMaxScaler
from constants import IKI_MIN_MS, IKI_MAX_MS, MODELS_DIR, SCALER_PATH, SCALE_FEATURES


def extract_all_features(df):
    """Main function. Returns one row of features per student."""
    results = []
    for sid, sdf in df.groupby('student_id'):
        tab = _tab_features(sdf)
        cp  = _copypaste_features(sdf)
        rt  = _response_time_features(sdf)
        ks  = _keystroke_features(sdf)
        results.append({
            'student_id':    sid,
            **tab, **cp, **rt,
            'mean_iki':      ks['mean_iki'],
            'std_iki':       ks['std_iki'],
            '_iki_sequence': ks['iki_sequence']
        })
    fdf = pd.DataFrame(results)
    print(f'[Features] Extracted features for {len(fdf)} students')
    return fdf


def _tab_features(df):
    sw = df[df['action_type'] == 'tab_switch']
    rt = df[df['action_type'] == 'tab_return']
    if sw.empty:
        return {'tab_switch_count': 0, 'avg_duration_away': 0.0, 'max_duration_away': 0.0}
    durs = []
    for t in sw['timestamp']:
        after = rt[rt['timestamp'] > t]
        if not after.empty:
            durs.append((after.iloc[0]['timestamp'] - t).total_seconds())
    return {
        'tab_switch_count':  len(sw),
        'avg_duration_away': float(np.mean(durs)) if durs else 0.0,
        'max_duration_away': float(np.max(durs))  if durs else 0.0
    }


def _copypaste_features(df):
    c = df['action_type'].value_counts()
    return {
        'copy_count':     int(c.get('copy',     0)),
        'paste_count':    int(c.get('paste',    0)),
        'cut_count':      int(c.get('cut',      0)),
        'shortcut_count': int(c.get('shortcut', 0))
    }


def _response_time_features(df):
    opens   = df[df['action_type'] == 'question_open'].set_index('question_id')
    submits = df[df['action_type'] == 'question_submit'].set_index('question_id')
    rts = []
    for qid in opens.index:
        if qid in submits.index:
            ot = opens.loc[qid, 'timestamp']
            st = submits.loc[qid, 'timestamp']
            if isinstance(ot, pd.Series): ot = ot.iloc[0]
            if isinstance(st, pd.Series): st = st.iloc[-1]
            d = (st - ot).total_seconds()
            if d > 0:
                rts.append(d)
    if not rts:
        return {'avg_rt': 0.0, 'min_rt': 0.0, 'max_rt': 0.0, 'std_rt': 0.0}
    rt = np.array(rts)
    return {
        'avg_rt': float(rt.mean()), 'min_rt': float(rt.min()),
        'max_rt': float(rt.max()), 'std_rt': float(rt.std())
    }


def _keystroke_features(df):
    ks = df[df['action_type'] == 'keystroke'].sort_values('timestamp')
    if len(ks) < 2:
        return {'mean_iki': 0.0, 'std_iki': 0.0, 'iki_sequence': []}
    iki = ks['timestamp'].diff().dt.total_seconds().dropna() * 1000
    iki = iki[(iki >= IKI_MIN_MS) & (iki <= IKI_MAX_MS)]
    if iki.empty:
        return {'mean_iki': 0.0, 'std_iki': 0.0, 'iki_sequence': []}
    return {
        'mean_iki':     float(iki.mean()),
        'std_iki':      float(iki.std()),
        'iki_sequence': iki.tolist()
    }


def normalize_features(features_df, is_training=False):
    """
    Scales only continuous numerical features to 0-1 range.

    IMPORTANT: copy_count, paste_count, cut_count, shortcut_count are
    intentionally excluded from scaling (see SCALE_FEATURES in constants.py).
    They are passed to One-Class SVM as raw counts. Scaling them would
    destroy the anomaly signal because all training values are 0, giving
    MinMaxScaler no range to work with.

    is_training=True  → fit and save a new scaler (run once via train_models.py)
    is_training=False → load and apply the saved scaler (every exam after that)
    """
    if is_training:
        scaler = MinMaxScaler()
        features_df[SCALE_FEATURES] = scaler.fit_transform(features_df[SCALE_FEATURES])
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print('[Normalization] Scaler fitted and saved (copy/paste counts excluded)')
    else:
        if not os.path.exists(SCALER_PATH):
            raise FileNotFoundError('Scaler not found. Run train_models.py first.')
        scaler = joblib.load(SCALER_PATH)
        features_df[SCALE_FEATURES] = scaler.transform(features_df[SCALE_FEATURES])
        print('[Normalization] Features scaled (copy/paste counts left as raw)')
    return features_df