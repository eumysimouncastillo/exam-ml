# app.py
from flask import Flask, request, jsonify
from datetime import datetime, timezone
import traceback, os
from db import get_logs_for_session, save_results_to_db
from cleaning import clean_logs
from validation import validate_logs
from features import extract_all_features, normalize_features
from models_ml import (
    train_isolation_forest_tab, score_isolation_forest_tab,
    train_isolation_forest_rt,  score_isolation_forest_rt,
    train_oc_svm, score_oc_svm,
    score_hmm
)
from cpi import apply_cpi_to_results
from constants import MODELS_DIR, ISO_TAB_PATH

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}, 200


@app.route('/process-exam', methods=['POST'])
def process_exam():
    """
    Called by your Laravel backend when an exam ends.
    Expected JSON body:
    {
        "session_id": "1",
        "exam_start": "2026-03-23T03:35:02Z",
        "exam_end":   "2026-03-23T03:43:09Z"
    }

    IMPORTANT: Always pass exam_start and exam_end in UTC (append Z).
    MySQL stores TIMESTAMP columns in UTC. phpMyAdmin displays them in
    Philippine time (UTC+8), so subtract 8 hours from what you see there.
    Example: phpMyAdmin shows 11:35:02 → pass 03:35:02Z in test.http.

    When Laravel triggers this automatically, pass exams.start_time and
    exams.end_time converted to UTC ISO 8601 format.
    """
    try:
        data = request.get_json()
        if not data or 'session_id' not in data:
            return {'error': 'session_id is required'}, 400

        session_id = data['session_id']

        # Default fallback window (wide, UTC-aware)
        exam_start = datetime(2000, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        exam_end   = datetime(2100, 1, 1, 0, 0, 0, tzinfo=timezone.utc)

        if 'exam_start' in data:
            exam_start = _parse_utc(data['exam_start'])

        if 'exam_end' in data:
            exam_end = _parse_utc(data['exam_end'])

        print(f'\n=== Processing session: {session_id} ===')
        print(f'[App] Window: {exam_start} → {exam_end}')

        # Step 1: Load raw logs
        raw_df = get_logs_for_session(session_id)
        print(f'Loaded {len(raw_df)} raw rows')

        # Step 2: Clean
        clean_df = clean_logs(raw_df)

        # Step 3: Validate
        valid_df = validate_logs(clean_df, exam_start, exam_end)
        
        # DEBUG - remove after fixing
        if valid_df.empty and not clean_df.empty:
            print(f'[Debug] exam_start: {exam_start}')
            print(f'[Debug] exam_end: {exam_end}')
            print(f'[Debug] First 3 timestamps in clean_df:')
            print(clean_df['timestamp'].head(3).tolist())

        # Step 4: Extract features
        features_df = extract_all_features(valid_df)

        # Guard: if no students survived validation/feature extraction, return early
        if features_df is None or features_df.empty:
            print('[App] No students to process after feature extraction.')
            return {
                'session_id':         session_id,
                'students_processed': 0,
                'students_flagged':   0,
                'results':            []
            }, 200

        # Step 5: Load saved models (must run train_models.py first)
        if not os.path.exists(ISO_TAB_PATH):
            return {'error': 'Models not found. Run train_models.py first.'}, 500

        # Step 6: Normalize and score all algorithms
        norm_df   = normalize_features(features_df.copy(), is_training=False)
        scored_df = score_isolation_forest_tab(norm_df)
        scored_df = score_isolation_forest_rt(scored_df)
        scored_df = score_oc_svm(scored_df)
        scored_df['_iki_sequence'] = features_df['_iki_sequence'].values
        scored_df = score_hmm(scored_df)

        # Step 7: Compute CPI for each student
        # CPI replaces the old OR-logic for is_flagged.
        # is_flagged is now True only when CPI >= 50% (Likely or Highly Likely).
        scored_df = apply_cpi_to_results(scored_df)

        # Step 8: Build results list
        results = []
        for _, row in scored_df.iterrows():
            results.append({
                'student_id':      row['student_id'],
                # CPI-based flag and score
                'is_flagged':      bool(row['is_flagged']),
                'cpi_score':       float(row['cpi_score']),
                'cpi_label':       str(row['cpi_label']),
                # Per-algorithm flags (kept for transparency)
                'iso_tab_flagged': bool(row.get('iso_tab_flagged', False)),
                'rt_flagged':      bool(row.get('rt_flagged',      False)),
                'svm_flagged':     bool(row.get('svm_flagged',     False)),
                'hmm_flagged':     bool(row.get('hmm_flagged',     False)),
                # Raw scores
                'iso_tab_score':   round(float(row.get('iso_tab_score', 0)), 4),
                'rt_score':        round(float(row.get('rt_score',      0)), 4),
                'svm_score':       round(float(row.get('svm_score',     0)), 4),
                'hmm_score':       round(float(row.get('hmm_score',     0)), 4)
            })

        # Step 9: Save to database
        save_results_to_db(session_id, results)

        flagged = [r for r in results if r['is_flagged']]
        return {
            'session_id':         session_id,
            'students_processed': len(results),
            'students_flagged':   len(flagged),
            'results':            results
        }, 200

    except Exception as e:
        print('[App] ERROR:', traceback.format_exc())
        return {'error': str(e)}, 500


def _parse_utc(value: str) -> datetime:
    """
    Parses a datetime string and always returns a UTC-aware datetime.
    Accepts:
      - "2026-03-23T03:35:02Z"       → treated as UTC
      - "2026-03-23T03:35:02+00:00"  → treated as UTC
      - "2026-03-23 03:35:02"        → assumed UTC (no tz marker)
    """
    value = value.strip().replace(' ', 'T')
    if value.endswith('Z'):
        value = value[:-1] + '+00:00'
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)