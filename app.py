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
from constants import MODELS_DIR, ISO_TAB_PATH

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    return {'status': 'ok'}, 200

@app.route('/process-exam', methods=['POST'])
def process_exam():
    try:
        data = request.get_json()
        if not data or 'session_id' not in data:
            return {'error': 'session_id is required'}, 400

        session_id = data['session_id']
        exam_start = datetime(2025, 6, 1, 9,  0, 0, tzinfo=timezone.utc)
        exam_end   = datetime(2025, 6, 1, 11, 0, 0, tzinfo=timezone.utc)
        if 'exam_start' in data:
            exam_start = datetime.fromisoformat(data['exam_start'].replace('Z','+00:00'))
        if 'exam_end' in data:
            exam_end = datetime.fromisoformat(data['exam_end'].replace('Z','+00:00'))

        print(f'\n=== Processing session: {session_id} ===')

        raw_df      = get_logs_for_session(session_id)
        clean_df    = clean_logs(raw_df)
        valid_df    = validate_logs(clean_df, exam_start, exam_end)
        features_df = extract_all_features(valid_df)

        # Load saved models (must run train_models.py first)
        if not os.path.exists(ISO_TAB_PATH):
            return {'error': 'Models not found. Run train_models.py first.'}, 500

        norm_df   = normalize_features(features_df.copy(), is_training=False)
        scored_df = score_isolation_forest_tab(norm_df)
        scored_df = score_isolation_forest_rt(scored_df)
        scored_df = score_oc_svm(scored_df)
        scored_df['_iki_sequence'] = features_df['_iki_sequence'].values
        scored_df = score_hmm(scored_df)

        results = []
        for _, row in scored_df.iterrows():
            flagged = bool(
                row.get('iso_tab_flagged', False) or
                row.get('rt_flagged',      False) or
                row.get('svm_flagged',     False) or
                row.get('hmm_flagged',     False)
            )
            results.append({
                'student_id':     row['student_id'],
                'flagged':        flagged,
                'iso_tab_flagged':bool(row.get('iso_tab_flagged', False)),
                'rt_flagged':     bool(row.get('rt_flagged',      False)),
                'svm_flagged':    bool(row.get('svm_flagged',     False)),
                'hmm_flagged':    bool(row.get('hmm_flagged',     False)),
                'iso_tab_score':  round(float(row.get('iso_tab_score', 0)), 4),
                'rt_score':       round(float(row.get('rt_score',      0)), 4),
                'svm_score':      round(float(row.get('svm_score',     0)), 4),
                'hmm_score':      round(float(row.get('hmm_score',     0)), 4)
            })

        save_results_to_db(session_id, results)
        return {'session_id': session_id, 'students_processed': len(results),
                'results': results}, 200

    except Exception as e:
        print('[App] ERROR:', traceback.format_exc())
        return {'error': str(e)}, 500

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
