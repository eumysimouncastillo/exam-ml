# app.py
from flask import Flask, request, jsonify
from datetime import datetime, timezone
import traceback, os
from db import get_logs_for_session, save_results_to_db
from cleaning import clean_logs
from validation import validate_logs
from features import extract_all_features, normalize_features
from models_ml import (
    train_isolation_forest, score_isolation_forest,
    train_oc_svm, score_oc_svm,
    score_zscore, score_hmm
)
from constants import MODELS_DIR

app = Flask(__name__)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/process-exam', methods=['POST'])
def process_exam():
    '''
    Called by your Node.js or PHP backend when an exam ends.
    Expected JSON body:
    { "session_id": "S001", "exam_start": "2025-06-01T09:00:00Z", "exam_end": "2025-06-01T11:00:00Z" }
    '''
    try:
        data = request.get_json()
        if not data or 'session_id' not in data:
            return jsonify({'error': 'session_id is required'}), 400

        session_id = data['session_id']
        exam_start = datetime(2025, 6, 1, 9,  0, 0, tzinfo=timezone.utc)  # Mock default
        exam_end   = datetime(2025, 6, 1, 11, 0, 0, tzinfo=timezone.utc)  # Mock default
        if 'exam_start' in data:
            exam_start = datetime.fromisoformat(data['exam_start'].replace('Z', '+00:00'))
        if 'exam_end' in data:
            exam_end = datetime.fromisoformat(data['exam_end'].replace('Z', '+00:00'))

        print(f'\n=== Processing session: {session_id} ===')

        # Step 1: Load
        raw_df = get_logs_for_session(session_id)
        print(f'Loaded {len(raw_df)} raw rows')

        # Step 2: Clean
        clean_df = clean_logs(raw_df)

        # Step 3: Validate
        valid_df = validate_logs(clean_df, exam_start, exam_end)

        # Step 4: Extract features
        features_df = extract_all_features(valid_df)

        # Step 5: Train models if none exist, otherwise load saved ones
        models_exist = os.path.exists(os.path.join(MODELS_DIR, 'isolation_forest.pkl'))
        if not models_exist:
            print('[App] No saved models — training now on current session data.')
            print('[App] NOTE: Retrain on real clean data when available for accuracy.')
            norm_df = normalize_features(features_df.copy(), is_training=True)
            train_isolation_forest(norm_df)
            train_oc_svm(norm_df)
        else:
            norm_df = normalize_features(features_df.copy(), is_training=False)

        # Step 6: Score all algorithms
        scored_df = score_isolation_forest(norm_df)
        scored_df = score_oc_svm(scored_df)
        scored_df = score_zscore(scored_df)
        # Carry _iki_sequence from features_df into scored_df for HMM
        scored_df['_iki_sequence'] = features_df['_iki_sequence'].values
        scored_df = score_hmm(scored_df)

        # Step 7: Build results
        results = []
        for _, row in scored_df.iterrows():
            flagged = bool(
                row.get('iso_flagged', False) or row.get('svm_flagged', False) or
                row.get('z_flagged',   False) or row.get('hmm_flagged', False)
            )
            results.append({
                'student_id':  row['student_id'],
                'flagged':     flagged,
                'iso_flagged': bool(row.get('iso_flagged', False)),
                'svm_flagged': bool(row.get('svm_flagged', False)),
                'z_flagged':   bool(row.get('z_flagged',   False)),
                'hmm_flagged': bool(row.get('hmm_flagged', False)),
                'iso_score':   round(float(row.get('iso_score', 0)), 4),
                'svm_score':   round(float(row.get('svm_score', 0)), 4),
                'z_score':     round(float(row.get('z_score',   0)), 4),
                'hmm_score':   round(float(row.get('hmm_score', 0)), 4)
            })

        # Step 8: Save
        save_results_to_db(session_id, results)

        return jsonify({
            'session_id': session_id,
            'students_processed': len(results),
            'results': results
        }), 200

    except Exception as e:
        print('[App] ERROR:', traceback.format_exc())
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5001, debug=False)
