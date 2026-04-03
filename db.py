# db.py
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import random

load_dotenv()

USE_MOCK_DATA = False   # Set to False when real MySQL tables are ready

DB_CONFIG = {
    'host':     os.getenv('DB_HOST',     'localhost'),
    'user':     os.getenv('DB_USER',     'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME',     'sect_app'),
    'unix_socket': os.getenv('DB_SOCKET', '')
}


def get_logs_for_session(session_id):
    """
    Returns a unified DataFrame of all behavior events for one submission.
    session_id = exam_submissions.id (sent as a string from your backend).
    """
    if USE_MOCK_DATA:
        return _generate_mock_logs(session_id)
    return _load_from_mysql(session_id)


def save_results_to_db(session_id, results):
    """
    Saves ML results and CPI scores to exam_results.
    session_id = exam_submissions.id.
    In mock mode, prints to console instead.
    """
    if USE_MOCK_DATA:
        print('--- MOCK RESULTS ---')
        for r in results:
            print(f"  {r['student_id']} | CPI: {r['cpi_score']}% "
                  f"({r['cpi_label']}) | Flagged: {r['is_flagged']}")
        return

    import mysql.connector
    conn   = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()

    submission_id = int(session_id)

    # Get exam_id from exam_submissions
    cursor.execute('SELECT exam_id FROM exam_submissions WHERE id = %s', (submission_id,))
    row = cursor.fetchone()
    exam_id = row[0] if row else None

    for r in results:
        student_id = int(r['student_id'])
        cursor.execute(
            '''INSERT INTO exam_results
               (submission_id, exam_id, student_id, is_flagged,
                cpi_score, cpi_label,
                iso_tab_flagged, svm_flagged, rt_flagged, hmm_flagged,
                iso_tab_score, svm_score, rt_score, hmm_score,
                processed_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,NOW())
               ON DUPLICATE KEY UPDATE
               is_flagged=%s, cpi_score=%s, cpi_label=%s,
               iso_tab_flagged=%s, svm_flagged=%s, rt_flagged=%s, hmm_flagged=%s,
               iso_tab_score=%s, svm_score=%s, rt_score=%s, hmm_score=%s,
               processed_at=NOW()''',
            (
                # INSERT values
                submission_id, exam_id, student_id, r['is_flagged'],
                r['cpi_score'], r['cpi_label'],
                r['iso_tab_flagged'], r['svm_flagged'],
                r['rt_flagged'], r['hmm_flagged'],
                r['iso_tab_score'], r['svm_score'],
                r['rt_score'], r['hmm_score'],
                # UPDATE values
                r['is_flagged'], r['cpi_score'], r['cpi_label'],
                r['iso_tab_flagged'], r['svm_flagged'],
                r['rt_flagged'], r['hmm_flagged'],
                r['iso_tab_score'], r['svm_score'],
                r['rt_score'], r['hmm_score']
            )
        )
    conn.commit()
    cursor.close()
    conn.close()


def _load_from_mysql(session_id):
    """
    Reads from all four behavior tables using submission_id,
    merges into one unified DataFrame that the pipeline expects.
    """
    import mysql.connector
    submission_id = int(session_id)
    conn = mysql.connector.connect(**DB_CONFIG)

    # Use dictionary=True cursor for all queries to avoid pd.read_sql warnings
    # and to correctly handle reserved words like `keys`
    cursor = conn.cursor(dictionary=True)

    # Get submission info (student_id, exam_id)
    cursor.execute(
        'SELECT id, exam_id, student_id FROM exam_submissions WHERE id = %s',
        (submission_id,)
    )
    sub_row = cursor.fetchone()
    if not sub_row:
        cursor.close()
        conn.close()
        print(f'[DB] No submission found for id={submission_id}')
        return pd.DataFrame()

    student_id = int(sub_row['student_id'])
    exam_id    = int(sub_row['exam_id'])

    # Get student email from users table
    cursor.execute('SELECT email FROM users WHERE id = %s', (student_id,))
    user_row = cursor.fetchone()
    email = user_row['email'] if user_row else f'user{student_id}@school.edu'

    logs = []

    # ── 1. Tab Switch Logs ────────────────────────────────────────────────────
    # is_return_event=0 → student left tab (tab_switch)
    # is_return_event=1 → student returned (tab_return)
    cursor.execute(
        'SELECT is_return_event, hidden_duration_ms, occurred_at '
        'FROM tab_switch_logs WHERE submission_id = %s ORDER BY occurred_at ASC',
        (submission_id,)
    )
    for row in cursor.fetchall():
        action = 'tab_return' if row['is_return_event'] else 'tab_switch'
        logs.append({
            'session_id':  str(submission_id),
            'student_id':  str(student_id),
            'email':       email,
            'action_type': action,
            'question_id': None,
            'key_pressed': None,
            'timestamp':   row['occurred_at']
        })

    # ── 2. Keyboard Shortcut Logs ─────────────────────────────────────────────
    # Map keys column to action_type
    # Backticks around `keys` are handled safely via cursor.execute parameters
    cursor.execute(
        'SELECT `keys`, is_paste, question_id, occurred_at '
        'FROM keyboard_shortcut_logs WHERE submission_id = %s ORDER BY occurred_at ASC',
        (submission_id,)
    )
    for row in cursor.fetchall():
        keys = str(row['keys']).lower().strip()
        if 'ctrl+c' in keys or keys == 'copy':
            action = 'copy'
        elif row['is_paste'] or 'ctrl+v' in keys or keys == 'paste':
            action = 'paste'
        elif 'ctrl+x' in keys or keys == 'cut':
            action = 'cut'
        else:
            action = 'shortcut'
        logs.append({
            'session_id':  str(submission_id),
            'student_id':  str(student_id),
            'email':       email,
            'action_type': action,
            'question_id': str(row['question_id']) if row['question_id'] else None,
            'key_pressed': str(row['keys']),
            'timestamp':   row['occurred_at']
        })

    # ── 3. Keystroke Dynamics Logs ────────────────────────────────────────────
    # flight_times_ms is the IKI sequence — expand into individual keystroke rows
    # Only use is_baseline=0 rows for exam scoring
    cursor.execute(
        'SELECT question_id, flight_times_ms, occurred_at '
        'FROM keystroke_dynamics_logs '
        'WHERE submission_id = %s AND is_baseline = 0 ORDER BY occurred_at ASC',
        (submission_id,)
    )
    for row in cursor.fetchall():
        try:
            flight_times = json.loads(row['flight_times_ms']) if row['flight_times_ms'] else []
        except (json.JSONDecodeError, TypeError):
            flight_times = []
        base_time  = pd.to_datetime(row['occurred_at'])
        cumulative = 0
        for ft in flight_times:
            cumulative += float(ft)
            t = base_time + pd.Timedelta(milliseconds=cumulative)
            logs.append({
                'session_id':  str(submission_id),
                'student_id':  str(student_id),
                'email':       email,
                'action_type': 'keystroke',
                'question_id': str(row['question_id']),
                'key_pressed': 'key',
                'timestamp':   t
            })

    # ── 4. Response Time Logs ─────────────────────────────────────────────────
    # Reconstruct question_open + question_submit pairs from response_time_ms
    # occurred_at = submit time; open time = occurred_at - response_time_ms
    cursor.execute(
        'SELECT question_id, response_time_ms, occurred_at '
        'FROM response_time_logs '
        'WHERE submission_id = %s AND is_baseline = 0 ORDER BY occurred_at ASC',
        (submission_id,)
    )
    for row in cursor.fetchall():
        submit_time = pd.to_datetime(row['occurred_at'])
        rt_seconds  = float(row['response_time_ms']) / 1000.0
        open_time   = submit_time - pd.Timedelta(seconds=rt_seconds)
        qid         = str(row['question_id'])
        logs.append({
            'session_id':  str(submission_id),
            'student_id':  str(student_id),
            'email':       email,
            'action_type': 'question_open',
            'question_id': qid,
            'key_pressed': None,
            'timestamp':   open_time
        })
        logs.append({
            'session_id':  str(submission_id),
            'student_id':  str(student_id),
            'email':       email,
            'action_type': 'question_submit',
            'question_id': qid,
            'key_pressed': None,
            'timestamp':   submit_time
        })

    cursor.close()
    conn.close()

    if not logs:
        print(f'[DB] No behavior data found for submission_id={submission_id}')
        return pd.DataFrame()

    df = pd.DataFrame(logs)
    # Keep timestamps timezone-naive to match exam_start/exam_end from app.py
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['timestamp'] = df['timestamp'].dt.tz_localize('Asia/Manila').dt.tz_convert('UTC')
    df = df.sort_values('timestamp').reset_index(drop=True)
    print(f'[DB] Loaded {len(df)} merged rows for submission_id={submission_id}')
    return df


def _generate_mock_logs(session_id):
    """Generates fake logs mirroring the unified structure the pipeline expects."""
    random.seed(45)
    np.random.seed(45)
    exam_start = datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
    logs = []
    students = [
        ('1', 'alice@school.edu',  False),
        ('2', 'bob@school.edu',    False),
        ('3', 'carol@school.edu',  False),
        ('4', 'dave@school.edu',   True),   # Suspicious
        ('5', 'eve@school.edu',    False),
    ]
    for sid, email, suspicious in students:
        t = exam_start + timedelta(seconds=random.uniform(3, 10))
        for q_num in range(1, 11):
            q_id = str(q_num)
            logs.append({'session_id': session_id, 'student_id': sid, 'email': email,
                'action_type': 'question_open', 'question_id': q_id,
                'key_pressed': None, 'timestamp': t})
            for _ in range(random.randint(15, 60)):
                iki = random.gauss(200, 40) if not suspicious else random.gauss(80, 20)
                t  += timedelta(milliseconds=max(30, iki))
                logs.append({'session_id': session_id, 'student_id': sid, 'email': email,
                    'action_type': 'keystroke', 'question_id': q_id,
                    'key_pressed': 'key', 'timestamp': t})
            if suspicious and q_num == 3:
                logs.append({'session_id': session_id, 'student_id': sid, 'email': email,
                    'action_type': 'tab_switch', 'question_id': None,
                    'key_pressed': None, 'timestamp': t})
                t += timedelta(seconds=random.uniform(30, 90))
                logs.append({'session_id': session_id, 'student_id': sid, 'email': email,
                    'action_type': 'tab_return', 'question_id': None,
                    'key_pressed': None, 'timestamp': t})
            if suspicious and q_num == 5:
                for act in ['copy', 'paste']:
                    logs.append({'session_id': session_id, 'student_id': sid, 'email': email,
                        'action_type': act, 'question_id': q_id,
                        'key_pressed': None, 'timestamp': t})
                    t += timedelta(milliseconds=200)
            rt = random.uniform(20, 150) if not suspicious else random.uniform(2, 8)
            t += timedelta(seconds=rt)
            logs.append({'session_id': session_id, 'student_id': sid, 'email': email,
                'action_type': 'question_submit', 'question_id': q_id,
                'key_pressed': None, 'timestamp': t})
    return pd.DataFrame(logs)