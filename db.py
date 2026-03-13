# db.py
import os, pandas as pd, numpy as np
from datetime import datetime, timezone, timedelta
from dotenv import load_dotenv
import random

load_dotenv()

USE_MOCK_DATA = True   # Set to False when real MySQL tables are ready

DB_CONFIG = {
    'host':     os.getenv('DB_HOST',     'localhost'),
    'user':     os.getenv('DB_USER',     'root'),
    'password': os.getenv('DB_PASSWORD', ''),
    'database': os.getenv('DB_NAME',     'exam_db')
}

def get_logs_for_session(session_id):
    if USE_MOCK_DATA:
        return _generate_mock_logs(session_id)
    return _load_from_mysql(session_id)

def save_results_to_db(session_id, results):
    if USE_MOCK_DATA:
        print('--- MOCK RESULTS ---')
        for r in results: print(r)
        return
    import mysql.connector
    conn = mysql.connector.connect(**DB_CONFIG)
    cursor = conn.cursor()
    for r in results:
        cursor.execute(
            '''INSERT INTO exam_results
               (session_id, student_id, is_flagged, iso_tab_flagged,
                svm_flagged, rt_flagged, hmm_flagged, processed_at)
               VALUES (%s,%s,%s,%s,%s,%s,%s,NOW())
               ON DUPLICATE KEY UPDATE is_flagged=%s, iso_tab_flagged=%s,
               svm_flagged=%s, rt_flagged=%s, hmm_flagged=%s, processed_at=NOW()''',
            (session_id, r['student_id'], r['flagged'],
             r['iso_tab_flagged'], r['svm_flagged'], r['rt_flagged'], r['hmm_flagged'],
             r['flagged'], r['iso_tab_flagged'], r['svm_flagged'],
             r['rt_flagged'], r['hmm_flagged'])
        )
    conn.commit(); cursor.close(); conn.close()

def _load_from_mysql(session_id):
    import mysql.connector
    conn = mysql.connector.connect(**DB_CONFIG)
    df = pd.read_sql(
        'SELECT session_id,student_id,email,action_type,question_id,key_pressed,timestamp'
        ' FROM exam_logs WHERE session_id=%s ORDER BY timestamp ASC',
        conn, params=(session_id,))
    conn.close()
    return df

def _generate_mock_logs(session_id):
    random.seed(42); np.random.seed(42)
    exam_start = datetime(2025, 6, 1, 9, 0, 0, tzinfo=timezone.utc)
    logs = []
    students = [
        ('STU0001','alice@school.edu',False),
        ('STU0002','bob@school.edu',False),
        ('STU0003','carol@school.edu',False),
        ('STU0004','dave@school.edu',True),   # Suspicious
        ('STU0005','eve@school.edu',False),
    ]
    for sid, email, suspicious in students:
        t = exam_start + timedelta(seconds=random.uniform(3, 10))
        for q_num in range(1, 11):
            q_id = f'Q{q_num}'
            logs.append({'session_id':session_id,'student_id':sid,'email':email,
                'action_type':'question_open','question_id':q_id,'key_pressed':None,'timestamp':t})
            for _ in range(random.randint(15, 60)):
                iki = random.gauss(200,40) if not suspicious else random.gauss(80,20)
                t += timedelta(milliseconds=max(30, iki))
                logs.append({'session_id':session_id,'student_id':sid,'email':email,
                    'action_type':'keystroke','question_id':q_id,'key_pressed':'a','timestamp':t})
            if suspicious and q_num == 3:
                logs.append({'session_id':session_id,'student_id':sid,'email':email,
                    'action_type':'tab_switch','question_id':None,'key_pressed':None,'timestamp':t})
                t += timedelta(seconds=random.uniform(30, 90))
                logs.append({'session_id':session_id,'student_id':sid,'email':email,
                    'action_type':'tab_return','question_id':None,'key_pressed':None,'timestamp':t})
            if suspicious and q_num == 5:
                for act in ['copy','paste']:
                    logs.append({'session_id':session_id,'student_id':sid,'email':email,
                        'action_type':act,'question_id':q_id,'key_pressed':None,'timestamp':t})
                    t += timedelta(milliseconds=200)
            rt = random.uniform(20,150) if not suspicious else random.uniform(2,8)
            t += timedelta(seconds=rt)
            logs.append({'session_id':session_id,'student_id':sid,'email':email,
                'action_type':'question_submit','question_id':q_id,'key_pressed':None,'timestamp':t})
    return pd.DataFrame(logs)
