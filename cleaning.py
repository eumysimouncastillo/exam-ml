# cleaning.py
import pandas as pd

def clean_logs(df):
    print(f'[Cleaning] Starting with {len(df)} rows...')
    df['timestamp']   = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')
    df['student_id']  = df['student_id'].astype(str).str.strip()
    df['email']       = df['email'].astype(str).str.strip().str.lower()
    df['action_type'] = df['action_type'].astype(str).str.strip().str.lower()
    df['session_id']  = df['session_id'].astype(str).str.strip()
    df['key_pressed'] = df['key_pressed'].fillna('no_key')
    df['question_id'] = df['question_id'].fillna('none')
    before = len(df)
    df = df.drop_duplicates(subset=['student_id','timestamp','action_type'], keep='first')
    print(f'[Cleaning] Removed {before - len(df)} duplicates')
    before = len(df)
    df = df.dropna(subset=['student_id','session_id','timestamp','action_type'])
    print(f'[Cleaning] Removed {before - len(df)} rows with missing fields')
    df = df.sort_values(['student_id','timestamp']).reset_index(drop=True)
    print(f'[Cleaning] Finished with {len(df)} rows')
    return df
