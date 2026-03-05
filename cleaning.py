# cleaning.py
import pandas as pd
from constants import VALID_ACTION_TYPES


def clean_logs(df):
    '''
    Cleans raw exam logs. Always call this first.
    Steps: fix timestamps → fix text fields → remove duplicates → drop missing rows → sort.
    '''
    print(f'[Cleaning] Starting with {len(df)} rows...')

    # Fix timestamps — convert any format to UTC datetime objects
    # errors='coerce' turns unparseable timestamps into NaT instead of crashing
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True, errors='coerce')

    # Fix text fields — strip spaces, make lowercase for consistency
    df['student_id']  = df['student_id'].astype(str).str.strip()
    df['email']       = df['email'].astype(str).str.strip().str.lower()
    df['action_type'] = df['action_type'].astype(str).str.strip().str.lower()
    df['session_id']  = df['session_id'].astype(str).str.strip()

    # Fill optional fields with defaults so they are never empty
    df['key_pressed'] = df['key_pressed'].fillna('no_key')
    df['question_id'] = df['question_id'].fillna('none')

    # Remove duplicates — identical student + timestamp + action = browser retry
    before = len(df)
    df = df.drop_duplicates(subset=['student_id', 'timestamp', 'action_type'], keep='first')
    print(f'[Cleaning] Removed {before - len(df)} duplicate rows')

    # Drop rows where critical fields are empty or could not be parsed
    before = len(df)
    df = df.dropna(subset=['student_id', 'session_id', 'timestamp', 'action_type'])
    print(f'[Cleaning] Removed {before - len(df)} rows with missing critical fields')

    # Sort by student then time — required for time-based calculations later
    df = df.sort_values(['student_id', 'timestamp'])
    df = df.reset_index(drop=True)

    print(f'[Cleaning] Finished with {len(df)} clean rows')
    return df
