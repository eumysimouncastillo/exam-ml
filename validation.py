# validation.py
import re, pandas as pd
from email_validator import validate_email, EmailNotValidError
from constants import VALID_ACTION_TYPES, STUDENT_ID_REGEX

def validate_logs(df, exam_start, exam_end):
    print(f'[Validation] Starting with {len(df)} rows...')
    before = len(df)
    df = df[(df['timestamp'] >= exam_start) & (df['timestamp'] <= exam_end)]
    print(f'[Validation] Removed {before - len(df)} out-of-window events')
    df = df.sort_values(['student_id','timestamp'])
    df['time_delta_sec'] = df.groupby('student_id')['timestamp'].diff().dt.total_seconds()
    before = len(df)
    df = df[df['time_delta_sec'].isna() | (df['time_delta_sec'] >= 0)]
    print(f'[Validation] Removed {before - len(df)} rows with negative time deltas')
    before = len(df)
    df = df[df.apply(_is_valid, axis=1)]
    print(f'[Validation] Removed {before - len(df)} rows with invalid fields')
    df = df.reset_index(drop=True)
    print(f'[Validation] Finished with {len(df)} rows')
    return df

def _is_valid(row):
    try:
        validate_email(row['email'], check_deliverability=False)
    except EmailNotValidError:
        return False
    if not re.match(STUDENT_ID_REGEX, row['student_id']):
        return False
    if row['action_type'] not in VALID_ACTION_TYPES:
        return False
    return True
