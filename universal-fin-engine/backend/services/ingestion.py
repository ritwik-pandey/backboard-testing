import pandas as pd
import os

def smart_read_statement(file_path):
    """
    Robustly reads Excel/CSV, finding the header row by scanning ALL columns.
    """
    ext = os.path.splitext(file_path)[1].lower()
    
    lines = []
    if ext == '.csv':
        with open(file_path, 'r') as f:
            lines = f.readlines()
    else:
        # Read first 30 rows of Excel (without header)
        df_temp = pd.read_excel(file_path, header=None, nrows=30)
        # CONVERT ENTIRE ROW TO STRING (Fixes the bug where we missed headers in Col B/C)
        lines = df_temp.apply(lambda row: ' '.join(row.dropna().astype(str)), axis=1).tolist()

    # Heuristic: Find row with "Date" AND ("Debit" OR "Withdrawal" OR "Chq")
    header_row_index = 0
    found = False
    
    for i, line in enumerate(lines):
        line_str = str(line).lower()
        # Relaxed heuristic: Just finding 'date' and 'balance' or 'debit' is usually enough
        if 'date' in line_str and ('balance' in line_str or 'debit' in line_str or 'withdrawal' in line_str or 'dr' in line_str):
            header_row_index = i
            found = True
            break
            
    # Fallback: If heuristic fails, try row 0 or row 1 (common defaults)
    if not found:
        print(f"⚠️ Warning: Could not auto-detect header in {os.path.basename(file_path)}. Defaulting to row 0.")
        header_row_index = 0
            
    # Read the full file with correct header
    if ext == '.csv':
        df = pd.read_csv(file_path, header=header_row_index)
    else:
        df = pd.read_excel(file_path, header=header_row_index)
        
    return df, header_row_index