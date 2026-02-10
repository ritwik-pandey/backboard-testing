import pandas as pd

def normalize_columns(df):
    """
    Maps varying column names to a strict schema.
    """
    # Normalize headers to lowercase
    df.columns = df.columns.str.strip().str.lower()
    
    # Mapping Dictionary
    column_map = {
        'date': 'date', 'txn date': 'date', 'transaction date': 'date', 'tran date': 'date',
        'particulars': 'description', 'transaction details': 'description', 'narration': 'description', 'description': 'description',
        'debit': 'debit', 'withdrawal': 'debit', 'dr': 'debit',
        'credit': 'credit', 'deposit': 'credit', 'cr': 'credit',
        'balance': 'balance', 'bal': 'balance', 'closing balance': 'balance',
        'chq no': 'ref_no', 'cheque no': 'ref_no', 'ref no./cheque no.': 'ref_no'
    }
    
    # Rename columns
    new_cols = {}
    for col in df.columns:
        for key, value in column_map.items():
            if key == col:
                new_cols[col] = value
                break
    
    df = df.rename(columns=new_cols)
    
    # Ensure required columns exist, fill with 0 or empty if missing
    required = ['date', 'description', 'debit', 'credit', 'balance']
    for req in required:
        if req not in df.columns:
            # If a column is missing, add it with default values
            df[req] = 0 if req in ['debit', 'credit', 'balance'] else ""
            
    return df[required]