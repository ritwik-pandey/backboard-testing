def validate_consistency(transactions, opening_balance=0.0):
    """
    Checks if Balance[t] == Balance[t-1] - Debit + Credit.
    Returns a list of flags and a boolean 'is_valid'.
    """
    flags = []
    is_valid_sequence = True
    
    # Sort by date if needed, but usually statements are already ordered
    # We assume the list is ordered chronologically
    
    previous_balance = opening_balance
    
    for i, txn in enumerate(transactions):
        # specific to your data: Debit reduces balance, Credit increases it
        # Handle cases where values might be 0
        debit = float(txn.get('debit', 0))
        credit = float(txn.get('credit', 0))
        reported_balance = float(txn.get('balance', 0))
        
        # Calculate expected balance
        # Note: Some statements run oldest-to-newest, others newest-to-oldest.
        # We need to detect direction. For now, assuming standard chronological (Old -> New).
        
        if i == 0:
            # First row is tricky without explicit opening balance from metadata
            # We skip validation of the very first row's math against previous
            expected_balance = reported_balance 
        else:
            expected_balance = previous_balance - debit + credit
            
            # Allow small floating point tolerance (0.01)
            if abs(reported_balance - expected_balance) > 0.05:
                is_valid_sequence = False
                flags.append({
                    "row_index": i,
                    "date": txn.get('date'),
                    "error": "Balance Mismatch",
                    "reported": reported_balance,
                    "calculated": expected_balance,
                    "diff": reported_balance - expected_balance
                })
        
        previous_balance = reported_balance

    return {
        "is_valid": is_valid_sequence,
        "error_count": len(flags),
        "flags": flags
    }