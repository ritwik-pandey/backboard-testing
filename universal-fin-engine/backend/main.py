import os
import json
import asyncio
import pandas as pd
import numpy as np

# --- MICROSERVICES IMPORTS ---
from services.ingestion import smart_read_statement
from services.normalizer import normalize_columns
from services.intelligence import extract_metadata_real 
from services.validator import validate_consistency

def clean_transactions(df):
    """
    Filters out garbage rows (Totals, Footers, Empty Dates) that break the engine.
    """
    # 1. Drop rows where Date is missing (Essential for a transaction)
    df = df.dropna(subset=['date'])
    df = df[df['date'].astype(str).str.strip() != '']
    df = df[df['date'].astype(str).str.lower() != 'nan']

    # 2. Filter out Summary/Footer Keywords in Description
    # We lower-case the description for checking
    exclude_keywords = [
        'total', 'closing balance', 'opening balance', 
        'page', 'sr. no.', 'recover date', 'summary'
    ]
    
    # Create a mask: True if row is GOOD, False if it contains bad keywords
    # We use astype(str) to handle potential non-string descriptions
    mask = ~df['description'].astype(str).str.lower().apply(
        lambda x: any(k in x for k in exclude_keywords)
    )
    df = df[mask]

    # 3. Drop rows where Debit AND Credit are both 0 (Empty rows)
    df = df[~((df['debit'] == 0) & (df['credit'] == 0))]

    return df

async def process_file_async(file_path):
    filename = os.path.basename(file_path)
    print(f"🔄 Processing {filename}...")
    
    # 1. STRUCTURAL EXTRACTION
    try:
        raw_df, header_idx = smart_read_statement(file_path)
        clean_df = normalize_columns(raw_df)
        
        # --- DATA TYPING ---
        # Convert Date to string
        if 'date' in clean_df.columns:
            clean_df['date'] = clean_df['date'].astype(str)

        # Clean Numbers (Remove commas)
        for col in ['debit', 'credit', 'balance']:
            clean_df[col] = clean_df[col].astype(str).str.replace(',', '', regex=True)
            clean_df[col] = pd.to_numeric(clean_df[col], errors='coerce').fillna(0.0)

        # --- NEW: FILTER GARBAGE ROWS ---
        clean_df = clean_transactions(clean_df)
            
    except Exception as e:
        print(f"❌ Read Error {filename}: {e}")
        return None

    # 2. INTELLIGENCE LAYER
    print(f"🧠 Sending {filename} to Backboard AI...")
    metadata = await extract_metadata_real(file_path, header_idx)
    
    # 3. VALIDATION LAYER
    transactions_list = clean_df.to_dict(orient='records')
    # Replace NaN with None for JSON compliance
    transactions_list = [{k: (None if pd.isna(v) else v) for k, v in t.items()} for t in transactions_list]
    
    validation_report = validate_consistency(transactions_list)

    # 4. KNOWLEDGE OBJECT
    financial_object = {
        "document_id": filename,
        "status": "Validated" if validation_report['is_valid'] else "Review Required",
        "ai_metadata": metadata,
        "summary": {
            "total_debit": float(clean_df['debit'].sum()),
            "total_credit": float(clean_df['credit'].sum()),
            "record_count": len(clean_df),
            # Handle empty case safely
            "closing_balance": float(clean_df['balance'].iloc[-1]) if not clean_df.empty else 0.0
        },
        "validation_flags": {
            "balance_continuity": validation_report['is_valid'],
            "error_count": validation_report['error_count']
        },
        "transactions": transactions_list
    }
    
    print(f"✅ Finished {filename}")
    return financial_object

async def run_pipeline():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw_uploads")
    output_file = os.path.join(base_dir, "data", "processed_json", "financial_knowledge_graph.json")
    
    if not os.path.exists(raw_dir):
        print(f"❌ Error: {raw_dir} not found.")
        return

    tasks = []
    for filename in os.listdir(raw_dir):
        if filename.endswith(".xlsx") or filename.endswith(".csv"):
            full_path = os.path.join(raw_dir, filename)
            tasks.append(process_file_async(full_path))

    if not tasks:
        print("⚠️ No files found.")
        return

    results = await asyncio.gather(*tasks)
    valid_results = [r for r in results if r is not None]

    with open(output_file, "w") as f:
        json.dump(valid_results, f, indent=4)
    
    print(f"\n🚀 Pipeline Complete. Processed {len(valid_results)} documents.")
    print(f"💾 Knowledge Graph saved to: {output_file}")

if __name__ == "__main__":
    asyncio.run(run_pipeline())