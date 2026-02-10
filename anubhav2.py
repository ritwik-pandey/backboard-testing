import asyncio
import os
import io
import csv
import re
import json
import pdfplumber
import pandas as pd
import time
from backboard import BackboardClient

# --- CONFIGURATION ---
PDF_FOLDER_PATH = "/Users/anubhavkayal/Data Scrapping/backboard-testing/reports"
OUTPUT_FILE = "monte_carlo_final_data.csv" 
API_KEY = "espr_Hlape8JgiHiwIEt-Ash9LE55tIyTECHENuUHRAYjlKU"

# 1. THE EXACT 32-METRIC SCHEMA
METRICS_SCHEMA = [
    {"metric_name": "revenue_total", "aliases": ["total income", "interest earned", "revenue from operations", "sales"]},
    {"metric_name": "cost_of_revenue", "aliases": ["interest expended", "cost of goods sold", "cogs", "finance costs"]},
    {"metric_name": "gross_profit", "aliases": ["net interest income", "nii", "gross profit"]},
    {"metric_name": "operating_expenses_total", "aliases": ["operating expenses", "total expenditure", "total expenses"]},
    {"metric_name": "employee_costs", "aliases": ["employee benefit expenses", "payments to and provisions for employees", "staff costs"]},
    {"metric_name": "selling_marketing_expense", "aliases": ["selling and distribution expenses", "advertisement", "business promotion"]},
    {"metric_name": "rnd_expense", "aliases": ["research and development", "r&d expenses"]},
    {"metric_name": "general_admin_expense", "aliases": ["general and administrative expenses", "other expenses", "other operating expenses"]},
    {"metric_name": "operating_income", "aliases": ["operating profit", "profit before provisions and tax", "ebit"]},
    {"metric_name": "interest_expense", "aliases": ["finance costs", "interest costs"]},
    {"metric_name": "tax_expense", "aliases": ["tax expense", "total tax expenses", "provision for tax"]},
    {"metric_name": "net_income", "aliases": ["net profit", "profit after tax", "net profit for the period", "pat"]},
    
    # Cash Flow & Balance Sheet (Standard)
    {"metric_name": "cash_from_operations", "aliases": ["net cash flow from operating activities"]},
    {"metric_name": "capital_expenditure", "aliases": ["purchase of property, plant and equipment", "capex"]},
    {"metric_name": "cash_from_investing", "aliases": ["net cash flow from investing activities"]},
    {"metric_name": "cash_from_financing", "aliases": ["net cash flow from financing activities"]},
    {"metric_name": "dividends_paid", "aliases": ["dividend paid", "dividends on equity shares"]},
    {"metric_name": "net_change_in_cash", "aliases": ["net increase/(decrease) in cash"]},
    {"metric_name": "cash_end_period", "aliases": ["cash and cash equivalents at the end of the period"]},
    
    {"metric_name": "cash_and_equivalents", "aliases": ["cash and bank balances", "cash and cash equivalents"]},
    {"metric_name": "short_term_investments", "aliases": ["current investments", "investments"]},
    {"metric_name": "total_current_assets", "aliases": ["total current assets"]},
    {"metric_name": "total_assets", "aliases": ["total assets", "grand total - assets"]},
    {"metric_name": "short_term_debt", "aliases": ["short term borrowings"]},
    {"metric_name": "long_term_debt", "aliases": ["long term borrowings"]},
    {"metric_name": "total_liabilities", "aliases": ["total liabilities", "total capital and liabilities"]},
    {"metric_name": "shareholders_equity", "aliases": ["total equity", "net worth", "share capital + reserves"]},
    
    # Operational
    {"metric_name": "employee_count", "aliases": ["number of employees", "headcount"]},
    {"metric_name": "avg_employee_cost", "aliases": ["cost per employee"]},
    {"metric_name": "branch_count", "aliases": ["number of branches", "banking outlets"]},
    {"metric_name": "customer_count", "aliases": ["number of customers", "customer base"]},
    {"metric_name": "segment_revenue", "aliases": ["revenue by segment"]}
]

# 2. HELPER: ROBUST Period Finder
def extract_period_from_text(text):
    text_lower = text.lower()
    
    # PATTERN: "Ended 30th September 2023"
    matches = re.findall(r"(\d{1,2})?[\s-]*([a-z]+)[\s-]*(\d{1,2})?,?[\s-]*(\d{4})", text_lower[:2000]) 
    
    months_map = {
        "january": ("Q4", 0), "february": ("Q4", 0), "march": ("Q4", 0),
        "april": ("Q1", 0), "may": ("Q1", 0), "june": ("Q1", 0),
        "july": ("Q2", 0), "august": ("Q2", 0), "september": ("Q2", 0),
        "october": ("Q3", 0), "november": ("Q3", 0), "december": ("Q3", 0)
    }

    best_period = "Unknown Period"
    
    if matches:
        for match in matches:
            parts = [p for p in match if p]
            
            found_month = None
            found_year = None
            
            for p in parts:
                if p in months_map:
                    found_month = p
                elif len(p) == 4 and p.isdigit():
                    found_year = int(p)
            
            if found_month and found_year:
                q_label, year_offset = months_map[found_month]
                fiscal_year = found_year
                if found_month in ["january", "february", "march"]:
                    fiscal_year = found_year 
                else:
                    fiscal_year = found_year + 1 
                
                best_period = f"{q_label} FY{str(fiscal_year)[2:]} ({found_month.title()} {found_year})"
                return best_period 

    return best_period

# 3. HELPER: Page Selector
def get_filtered_text(pdf_path):
    print(f"📖 Scanning {os.path.basename(pdf_path)}...")
    relevant_text = ""
    keywords = ["revenue", "income", "profit", "assets", "liabilities", "cash flow", "consolidated", "standalone"]
    
    context_pages_text = "" 
    pages_kept = 0

    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text: continue
                
                if i < 2: context_pages_text += text + "\n"

                score = 0
                text_lower = text.lower()
                for kw in keywords:
                    if kw in text_lower: score += 1
                
                tables = page.extract_tables()
                if tables: score += 5
                
                if score >= 6 or i < 4:
                    if tables:
                        relevant_text += f"\n--- DATA PAGE {i+1} ---\n"
                        for table in tables:
                            for row in table:
                                clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                                relevant_text += " | ".join(clean_row) + "\n"
                    relevant_text += f"\n--- TEXT PAGE {i+1} ---\n{text}\n"
                    pages_kept += 1
                    
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return "", ""
    
    return relevant_text, context_pages_text

# 4. CORE: AI Extraction
async def process_file(client, pdf_path):
    filename = os.path.basename(pdf_path)
    filtered_context, context_pages = get_filtered_text(pdf_path)
    
    detected_period = extract_period_from_text(context_pages)
    print(f"   📅 Detected: {detected_period}")
    
    if len(filtered_context) < 50: return None

    schema_str = json.dumps(METRICS_SCHEMA, indent=2)

    system_prompt = """
    You are a Financial Data Engine.
    
    TASK: Extract 32 exact metrics for Monte Carlo simulation.
    
    ────────────────────────
    BANKING MAPPING RULES
    ────────────────────────
    1. REVENUE: Sum "Interest Earned" + "Other Income".
    2. COST OF REVENUE: Use "Interest Expended".
    3. GROSS PROFIT: Use "Net Interest Income" (NII).
    4. ADMIN EXPENSE: If "Marketing" is missing, put ALL "Other Operating Expenses" here.
    
    ────────────────────────
    FORMATTING RULES
    ────────────────────────
    1. SCALE: Convert "Lakhs" (*100,000) or "Crores" (*10,000,000) to absolute integers.
    2. MISSING DATA: Return 0 (Zero). Do NOT use N/A.
    3. FORMAT: CSV ONLY (metric_name, value).
    """

    user_message = f"""
    CONTEXT:
    {filtered_context[:95000]}

    SCHEMA:
    {schema_str}

    INSTRUCTIONS:
    1. Extract for period: "{detected_period}".
    2. If "Gross Profit" row is missing, calculate it as (Revenue - Cost of Revenue).
    3. Output CSV.
    """

    for attempt in range(3):
        try:
            assistant_obj = await client.create_assistant(
                name="FinEngineV10", 
                system_prompt=system_prompt
            )
            thread = await client.create_thread(assistant_id=assistant_obj.assistant_id)
            print(f"   🤖 AI Analysis: {filename} ({detected_period})...")
            
            response = await client.add_message(
                thread_id=thread.thread_id,
                content=user_message
            )
            
            content = getattr(response, 'content', str(response)).strip()
            content = content.replace("```csv", "").replace("```", "").strip()
            
            data = {"period": detected_period, "source_file": filename}
            
            f = io.StringIO(content)
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    metric = row[0].strip()
                    val = row[1].strip()
                    val = re.sub(r"[^0-9.-]", "", val) 
                    
                    if not val: val = "0"
                    
                    if any(m['metric_name'] == metric for m in METRICS_SCHEMA):
                        data[metric] = float(val)

            return data

        except Exception as e:
            print(f"   ⚠️ Error (Attempt {attempt+1}): {e}")
            time.sleep(2)

    return None

# 5. MAIN + PYTHON MATH REPAIR
async def main():
    client = BackboardClient(api_key=API_KEY)
    
    pdf_files = [f for f in os.listdir(PDF_FOLDER_PATH) if f.lower().endswith('.pdf')]
    print(f"📂 Found {len(pdf_files)} PDF files.")
    
    all_results = []
    
    for pdf_file in pdf_files:
        full_path = os.path.join(PDF_FOLDER_PATH, pdf_file)
        result = await process_file(client, full_path)
        
        if result:
            all_results.append(result)
            save_to_csv(all_results) # Changed to CSV

def save_to_csv(results):
    if not results: return
    schema_cols = [m['metric_name'] for m in METRICS_SCHEMA]
    columns = ["period"] + schema_cols + ["source_file"]
    
    df = pd.DataFrame(results)
    
    # 1. Fill Missing with 0
    for col in columns:
        if col not in df.columns: df[col] = 0.0
        
    # 2. PYTHON MATH REPAIR
    df['gross_profit'] = df.apply(
        lambda row: row['revenue_total'] - row['cost_of_revenue'] if row['gross_profit'] == 0 else row['gross_profit'], axis=1
    )
    
    df['operating_expenses_total'] = df.apply(
        lambda row: (row['employee_costs'] + row['selling_marketing_expense'] + row['general_admin_expense']) 
        if row['operating_expenses_total'] == 0 else row['operating_expenses_total'], axis=1
    )
    
    df = df[columns]
    try: df = df.sort_values(by="period")
    except: pass
    
    # --- CHANGED: SAVE AS CSV ---
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"   💾 CSV Updated with Math Repair.")

if __name__ == "__main__":
    asyncio.run(main())