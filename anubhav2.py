import csv
import asyncio
import pdfplumber
from backboard import BackboardClient

# 1. SETUP: Your Schema
metrics_schema = [
    {"metric_name": "revenue_total", "aliases": ["total revenue", "net sales", "turnover"]},
    {"metric_name": "cost_of_revenue", "aliases": ["cost of goods sold", "cogs", "cost of sales"]},
    {"metric_name": "gross_profit", "aliases": ["gross profit", "gross margin"]},
    {"metric_name": "operating_expenses_total", "aliases": ["total operating expenses", "opex"]},
    {"metric_name": "net_income", "aliases": ["net income", "net profit", "loss for the period"]},
    {"metric_name": "cash_from_operations", "aliases": ["net cash from operating", "cash flow from operations"]},
    {"metric_name": "capital_expenditure", "aliases": ["purchase of property", "capex"]},
    {"metric_name": "cash_end_period", "aliases": ["cash and cash equivalents at end", "ending cash balance"]},
    {"metric_name": "total_assets", "aliases": ["total assets"]},
    {"metric_name": "total_liabilities", "aliases": ["total liabilities"]},
    {"metric_name": "shareholders_equity", "aliases": ["total equity", "shareholders' equity"]},
    {"metric_name": "employee_count", "aliases": ["employees", "headcount"]}
]

# 2. EXTRACT ONLY RELEVANT PAGES (The Cost Saver)
def get_filtered_text(pdf_path):
    print(f"📖 Scanning PDF for relevant financial pages: {pdf_path}")
    relevant_text = ""
    
    # Keywords that usually appear on the important pages
    # We trigger extraction ONLY if we see these headers
    target_headers = [
    # --- 1. Income Statement Variations ---
    "consolidated statements of operations",
    "consolidated statement of income",
    "consolidated statements of income",
    "statement of earnings",
    "consolidated statement of comprehensive income",
    "profit and loss",
    "income statement",
    "results of operations",

    # --- 2. Balance Sheet Variations ---
    "consolidated balance sheets",
    "consolidated statement of financial position",
    "balance sheet",
    "financial position",
    "statement of financial condition",
    "assets and liabilities",

    # --- 3. Cash Flow Variations ---
    "consolidated statements of cash flows",
    "consolidated statement of cash flows",
    "cash flow statement",
    "statement of cash flows",
    "cash flows",

    # --- 4. Equity (Crucial for Share Counts/Dividends) ---
    "consolidated statements of stockholders' equity",
    "consolidated statements of shareholders' equity",
    "statement of changes in equity",
    "consolidated statement of equity",

    # --- 5. Key Summaries (Where the "easy" numbers live) ---
    "our key figures",
    "financial highlights",
    "selected financial data",
    "key performance indicators",
    "financial summary",
    "performance highlights",
    "metric highlights",
    
    # --- 6. Specific Data Sections ---
    "segment information",   # <--- CRITICAL for your 'segment_revenue' field
    "revenue by geography",
    "shareholder information"
]
    
    pages_kept = 0
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                text = page.extract_text()
                if not text: 
                    continue
                
                # Check if this page has one of our target headers
                # We convert to lower case for case-insensitive matching
                page_lower = text.lower()
                if any(header in page_lower for header in target_headers):
                    print(f"   ✅ Keeping Page {i+1} (Found financial data)")
                    
                    # Extract tables specifically from this page
                    tables = page.extract_tables()
                    if tables:
                        relevant_text += f"\n--- DATA FROM PAGE {i+1} ---\n"
                        for table in tables:
                            for row in table:
                                clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                                relevant_text += " | ".join(clean_row) + "\n"
                    
                    # Also keep raw text for context
                    relevant_text += text + "\n"
                    pages_kept += 1
                else:
                    # Skip marketing/legal pages to save money
                    pass
                    
    except Exception as e:
        print(f"❌ Error: {e}")
        return ""
    
    print(f"📉 Reduced document from {len(pdf.pages)} pages to {pages_kept} pages.")
    return relevant_text

# 3. ASYNC MAIN
async def main():
    pdf_path = "/Users/anubhavkayal/Data Scrapping/reportF.pdf"
    
    # Get ONLY the important pages
    filtered_context = get_filtered_text(pdf_path)
    
    if len(filtered_context) < 50:
        print("❌ CRITICAL: No financial tables found! The specific headers might be named differently in this PDF.")
        # Fallback: If filter fails, maybe take the first 10 pages?
        # But for now, let's stop to save money.
        return

    # Initialize Backboard
    # CRITICAL: If your key is dead, you MUST get a new one.
    client = BackboardClient(api_key="espr_Hlape8JgiHiwIEt-Ash9LE55tIyTECHENuUHRAYjlKU") 

    print("🤖 Creating Assistant...")
    assistant = await client.create_assistant(
        name="Financial Auditor",
        system_prompt="You are an expert financial auditor. Extract metrics from the provided table data. Return JSON only."
    )
    
    thread = await client.create_thread(assistant_id=assistant.assistant_id)
    
    # Upload smaller context
    print(f"📤 Uploading filtered context ({len(filtered_context)} chars)...")
    await client.add_message(
        thread_id=thread.thread_id,
        content=f"FINANCIAL STATEMENTS:\n{filtered_context}"
    )

    extracted_data = {}
    print("🔍 Extracting metrics (Single Batch Request)...")

    # COST SAVING: Ask for EVERYTHING in ONE request to minimize API calls
    fields_list = ", ".join([item['metric_name'] for item in metrics_schema])
    
    prompt = f"""
    Analyze the uploaded financial statements.
    Extract the following metrics: {fields_list}
    
    Return a single JSON object with the keys.
    If a value is missing, use "N/A".
    RETURN ONLY JSON.
    """
    
    try:
        response = await client.add_message(
            thread_id=thread.thread_id,
            content=prompt
        )
        
        content = getattr(response, 'content', str(response)).strip()
        # Clean markdown
        content = content.replace("```json", "").replace("```", "").strip()
        
        import json
        data = json.loads(content)
        extracted_data = data
        print("✅ Success!")
        print(data)

    except Exception as e:
        print(f"❌ Error: {e}")
        # Print the raw response to see if it's another billing error
        if 'content' in locals():
            print(f"Raw response: {content}")

    # Save to CSV
    headers = [item['metric_name'] for item in metrics_schema]
    # Ensure all keys exist
    final_row = {h: extracted_data.get(h, "N/A") for h in headers}
    
    with open('financial_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerow(final_row)
        
    print("DONE! Check financial_data.csv")

if __name__ == "__main__":
    asyncio.run(main())