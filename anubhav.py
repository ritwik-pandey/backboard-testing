import csv
import asyncio
from backboard import BackboardClient
from pypdf import PdfReader

# 1. SETUP: Define your "Target Fields"
target_fields = [
    # A. Income Statement
    "revenue_total", "cost_of_revenue", "gross_profit", 
    "operating_expenses_total", "employee_costs", "selling_marketing_expense",
    "rnd_expense", "general_admin_expense", "operating_income", 
    "interest_expense", "tax_expense", "net_income",

    # B. Cash Flow
    "cash_from_operations", "capital_expenditure", "cash_from_investing",
    "cash_from_financing", "dividends_paid", "net_change_in_cash",
    "cash_end_period",

    # C. Balance Sheet
    "cash_and_equivalents", "short_term_investments", "total_current_assets",
    "total_assets", "short_term_debt", "long_term_debt", 
    "total_liabilities", "shareholders_equity",

    # D. Operational Metrics
    "employee_count", "avg_employee_cost", "branch_count", 
    "customer_count", "segment_revenue"
]

# 2. EXTRACT TEXT
def get_pdf_text(pdf_path):
    print(f"Reading PDF from: {pdf_path}")
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text()
    return text

# --- HELPER: CHUNK TEXT ---
def chunk_text(text, chunk_size=1500):
    return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

# 3. ASYNC MAIN FUNCTION (DEBUGGED VERSION)
async def main():
    # --- CHECK THIS PATH ---
    pdf_path = "/Users/anubhavkayal/Data Scrapping/financial_report_Q3.pdf"
    
    # [DIAGNOSIS 1] Check if PDF is actually readable
    try:
        raw_financial_text = get_pdf_text(pdf_path)
        print(f"\n--- PDF DIAGNOSTIC ---")
        print(f"Total Characters Extracted: {len(raw_financial_text)}")
        print(f"First 500 chars preview:\n{raw_financial_text[:500]}")
        print(f"----------------------\n")
        
        if len(raw_financial_text.strip()) < 100:
            print("❌ CRITICAL ERROR: The PDF text is empty or too short!")
            print("This PDF is likely a SCANNED IMAGE. 'pypdf' cannot read images.")
            print("You need to use an OCR tool or a text-based PDF.")
            return

    except FileNotFoundError:
        print(f"ERROR: Could not find file at {pdf_path}")
        return

    print("PDF valid. Initializing Backboard...")

    # Initialize Client
    client = BackboardClient(api_key="YOUR_BACKBOARD_API_KEY") 

    # Create Assistant
    print("Creating Assistant...")
    assistant = await client.create_assistant(
        name="Financial Analyst",
        system_prompt="You are a helpful financial analyst. You have access to a financial report in your memory. Use it to answer questions. If the value is not explicitly stated, try to infer it from context, or return 'N/A' only as a last resort."
    )
    print(f"Assistant Created: {assistant.assistant_id}")

    # Create Thread
    print("Creating Thread...")
    thread = await client.create_thread(assistant_id=assistant.assistant_id)
    print(f"Thread Created: {thread.thread_id}")

    # Upload Memory
    print("Splitting text into chunks...")
    text_chunks = chunk_text(raw_financial_text, chunk_size=1500)
    
    print(f"Uploading {len(text_chunks)} memory chunks...")
    for i, chunk in enumerate(text_chunks):
        # We add a tiny marker so we know which chunk is which
        print(f"  - Uploading chunk {i+1}/{len(text_chunks)}...")
        await client.add_memory(
            assistant_id=assistant.assistant_id,
            content=chunk
        )
        await asyncio.sleep(0.2)
    
    print("All memory uploaded.")
    
    # [DIAGNOSIS 2] The Indexing Nap
    print("💤 Waiting 20 seconds for Backboard to index the data...")
    await asyncio.sleep(20)
    print("Wake up! Starting extraction...")

    extracted_data = {}

    # Loop through fields
    for field in target_fields:
        prompt = f"Search the uploaded financial document memory for '{field}'. Return ONLY the numerical value. If not found, return 'N/A'."
        
        response = await client.add_message(
            thread_id=thread.thread_id,
            content=prompt
        )
        
        if hasattr(response, 'content'):
            value = response.content.strip()
        else:
            value = str(response).strip()

        extracted_data[field] = value
        print(f"Found {field}: {value}")

    # Save to CSV
    output_filename = 'financial_data.csv'
    with open(output_filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=target_fields)
        writer.writeheader()
        writer.writerow(extracted_data)

    print(f"Done! Data saved to {output_filename}")
if __name__ == "__main__":
    asyncio.run(main())