import csv
import asyncio
import pdfplumber
from backboard import BackboardClient

# 1. SETUP: Your Schema
metrics_schema = [
    {"metric_name": "revenue_total", "aliases": ["total revenue", "net sales", "turnover", "operating revenue"]},
    {"metric_name": "cost_of_revenue", "aliases": ["cost of goods sold", "cogs", "cost of sales"]},
    {"metric_name": "gross_profit", "aliases": ["gross profit", "gross margin"]},
    {"metric_name": "operating_expenses_total", "aliases": ["total operating expenses", "opex"]},
    {"metric_name": "net_income", "aliases": ["net income", "net profit", "loss for the period"]},
    {"metric_name": "cash_from_operations", "aliases": ["net cash from operating", "cash flow from operations"]},
    {"metric_name": "capital_expenditure", "aliases": ["purchase of property", "capex", "additions to property"]},
    {"metric_name": "cash_end_period", "aliases": ["cash and cash equivalents at end", "ending cash balance"]},
    {"metric_name": "total_assets", "aliases": ["total assets"]},
    {"metric_name": "total_liabilities", "aliases": ["total liabilities"]},
    {"metric_name": "shareholders_equity", "aliases": ["total equity", "shareholders' equity"]},
    {"metric_name": "employee_count", "aliases": ["employees", "headcount", "full-time employees"]}
]

# 2. EXTRACT TEXT + TABLES (The Secret Sauce)
def get_structured_text(pdf_path):
    print(f"📖 Reading PDF layout: {pdf_path}")
    full_context = ""
    
    try:
        with pdfplumber.open(pdf_path) as pdf:
            for i, page in enumerate(pdf.pages):
                # A. Extract Tables specifically
                tables = page.extract_tables()
                if tables:
                    full_context += f"\n--- TABLE DATA (PAGE {i+1}) ---\n"
                    for table in tables:
                        # Convert table to CSV format string which LLMs understand easily
                        for row in table:
                            # Filter out None values and join
                            clean_row = [str(cell).replace('\n', ' ') if cell else "" for cell in row]
                            full_context += " | ".join(clean_row) + "\n"
                
                # B. Extract Regular Text (for things not in tables like 'Employee Count')
                text = page.extract_text()
                if text:
                    full_context += f"\n--- TEXT DATA (PAGE {i+1}) ---\n{text}\n"
                    
    except Exception as e:
        print(f"❌ Error reading PDF: {e}")
        return ""
        
    return full_context

# 3. ASYNC MAIN
async def main():
    pdf_path = "/Users/anubhavkayal/Data Scrapping/reportF.pdf"
    
    # Get structured context
    structured_text = get_structured_text(pdf_path)
    
    if len(structured_text) < 50:
        print("❌ CRITICAL: Extracted text is empty. OCR might be bad or file is encrypted.")
        return

    # Initialize Backboard
    # SECURITY NOTE: Replace with your actual key
    client = BackboardClient(api_key="API KEY HERE") 

    print("🤖 Creating Assistant...")
    assistant = await client.create_assistant(
        name="Financial Auditor",
        system_prompt="""
        You are an expert financial auditor. 
        You are given raw text and structured table data from a PDF.
        Your goal is to find specific financial metrics.
        
        RULES:
        1. Context is provided as 'TABLE DATA' (rows separated by |) and 'TEXT DATA'.
        2. When searching for a value, prioritize the 'Current Period' or '2023/2024' column if multiple years exist.
        3. Return ONLY the numerical value formatted as a standard number (e.g., 1000000).
        4. If the value is strictly not present, return 'N/A'.
        """
    )
    
    # Create Thread
    thread = await client.create_thread(assistant_id=assistant.assistant_id)
    
    # --- CONTEXT STUFFING ---
    # We send the text to the thread so the AI "sees" it before we ask questions.
    # Note: If text is > 20k chars, we might need to split it, but let's try this first.
    print(f"📤 Uploading document context ({len(structured_text)} chars)...")
    
    # Chunking specifically for the message limit (not memory limit)
    chunk_size = 6000
    chunks = [structured_text[i:i+chunk_size] for i in range(0, len(structured_text), chunk_size)]
    
    for i, chunk in enumerate(chunks):
        await client.add_message(
            thread_id=thread.thread_id,
            content=f"DOCUMENT PART {i+1}:\n{chunk}"
        )
        await asyncio.sleep(0.5)

    extracted_data = {}
    print("🔍 Extracting metrics...")

    # OPTIMIZATION: Ask for multiple fields at once to save time/tokens
    # We group fields into batches of 4
    batch_size = 4
    for i in range(0, len(metrics_schema), batch_size):
        batch = metrics_schema[i:i+batch_size]
        
        # Build a composite prompt
        fields_str = ""
        for item in batch:
            fields_str += f"- {item['metric_name']} (Look for: {', '.join(item['aliases'])})\n"
            
        prompt = f"""
        Extract values for the following metrics from the document context provided above.
        
        METRICS TO FIND:
        {fields_str}
        
        OUTPUT FORMAT:
        Return a JSON object with the metric names as keys. 
        Example: {{"revenue_total": "1500000", "cost_of_revenue": "N/A"}}
        Do not include markdown formatting like ```json. Just the raw JSON string.
        """
        
        print(f"   Requesting batch: {[b['metric_name'] for b in batch]}...")
        
        try:
            response = await client.add_message(
                thread_id=thread.thread_id,
                content=prompt
            )
            
            content = getattr(response, 'content', str(response)).strip()
            
            # Clean up JSON (sometimes AI adds markdown)
            content = content.replace("```json", "").replace("```", "").strip()
            
            # Parse the JSON response
            import json
            try:
                data = json.loads(content)
                extracted_data.update(data)
                print(f"   ✅ Received: {data}")
            except json.JSONDecodeError:
                print(f"   ⚠️ JSON Parse Error. Raw response: {content}")
                # Fallback: Mark these as N/A if parsing fails
                for item in batch:
                    extracted_data[item['metric_name']] = "Error"
                    
        except Exception as e:
            print(f"❌ API Error: {e}")

    # Save to CSV
    # Ensure all fields are present (even if missed by AI)
    final_row = {field['metric_name']: extracted_data.get(field['metric_name'], "N/A") for field in metrics_schema}
    
    with open('financial_data.csv', 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=[i['metric_name'] for i in metrics_schema])
        writer.writeheader()
        writer.writerow(final_row)
        
    print("DONE! Check financial_data.csv")

if __name__ == "__main__":
    asyncio.run(main())