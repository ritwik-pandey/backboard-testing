import os
import json
import asyncio
import re
from backboard import BackboardClient

_client = None

def get_client():
    global _client
    if not _client:
        api_key = os.getenv("espr_nefwEDUC6L1VeXM3tqN3c2TA6V55BjIwOIQAipB3uU0")
        if not api_key:
            # Replace with your actual key if env var is missing
            api_key = "espr_nefwEDUC6L1VeXM3tqN3c2TA6V55BjIwOIQAipB3uU0"
        _client = BackboardClient(api_key=api_key)
    return _client

async def extract_metadata_real(file_path, header_row_index):
    """
    Uses Backboard AI with Regex-based JSON extraction to prevent parsing errors.
    """
    client = get_client()
    filename = os.path.basename(file_path)

    try:
        assistant = await client.create_assistant(
            name="FinDoc_Extractor",
            system_prompt="You are a strict JSON extractor. Output ONLY valid JSON."
        )

        thread = await client.create_thread(assistant.assistant_id)

        prompt = f"""
        Read the header (top 20 lines) of '{filename}'. 
        Extract these exact fields into JSON:
        {{
            "bank_name": "string or null",
            "account_holder": "string or null",
            "account_number": "string or null",
            "currency": "INR/USD",
            "period_start": "date string or null",
            "period_end": "date string or null"
        }}
        """

        response = await client.add_message(
            thread_id=thread.thread_id,
            content=prompt,
            files=[file_path]
        )

        raw_content = response.content
        
        # --- FIX: REGEX TO FIND JSON BLOCK ---
        # This looks for the first '{' and the last '}' regardless of markdown
        json_match = re.search(r'\{.*\}', raw_content, re.DOTALL)
        
        if json_match:
            clean_json_str = json_match.group(0)
            return json.loads(clean_json_str)
        else:
            # Fallback if regex fails (rare)
            return {"error": "No JSON found in response", "raw": raw_content[:100]}

    except Exception as e:
        print(f"⚠️ Intelligence Error on {filename}: {e}")
        return {"error": str(e), "source": "Backboard_Fallback"}