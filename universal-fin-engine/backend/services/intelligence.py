import os
import json
import asyncio
from backboard import BackboardClient

_client = None

def get_client():
    global _client
    if not _client:
        api_key = os.getenv("espr_nefwEDUC6L1VeXM3tqN3c2TA6V55BjIwOIQAipB3uU0")
        if not api_key:
            # Fallback for hackathon
            api_key = "espr_nefwEDUC6L1VeXM3tqN3c2TA6V55BjIwOIQAipB3uU0"
        _client = BackboardClient(api_key=api_key)
    return _client

async def extract_metadata_real(file_path, header_row_index):
    """
    Uses Backboard AI to extract unstructured metadata.
    """
    client = get_client()
    filename = os.path.basename(file_path)

    try:
        # 1. Create Assistant (Removed 'memory' param to fix crash)
        assistant = await client.create_assistant(
            name="FinDoc_Extractor",
            system_prompt="Extract financial metadata as pure JSON."
        )

        # 2. Create Thread
        thread = await client.create_thread(assistant.assistant_id)

        # 3. Prompt
        prompt = f"""
        Analyze the header of '{filename}'. Return JSON:
        {{
            "bank_name": "string",
            "account_holder": "string",
            "account_type": "string",
            "currency": "string",
            "opening_balance": number
        }}
        """

        # 4. Add Message (Removed 'memory' param here too)
        response = await client.add_message(
            thread_id=thread.thread_id,
            content=prompt,
            files=[file_path]
        )

        # 5. Parse
        raw_content = response.content
        if "```json" in raw_content:
            raw_content = raw_content.split("```json")[1].split("```")[0]
        elif "```" in raw_content:
            raw_content = raw_content.split("```")[1].split("```")[0]
            
        return json.loads(raw_content.strip())

    except Exception as e:
        print(f"⚠️ Backboard AI Warning: {e}")
        return {"error": str(e), "source": "Backboard_Skipped"}