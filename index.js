import fs from "fs";
import { BackboardClient } from "backboard-sdk";

const client = new BackboardClient({
  apiKey: "",
});

const assistant = await client.createAssistant({
  name: "PDF Tuple Extractor",
  system_prompt: `
You extract structured data from PDFs.
Return ONLY tuples.
No explanations.
No markdown.
  `.trim(),
});

const thread = await client.createThread(assistant.assistantId);

const response = await client.addMessage(thread.threadId, {
  content: `
From the attached PDF, extract:
give me the data for Q4 2024 in tuple format  `.trim(),

    files: ["./report.pdf"],  

  llm_provider: "openai",
  model_name: "gpt-4o",
  stream: false,
});

console.log(response.content);
