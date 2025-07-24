import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from langdetect import detect
import re
from collections import deque

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'pmc-bot-index')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize Pinecone and Gemini
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)
genai.configure(api_key=GEMINI_API_KEY)

# Settings
TOP_K = 5
CONTEXT_RESULTS = 3  # Number of Pinecone results to pass to LLM
MAX_HISTORY = 2      # Number of previous turns to include

# Keywords for 'recent/latest' intent
def is_latest_query(query):
    LATEST_KEYWORDS = [
        r"latest", r"most recent", r"newest", r"recent", r"last", r"current", r"today(?:'s)?", r"this week", r"fresh", r"updated", r"up-to-date", r"just published", r"just released", r"recently issued", r"recently published", r"recently released"
    ]
    pattern = re.compile(r"|".join(LATEST_KEYWORDS), re.IGNORECASE)
    return bool(pattern.search(query))

# Helper: Detect language
def detect_language(text):
    try:
        lang = detect(text)
        if lang == 'mr':
            return 'mr'
        return 'en'
    except Exception:
        return 'en'

# Helper: Embed query
def embed_query(text):
    response = genai.embed_content(model='models/embedding-001', content=text, task_type='retrieval_query')
    return response['embedding']

# Helper: Format Pinecone results for LLM context (with links)
def format_pinecone_results(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get('metadata', {})
        # Serialize the entire metadata dict, pretty-printed for readability
        meta_json = json.dumps(meta, ensure_ascii=False, indent=2)
        formatted.append(f"Result {i} (full record):\n{meta_json}")
    return '\n---\n'.join(formatted)

# Helper: Build LLM prompt
def build_llm_prompt(user_query, pinecone_context, chat_history, lang):
    prompt = f"""
User Query:
{user_query}

Relevant Information from PMC Database:
{pinecone_context}
"""
    if chat_history:
        prompt += "\nPrevious Conversation Context:\n"
        for turn in chat_history:
            prompt += f"User: {turn['user']}\nBot: {turn['bot']}\n"
    prompt += f"""
Instructions:
- Carefully read the information above and answer the user's query in a clear, friendly, and human-like manner.
- Summarize and explain the relevant details in simple language, rather than listing raw data, JSON, or technical field names.
- If a link is present and relevant, include it naturally in your answer (e.g., 'You can find more details here: [link]').
- Do not repeat unnecessary technical details or field names.
- If the answer is not found in the context, politely state that the information is not available.
- Respond in {'Marathi' if lang == 'mr' else 'English'}.
"""
    return prompt

# Main chat loop
def main():
    print("PMC Chatbot v2 (English/Marathi). Type 'exit' to quit.")
    history = deque(maxlen=MAX_HISTORY)
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not user_input:
            continue  # Skip empty input
        lang = detect_language(user_input)
        # Embed and search Pinecone
        query_emb = embed_query(user_input)
        results = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)
        docs = results.get('matches', [])
        # If 'recent/latest' intent, sort by date
        if is_latest_query(user_input):
            def parse_date_safe(meta):
                d = meta.get('metadata', {})
                return d.get('date') or d.get('display_date') or ''
            docs = sorted(docs, key=parse_date_safe, reverse=True)
        # Use top N results for context
        context_docs = docs[:CONTEXT_RESULTS]
        pinecone_context = format_pinecone_results(context_docs) if context_docs else "No relevant information found."
        # Prepare chat history for LLM
        chat_history = list(history)
        # Build LLM prompt
        prompt = build_llm_prompt(user_input, pinecone_context, chat_history, lang)
        # Generate answer using Gemini
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        answer = response.text.strip()
        print(f"Bot: {answer}")
        # Update history
        history.append({'user': user_input, 'bot': answer})

if __name__ == '__main__':
    main() 