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

def is_latest_query(query):
    LATEST_KEYWORDS = [
        r"latest", r"most recent", r"newest", r"recent", r"last", r"current", r"today(?:'s)?", r"this week", r"fresh", r"updated", r"up-to-date", r"just published", r"just released", r"recently issued", r"recently published", r"recently released"
    ]
    pattern = re.compile(r"|".join(LATEST_KEYWORDS), re.IGNORECASE)
    return bool(pattern.search(query))

def detect_language(text):
    try:
        lang = detect(text)
        if lang == 'mr':
            return 'mr'
        return 'en'
    except Exception:
        return 'en'

def embed_query(text):
    response = genai.embed_content(model='models/embedding-001', content=text, task_type='retrieval_query')
    return response['embedding']

def format_pinecone_results(docs):
    formatted = []
    for i, doc in enumerate(docs, 1):
        meta = doc.get('metadata', {})
        title = meta.get('title', '')
        description = meta.get('description', '')
        date = meta.get('date', meta.get('display_date', ''))
        department = meta.get('department', '')
        # Prefer PDF, then external, then url
        link = meta.get('pdf_url') or meta.get('external_link') or meta.get('url')
        s = f"Record {i}:\n"
        if title:
            s += f"Title: {title}\n"
        if description:
            s += f"Description: {description}\n"
        if date:
            s += f"Date: {date}\n"
        if department:
            s += f"Department: {department}\n"
        if link:
            s += f"Link: {link}\n"
        formatted.append(s.strip())
    return '\n---\n'.join(formatted)

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
- Carefully read all the records above and answer the user's query in a clear, friendly, and human-like manner.
- Use any record that is relevant to the user's question, even if only partially.
- The most relevant records are listed first.
- Summarize and explain the relevant details in simple language, not as raw data or JSON.
- If multiple records are relevant, summarize or list them as appropriate.
- If a link is present and relevant, include it naturally in your answer (e.g., 'You can find more details here: [link]').
- Do not say 'not found' if any relevant information is present in the context.
- Only if there is truly no relevant information, politely state that the information is not available.
- Respond in {'Marathi' if lang == 'mr' else 'English'}.
"""
    return prompt

def main():
    print("PMC Chatbot DEBUG (English/Marathi). Type 'exit' to quit.")
    history = deque(maxlen=MAX_HISTORY)
    while True:
        user_input = input("You: ").strip()
        if not user_input:
            continue  # Skip empty input
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        lang = detect_language(user_input)
        query_emb = embed_query(user_input)
        results = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)
        docs = results.get('matches', [])
        # DEBUG: Print the titles and IDs of the top K Pinecone results
        print("\n[DEBUG] Top Pinecone Results:")
        for i, doc in enumerate(docs, 1):
            meta = doc.get('metadata', {})
            print(f"  {i}. Title: {meta.get('title')} | ID: {meta.get('id')}")
        print()
        if is_latest_query(user_input):
            def parse_date_safe(meta):
                d = meta.get('metadata', {})
                return d.get('date') or d.get('display_date') or ''
            docs = sorted(docs, key=parse_date_safe, reverse=True)
        filtered_docs = docs
        if lang == 'mr':
            marathi_docs = [doc for doc in docs if doc.get('metadata', {}).get('lang') == 'mr']
            if marathi_docs:
                filtered_docs = marathi_docs
        context_docs = filtered_docs[:CONTEXT_RESULTS]
        pinecone_context = format_pinecone_results(context_docs) if context_docs else "No relevant information found."
        chat_history = list(history)
        prompt = build_llm_prompt(user_input, pinecone_context, chat_history, lang)
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        answer = response.text.strip()
        print(f"Bot: {answer}")
        history.append({'user': user_input, 'bot': answer})

if __name__ == '__main__':
    main() 