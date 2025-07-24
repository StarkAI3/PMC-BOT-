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

def is_followup_query(query, prev_user_query):
    pronouns = ["it", "that", "this", "one", "which", "above", "them", "those", "these"]
    q_lower = query.lower().strip()
    # If no previous user query, can't be a follow-up
    if not prev_user_query:
        return False
    # If query is very short and contains a pronoun, likely a follow-up
    if len(q_lower.split()) <= 6 and any(p in q_lower for p in pronouns):
        return True
    # If query is a full question (starts with 'which', 'what', etc.) and is long enough, treat as standalone
    question_words = ["which", "what", "where", "who", "when", "how"]
    if any(q_lower.startswith(w) for w in question_words) and len(q_lower.split()) > 6:
        return False
    # Default: not a follow-up
    return False

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
- Carefully read all the records above and answer the user's query in a clear, friendly, and human-like manner.
- You may use the previous conversation context, the retrieved records, or both, as appropriate to answer the user's query. Decide what is most relevant for the current query.
- Use any record that is relevant to the user's question, even if only partially.
- The most relevant records are listed first.
- Summarize and explain the relevant details in simple language, not as raw data or JSON.
- If multiple records are relevant, summarize or list them as appropriate.
- When including a link, use Markdown format with descriptive text, e.g. [here](URL) or [link](URL), not [URL](URL).
- Embed the link in the most relevant word or phrase, not as a raw URL.
- If a link is already included in your answer, do not repeat it. Avoid repeating the same link multiple times in your response.
- Do not say 'not found' if any relevant information is present in the context.
- Only if there is truly no relevant information, politely state that the information is not available.
- Respond in {'Marathi' if lang == 'mr' else 'English'}.
"""
    return prompt

def replace_url_markdown_with_here(text):
    # Replace [URL](URL) with [here](URL)
    def replacer(match):
        url = match.group(1)
        return f"[here]({url})"
    # Regex: \[https?://...\]\(https?://...\)
    return re.sub(r'\[(https?://[^\]]+)\]\(\1\)', replacer, text)

def remove_duplicate_links(text):
    urls = re.findall(r'https?://[^\s)]+', text)
    seen = set()
    def url_replacer(match):
        url = match.group(0)
        if url in seen:
            return ''  # Remove duplicate
        seen.add(url)
        return url
    # Replace duplicate URLs with empty string
    return re.sub(r'https?://[^\s)]+', url_replacer, text)

# Main chat loop
def main():
    print("PMC Chatbot v2 (English/Marathi). Type 'exit' to quit.")
    history = deque(maxlen=MAX_HISTORY)
    prev_user_query = None
    prev_bot_answer = None
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        if not user_input:
            continue  # Skip empty input
        lang = detect_language(user_input)
        # Generalized follow-up handling
        query_for_search = user_input
        if is_followup_query(user_input, prev_user_query):
            # Combine previous user query and/or bot answer with current query
            query_for_search = f"{user_input.strip()} (context: {prev_user_query})"
        # Embed and search Pinecone
        query_emb = embed_query(query_for_search)
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
        answer = remove_duplicate_links(answer)
        answer = replace_url_markdown_with_here(answer)
        print(f"Bot: {answer}")
        # Update history and previous queries
        history.append({'user': user_input, 'bot': answer})
        prev_user_query = user_input
        prev_bot_answer = answer

if __name__ == '__main__':
    main() 