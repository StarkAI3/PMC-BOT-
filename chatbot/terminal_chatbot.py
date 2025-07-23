import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from tqdm import tqdm
import re
import spacy

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
MAX_HISTORY = 5

# Add this function to detect 'latest' intent
LATEST_KEYWORDS = [
    r"latest", r"most recent", r"newest", r"recent", r"last", r"current", r"today(?:'s)?", r"this week", r"fresh", r"updated", r"up-to-date", r"just published", r"just released", r"recently issued", r"recently published", r"recently released"
]
LATEST_PATTERN = re.compile(r"|".join(LATEST_KEYWORDS), re.IGNORECASE)

def is_latest_query(query):
    return bool(LATEST_PATTERN.search(query))

# Load spaCy English model for noun phrase extraction
try:
    nlp = spacy.load('en_core_web_sm')
except Exception:
    nlp = None

def extract_noun_phrases(text):
    if not nlp:
        return []
    doc = nlp(text)
    return [chunk.text.lower() for chunk in doc.noun_chunks]

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

# Helper: Format retrieved docs for LLM
def format_docs(docs):
    formatted = []
    for doc in docs:
        meta = doc['metadata']
        s = f"Title: {meta.get('title','')}."
        if meta.get('description'):
            s += f"\nDescription: {meta['description']}"
        if meta.get('pdf_url'):
            s += f"\nPDF: {meta['pdf_url']}"
        if meta.get('date'):
            s += f"\nDate: {meta['date']}"
        if meta.get('department'):
            s += f"\nDepartment: {meta['department']}"
        if meta.get('source_url'):
            s += f"\nSource: {meta['source_url']}"
        formatted.append(s)
    return '\n---\n'.join(formatted)

# Conversation context memory
last_answer_metadata = None
last_answer_title = None

# Main chat loop
if __name__ == '__main__':
    print("PMC Chatbot (English/Marathi). Type 'exit' to quit.")
    history = []
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        lang = detect_language(user_input)
        # Maintain last MAX_HISTORY turns
        history = history[-MAX_HISTORY:]
        # Detect if the query is a follow-up (contains pronouns or is short/ambiguous)
        def is_followup_query(q):
            pronouns = ["it", "that", "this", "one", "circular", "notice", "document", "file", "decision"]
            q_lower = q.lower()
            # If query is very short or contains a pronoun, treat as follow-up
            return (len(q.split()) <= 6 and any(p in q_lower for p in pronouns))

        # If follow-up, use last answer's metadata as context for Gemini
        if is_followup_query(user_input) and last_answer_metadata:
            # Compose prompt for Gemini using last answer's metadata
            context = json.dumps(last_answer_metadata, ensure_ascii=False, indent=2)
            prompt = f"You are a helpful assistant for Pune Municipal Corporation. Answer in {'Marathi' if lang=='mr' else 'English'}.\n\nHere is the document you are referring to:\n{context}\n\nUser follow-up question: {user_input}\n\nAnswer:"
            model = genai.GenerativeModel('gemini-2.5-pro')
            response = model.generate_content(prompt)
            answer = response.text.strip()
            print(f"Bot: {answer}")
            history.append({'user': user_input, 'bot': answer})
            continue
        # If follow-up, combine last answer's title with current query
        query_for_search = user_input
        if is_followup_query(user_input) and last_answer_title:
            query_for_search = f"{user_input.strip()} (referring to: {last_answer_title})"

        # Embed and search
        query_emb = embed_query(query_for_search)
        results = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)
        docs = results['matches'] if 'matches' in results else []
        context = format_docs(docs)
        # After Pinecone search, check for 'latest' intent
        if is_latest_query(query_for_search):
            # Hybrid: semantic + recency + topic keyword
            N = 10
            relevant_docs = docs[:N]
            def parse_date_safe(d):
                try:
                    return d.get('metadata', {}).get('date') or d.get('metadata', {}).get('display_date') or ''
                except Exception:
                    return ''
            sorted_results = sorted(
                relevant_docs,
                key=lambda r: parse_date_safe(r),
                reverse=True
            )
            # Extract topic keywords (noun phrases) after 'recent'/'latest' intent
            import re
            match = re.search(r'(?:recent|latest|newest|most recent|last|current|today(?:\'s)?|this week|fresh|updated|up-to-date|just published|just released|recently issued|recently published|recently released)\s+(.*)', query_for_search, re.IGNORECASE)
            topic_part = match.group(1) if match else query_for_search
            topic_phrases = extract_noun_phrases(topic_part)
            # Find the most recent match containing any topic phrase
            def contains_topic(meta):
                text_fields = [meta.get('title', ''), meta.get('description', ''), meta.get('text', '')]
                combined = ' '.join([t.lower() for t in text_fields if t])
                return any(tp in combined for tp in topic_phrases)
            filtered = [r for r in sorted_results if contains_topic(r['metadata'])]
            if filtered:
                top = filtered[0]['metadata']
                def clean(s):
                    return s.strip() if isinstance(s, str) else s
                title = clean(top.get('title', ''))
                date = clean(top.get('date', top.get('display_date', '')))
                pdf_url = clean(top.get('pdf_url'))
                url = clean(top.get('url'))
                if url and url.startswith('/'):
                    url = f"https://pmc.gov.in{url}"
                def humanize_response_en():
                    parts = []
                    if title:
                        parts.append(f"The latest relevant circular published by PMC is titled '{title}'.")
                    if date:
                        parts.append(f"It was released on {date}.")
                    if pdf_url:
                        parts.append(f"You can read the PDF here: {pdf_url}")
                    # Do NOT include web page or url
                    return ' '.join(parts) if parts else "Sorry, I couldn't find a suitable circular."
                def humanize_response_mr():
                    parts = []
                    if title:
                        parts.append(f"PMC द्वारे प्रकाशित केलेला संबंधित नवीनतम परिपत्रक '{title}' या शीर्षकाने आहे.")
                    if date:
                        parts.append(f"हे {date} रोजी प्रसिद्ध झाले आहे.")
                    if pdf_url:
                        parts.append(f"PDF पाहण्यासाठी येथे क्लिक करा: {pdf_url}")
                    # Do NOT include web page or url
                    return ' '.join(parts) if parts else "माफ करा, मला योग्य परिपत्रक सापडले नाही."
                if lang == 'mr':
                    answer = humanize_response_mr()
                else:
                    answer = humanize_response_en()
                print(f"Bot: {answer}")
                last_answer_metadata = top
                last_answer_title = top.get('title', '')
                continue
            else:
                # No match found for topic
                if lang == 'mr':
                    print("Bot: माफ करा, मागील विषयाशी संबंधित कोणतेही नवीनतम परिपत्रक सापडले नाही.")
                else:
                    print("Bot: Sorry, no recent circular found regarding your topic.")
                continue
        # For non-latest queries, just use the top relevant match
        if docs:
            top = docs[0]['metadata']
            def clean(s):
                return s.strip() if isinstance(s, str) else s
            title = clean(top.get('title', ''))
            date = clean(top.get('date', top.get('display_date', '')))
            pdf_url = clean(top.get('pdf_url'))
            url = clean(top.get('url'))
            if url and url.startswith('/'):
                url = f"https://pmc.gov.in{url}"
            def humanize_response_en():
                parts = []
                if title:
                    parts.append(f"The most relevant circular published by PMC is titled '{title}'.")
                if date:
                    parts.append(f"It was released on {date}.")
                if pdf_url:
                    parts.append(f"You can read the PDF here: {pdf_url}")
                # Do NOT include web page or url
                return ' '.join(parts) if parts else "Sorry, I couldn't find a suitable circular."
            def humanize_response_mr():
                parts = []
                if title:
                    parts.append(f"PMC द्वारे प्रकाशित केलेला संबंधित परिपत्रक '{title}' या शीर्षकाने आहे.")
                if date:
                    parts.append(f"हे {date} रोजी प्रसिद्ध झाले आहे.")
                if pdf_url:
                    parts.append(f"PDF पाहण्यासाठी येथे क्लिक करा: {pdf_url}")
                # Do NOT include web page or url
                return ' '.join(parts) if parts else "माफ करा, मला योग्य परिपत्रक सापडले नाही."
            if lang == 'mr':
                answer = humanize_response_mr()
            else:
                answer = humanize_response_en()
            print(f"Bot: {answer}")
            last_answer_metadata = top
            last_answer_title = top.get('title', '')
            continue
        # Compose prompt for LLM
        chat_history = '\n'.join([f"User: {h['user']}\nBot: {h['bot']}" for h in history])
        prompt = f"You are a helpful assistant for Pune Municipal Corporation. Answer in {'Marathi' if lang=='mr' else 'English'}.\n\nChat history:\n{chat_history}\n\nContext from PMC data:\n{context}\n\nUser query: {user_input}\n\nAnswer:"
        # Generate answer using correct Gemini API
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        answer = response.text.strip()
        print(f"Bot: {answer}")
        history.append({'user': user_input, 'bot': answer}) 