import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langdetect import detect
from tqdm import tqdm

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
        # Embed and search
        query_emb = embed_query(user_input)
        results = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)
        docs = results['matches'] if 'matches' in results else []
        context = format_docs(docs)
        # Compose prompt for LLM
        chat_history = '\n'.join([f"User: {h['user']}\nBot: {h['bot']}" for h in history])
        prompt = f"You are a helpful assistant for Pune Municipal Corporation. Answer in {'Marathi' if lang=='mr' else 'English'}.\n\nChat history:\n{chat_history}\n\nContext from PMC data:\n{context}\n\nUser query: {user_input}\n\nAnswer:"
        # Generate answer using correct Gemini API
        model = genai.GenerativeModel('gemini-2.5-pro')
        response = model.generate_content(prompt)
        answer = response.text.strip()
        print(f"Bot: {answer}")
        history.append({'user': user_input, 'bot': answer}) 