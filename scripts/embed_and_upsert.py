import os
import json
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
import hashlib
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'pmc-bot-index')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# File path
DATA_FILE = 'data/pmc_data_normalized.jsonl'
BATCH_SIZE = 50
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Initialize Gemini
if not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY not set in .env')
genai.configure(api_key=GEMINI_API_KEY)
embed_model = genai.embed_content

def embed_text(text):
    # Gemini embedding API
    try:
        response = embed_model(model='models/embedding-001', content=text, task_type='retrieval_document')
        return response['embedding']
    except Exception as e:
        print(f'Embedding error: {e}')
        return None

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def filter_metadata(meta):
    # Only keep primitive types or list of strings
    allowed_types = (str, int, float, bool)
    filtered = {}
    for k, v in meta.items():
        if k == 'raw':
            continue
        if isinstance(v, allowed_types):
            filtered[k] = v
        elif isinstance(v, list) and all(isinstance(i, str) for i in v):
            filtered[k] = v
    return filtered

def main():
    records = []
    with open(DATA_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            rec = json.loads(line)
            records.append(rec)
    print(f'Total records to embed: {len(records)}')
    batch = []
    for rec in tqdm(records, desc='Embedding and upserting'):
        text = rec.get('title', '')
        if rec.get('description'):
            text += '\n' + rec['description']
        # Chunk if too large
        if len(text) > 30000:
            chunks = chunk_text(text)
        else:
            chunks = [text]
        for i, chunk in enumerate(chunks):
            chunk_id = f"{rec['id']}_chunk{i+1}" if len(chunks) > 1 else rec['id']
            embedding = embed_text(chunk)
            if embedding:
                meta = rec.copy()
                meta['chunk_id'] = i+1
                meta['total_chunks'] = len(chunks)
                meta['text'] = chunk
                meta = filter_metadata(meta)
                batch.append({'id': chunk_id, 'values': embedding, 'metadata': meta})
            if len(batch) >= BATCH_SIZE:
                index.upsert(vectors=batch)
                batch = []
    if batch:
        index.upsert(vectors=batch)
    print('Embedding and upsert complete.')

if __name__ == '__main__':
    main() 