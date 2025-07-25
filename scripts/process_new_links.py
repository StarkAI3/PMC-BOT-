import os
import json
import hashlib
import requests
from tqdm import tqdm
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Config
ENG_LINKS = 'eng_links'
MR_LINKS = 'mr_links'
PROCESSED_IDS_FILE = 'data/processed_ids.txt'
NORMALIZED_FILE = 'data/pmc_data_normalized.jsonl'
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200
BATCH_SIZE = 50
PROCESSED_LINKS_FILE = 'data/processed_links.txt'

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'pmc-bot-index')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Initialize Gemini
if not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY not set in .env')
genai.configure(api_key=GEMINI_API_KEY)
embed_model = genai.embed_content

def embed_text(text):
    try:
        response = embed_model(model='models/embedding-001', content=text, task_type='retrieval_document')
        return response['embedding']
    except Exception as e:
        print(f'Embedding error: {e}')
        return None

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    return splitter.split_text(text)

def load_processed_ids():
    if not os.path.exists(PROCESSED_IDS_FILE):
        return set()
    with open(PROCESSED_IDS_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def save_processed_ids(ids):
    with open(PROCESSED_IDS_FILE, 'a', encoding='utf-8') as f:
        for id_ in ids:
            f.write(id_ + '\n')

def load_processed_links():
    if not os.path.exists(PROCESSED_LINKS_FILE):
        return set()
    with open(PROCESSED_LINKS_FILE, 'r', encoding='utf-8') as f:
        return set(line.strip() for line in f if line.strip())

def save_processed_links(links):
    with open(PROCESSED_LINKS_FILE, 'a', encoding='utf-8') as f:
        for link in links:
            f.write(link + '\n')

def extract_data_from_link(url, lang):
    try:
        resp = requests.get(url, timeout=15, verify=False)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
            return [item for item in data['data']]
        else:
            return [data]
    except Exception as e:
        print(f'Error fetching {url}: {e}')
        return []

def normalize_record(record, source_url, lang):
    # Minimal normalization for demo; expand as needed
    title = record.get('title') or record.get('name') or record.get('subject')
    if not title:
        return None
    norm = {
        'title': title,
        'date': record.get('date') or record.get('display_date') or record.get('publish_date') or record.get('changed'),
        'department': record.get('department'),
        'address': record.get('address'),
        'image': record.get('image'),
        'alt': record.get('alt'),
        'url': record.get('url'),
        'source_url': source_url,
        'lang': lang,
        'raw': record,
    }
    # Generate unique ID
    text = json.dumps(record, ensure_ascii=False)
    uid = hashlib.sha256((text + lang + source_url).encode('utf-8')).hexdigest()
    norm['id'] = uid
    return norm

def process_links(links_file, lang, processed_ids, new_norm_records, processed_links, new_links_processed):
    with open(links_file, 'r', encoding='utf-8') as f:
        links = [line.strip() for line in f if line.strip()]
    for url in tqdm(links, desc=f'Processing {lang} links'):
        if url in processed_links:
            continue
        records = extract_data_from_link(url, lang)
        if records:
            new_links_processed.append(url)
        for rec in records:
            norm = normalize_record(rec, url, lang)
            if not norm:
                continue
            if norm['id'] in processed_ids:
                continue
            new_norm_records.append(norm)
            processed_ids.add(norm['id'])

def embed_and_upsert(norm_records):
    for rec in tqdm(norm_records, desc='Embedding and upserting'):
        text = rec.get('title', '')
        if rec.get('description'):
            text += '\n' + rec['description']
        if len(text) > 30000:
            chunks = chunk_text(text)
        else:
            chunks = [text]
        for i, chunk in enumerate(chunks):
            chunk_id = f"{rec['id']}_chunk{i}"
            embedding = embed_text(chunk)
            if embedding:
                meta = {k: v for k, v in rec.items() if k != 'raw' and v is not None}
                meta['chunk_index'] = i
                index.upsert(vectors=[{'id': chunk_id, 'values': embedding, 'metadata': meta}])

def main():
    processed_ids = load_processed_ids()
    processed_links = load_processed_links()
    new_norm_records = []
    new_links_processed = []
    # Process English links
    if os.path.exists(ENG_LINKS):
        process_links(ENG_LINKS, 'en', processed_ids, new_norm_records, processed_links, new_links_processed)
    # Process Marathi links
    if os.path.exists(MR_LINKS):
        process_links(MR_LINKS, 'mr', processed_ids, new_norm_records, processed_links, new_links_processed)
    if not new_norm_records:
        print('No new records to process.')
    else:
        # Append new normalized records
        with open(NORMALIZED_FILE, 'a', encoding='utf-8') as f:
            for rec in new_norm_records:
                f.write(json.dumps(rec, ensure_ascii=False) + '\n')
        # Embed and upsert
        embed_and_upsert(new_norm_records)
        # Save processed IDs
        save_processed_ids([rec['id'] for rec in new_norm_records])
        print(f'Processed {len(new_norm_records)} new records.')
    # Save processed links
    if new_links_processed:
        save_processed_links(new_links_processed)
        print(f'Added {len(new_links_processed)} new links to processed_links.txt')

if __name__ == '__main__':
    main() 