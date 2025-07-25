import os
import sys
import json
from dotenv import load_dotenv
from pinecone import Pinecone
import google.generativeai as genai

# Config
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'pmc-bot-index')

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')

if not PINECONE_API_KEY:
    raise ValueError('PINECONE_API_KEY not set in .env')
if not GEMINI_API_KEY:
    raise ValueError('GEMINI_API_KEY not set in .env')

genai.configure(api_key=GEMINI_API_KEY)
embed_model = genai.embed_content

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

def embed_text(text):
    try:
        response = embed_model(model='models/embedding-001', content=text, task_type='retrieval_query')
        return response['embedding']
    except Exception as e:
        print(f'Embedding error: {e}')
        return None

def main():
    if len(sys.argv) < 2:
        print('Usage: python query_pinecone.py "your query here"')
        sys.exit(1)
    query = ' '.join(sys.argv[1:])
    embedding = embed_text(query)
    if not embedding:
        print('Failed to generate embedding for the query.')
        sys.exit(1)
    # Query Pinecone
    try:
        result = index.query(vector=embedding, top_k=5, include_metadata=True)
        print(f'\nTop {len(result["matches"])} results for query: "{query}"\n')
        for i, match in enumerate(result['matches'], 1):
            print(f'Result {i}:')
            print(f'  ID: {match["id"]}')
            print(f'  Score: {match["score"]:.4f}')
            print(f'  Metadata:')
            print(json.dumps(match.get('metadata', {}), ensure_ascii=False, indent=4))
            print('-' * 40)
    except Exception as e:
        print(f'Error querying Pinecone: {e}')
        sys.exit(1)

if __name__ == '__main__':
    main() 