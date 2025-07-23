import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_ENV = os.getenv('PINECONE_ENV')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'pmc-bot-index')

# Gemini embedding dimension (as of 2024):
EMBED_DIM = 768

if not PINECONE_API_KEY or not PINECONE_ENV:
    raise ValueError('Please set PINECONE_API_KEY and PINECONE_ENV in your .env file')

# Initialize Pinecone client
pc = Pinecone(api_key=PINECONE_API_KEY)

# Check if index exists
if PINECONE_INDEX in [idx.name for idx in pc.list_indexes()]:
    print(f"Index '{PINECONE_INDEX}' already exists.")
else:
    print(f"Creating index '{PINECONE_INDEX}'...")
    pc.create_index(
        name=PINECONE_INDEX,
        dimension=EMBED_DIM,
        metric='cosine',
        spec=ServerlessSpec(cloud='aws', region='us-east-1')  # Change region if needed
    )
    print(f"Index '{PINECONE_INDEX}' created.") 