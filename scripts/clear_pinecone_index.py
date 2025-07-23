import os
from dotenv import load_dotenv
from pinecone import Pinecone

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')
PINECONE_INDEX = os.getenv('PINECONE_INDEX', 'pmc-bot-index')

if not PINECONE_API_KEY or not PINECONE_INDEX:
    raise ValueError('Please set PINECONE_API_KEY and PINECONE_INDEX in your .env file')

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)
index = pc.Index(PINECONE_INDEX)

# Delete all vectors
print(f"Deleting all vectors from Pinecone index: {PINECONE_INDEX} ...")
index.delete(delete_all=True)
print("All vectors deleted successfully.") 