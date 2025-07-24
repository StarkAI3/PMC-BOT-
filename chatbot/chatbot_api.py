from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
from collections import deque
from fastapi.middleware.cors import CORSMiddleware

from .terminal_chatbot_v2 import (
    detect_language, is_followup_query, is_latest_query, embed_query, index, TOP_K, CONTEXT_RESULTS, format_pinecone_results, build_llm_prompt, remove_duplicate_links
)
import google.generativeai as genai

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Or restrict to ["http://localhost:3000"] if needed
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatHistoryItem(BaseModel):
    user: str
    bot: str

class ChatRequest(BaseModel):
    user_input: str
    history: Optional[List[ChatHistoryItem]] = None

class ChatResponse(BaseModel):
    answer: str

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    user_input = request.user_input.strip()
    history = deque(request.history or [], maxlen=2)
    prev_user_query = history[-1].user if history else None
    # Language detection
    lang = detect_language(user_input)
    # Follow-up handling
    query_for_search = user_input
    if is_followup_query(user_input, prev_user_query):
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
    chat_history = list(history)
    prompt = build_llm_prompt(user_input, pinecone_context, chat_history, lang)
    model = genai.GenerativeModel('gemini-2.5-pro')
    response = model.generate_content(prompt)
    answer = response.text.strip()
    answer = remove_duplicate_links(answer)
    return ChatResponse(answer=answer) 