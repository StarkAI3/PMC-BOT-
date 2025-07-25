from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import List, Optional, Dict
from collections import deque
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from .terminal_chatbot_v2 import (
    detect_language, is_followup_query, is_latest_query, embed_query, index, TOP_K, CONTEXT_RESULTS, format_pinecone_results, build_llm_prompt, remove_duplicate_links
)
from dotenv import load_dotenv
load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
openai.api_key = OPENAI_API_KEY

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
    try:
        print("[INFO] Received /chat request")
        print(f"[INFO] Request data: {request}")
        user_input = request.user_input.strip()
        history = deque(request.history or [], maxlen=2)
        prev_user_query = history[-1].user if history else None
        lang = detect_language(user_input)
        query_for_search = user_input
        if is_followup_query(user_input, prev_user_query):
            query_for_search = f"{user_input.strip()} (context: {prev_user_query})"
        query_emb = embed_query(query_for_search)
        results = index.query(vector=query_emb, top_k=TOP_K, include_metadata=True)
        docs = results.get('matches', [])
        if is_latest_query(user_input):
            def parse_date_safe(meta):
                d = meta.get('metadata', {})
                return d.get('date') or d.get('display_date') or ''
            docs = sorted(docs, key=parse_date_safe, reverse=True)
        context_docs = docs[:CONTEXT_RESULTS]
        pinecone_context = format_pinecone_results(context_docs) if context_docs else "No relevant information found."
        chat_history = list(history)
        prompt = build_llm_prompt(user_input, pinecone_context, chat_history, lang)
        print(f"[INFO] Prompt sent to OpenAI:\n{prompt}")
        # Generate answer using OpenAI GPT-4o-mini
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0.2
        )
        answer = response.choices[0].message.content.strip()
        print(f"[INFO] OpenAI answer: {answer}")
        answer = remove_duplicate_links(answer)
        return ChatResponse(answer=answer)
    except Exception as e:
        import traceback
        print("[ERROR] Exception in /chat endpoint:")
        traceback.print_exc()
        return ChatResponse(answer=f"[Server error: {str(e)}]") 