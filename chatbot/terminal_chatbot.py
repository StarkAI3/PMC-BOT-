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
import bisect

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

# Add type-specific answer formatting

def format_answer_by_type(record, lang):
    type_ = record.get('type', 'other')
    title = record.get('title', '')
    date = record.get('date', record.get('display_date', ''))
    department = record.get('department', '')
    description = record.get('description', '') or record.get('body', '')
    address = record.get('address', '')
    pdf_url = record.get('pdf_url')
    url = record.get('url')
    external_link = record.get('external_link')
    link = record.get('link')
    image = record.get('image')
    banner_image = record.get('banner_image')
    timings = record.get('timings', '')
    key_attractions = record.get('key_attractions', '')
    entry_details = record.get('entry_details', '')
    ward_name = record.get('ward_name', '')
    crematorium_type = record.get('crematorium_type', '')
    operator_contact = record.get('operator_contact', '')
    operator_name = record.get('operator_name', '')
    other_details = record.get('other_details', '')
    field_contact_no = record.get('field_contact_no', '')
    gallery = record.get('gallery', [])
    # Helper for links
    def best_link():
        return external_link or link or pdf_url or url or ''
    # Humanized templates
    if type_ == 'award':
        if lang == 'mr':
            msg = f"तुमच्यासाठी PMC चा एक सन्मान: '{title}'"
            if date:
                msg += f"\nपुरस्काराची तारीख: {date}"
            if department:
                msg += f"\nसंबंधित विभाग: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nअधिक माहितीसाठी येथे क्लिक करा: {best_link()}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"Here's a recognition by PMC: '{title}'"
            if date:
                msg += f"\nAwarded on: {date}"
            if department:
                msg += f"\nDepartment: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nClick here for more info: {best_link()}"
            msg += "\nLet me know if you want more details!"
            return msg.strip()
    elif type_ == 'circular':
        if lang == 'mr':
            msg = f"PMC कडून एक महत्त्वाचा परिपत्रक: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if date:
                msg += f"\nदिनांक: {date}"
            if pdf_url:
                msg += f"\nPDF पाहण्यासाठी येथे क्लिक करा: {pdf_url}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"Here's an important circular from PMC: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if date:
                msg += f"\nDate: {date}"
            if pdf_url:
                msg += f"\nRead the PDF here: {pdf_url}"
            msg += "\nLet me know if you need more info!"
            return msg.strip()
    elif type_ == 'citizen_charter':
        if lang == 'mr':
            msg = f"PMC नागरिक सनद: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if address:
                msg += f"\nपत्ता: {address}"
            if pdf_url:
                msg += f"\nPDF: {pdf_url}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Citizen Charter: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if address:
                msg += f"\nAddress: {address}"
            if pdf_url:
                msg += f"\nPDF: {pdf_url}"
            msg += "\nLet me know if you want more details!"
            return msg.strip()
    elif type_ == 'committee_decision':
        if lang == 'mr':
            msg = f"PMC समितीचा निर्णय: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if date:
                msg += f"\nदिनांक: {date}"
            if pdf_url:
                msg += f"\nPDF: {pdf_url}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Committee Decision: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if date:
                msg += f"\nDate: {date}"
            if pdf_url:
                msg += f"\nPDF: {pdf_url}"
            msg += "\nLet me know if you want more details!"
            return msg.strip()
    elif type_ == 'crematorium':
        if lang == 'mr':
            msg = f"स्मशानभूमीची माहिती: '{title}'"
            if crematorium_type:
                msg += f"\nप्रकार: {crematorium_type}"
            if address:
                msg += f"\nपत्ता: {address}"
            if operator_contact or operator_name:
                msg += f"\nसंपर्क: {operator_contact} {operator_name}"
            if other_details:
                msg += f"\n{other_details}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"Crematorium info: '{title}'"
            if crematorium_type:
                msg += f"\nType: {crematorium_type}"
            if address:
                msg += f"\nAddress: {address}"
            if operator_contact or operator_name:
                msg += f"\nContact: {operator_contact} {operator_name}"
            if other_details:
                msg += f"\n{other_details}"
            msg += "\nLet me know if you need more info!"
            return msg.strip()
    elif type_ == 'department':
        if lang == 'mr':
            msg = f"PMC विभाग: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nऑनलाइन सेवा/लिंक: {best_link()}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Department: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nOnline service/link: {best_link()}"
            msg += "\nLet me know if you want more details!"
            return msg.strip()
    elif type_ == 'event':
        if lang == 'mr':
            msg = f"PMC कार्यक्रम: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nअधिक माहिती: {best_link()}"
            msg += "\nअधिक कार्यक्रमांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Event: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nMore info: {best_link()}"
            msg += "\nAsk me for more events or details!"
            return msg.strip()
    elif type_ == 'faq':
        if lang == 'mr':
            msg = f"PMC वारंवार विचारले जाणारे प्रश्न: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nअधिक माहिती: {best_link()}"
            msg += "\nइतर प्रश्नांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC FAQ: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nMore info: {best_link()}"
            msg += "\nAsk me if you have more questions!"
            return msg.strip()
    elif type_ == 'fire_brigade':
        if lang == 'mr':
            msg = f"PMC अग्निशमन केंद्र: '{title}'"
            if address:
                msg += f"\nपत्ता: {address}"
            if field_contact_no:
                msg += f"\nसंपर्क: {field_contact_no}"
            if image:
                msg += f"\nप्रतिमा: {image}"
            msg += "\nअग्निशमन सेवेसाठी आणखी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Fire Brigade Station: '{title}'"
            if address:
                msg += f"\nAddress: {address}"
            if field_contact_no:
                msg += f"\nContact: {field_contact_no}"
            if image:
                msg += f"\nImage: {image}"
            msg += "\nAsk me for more fire stations or info!"
            return msg.strip()
    elif type_ == 'garden':
        if lang == 'mr':
            msg = f"पुण्यातील एक सुंदर बाग: '{title}'"
            if address:
                msg += f"\nपत्ता: {address}"
            if description:
                msg += f"\n{description}"
            if key_attractions:
                msg += f"\nमुख्य आकर्षण: {key_attractions}"
            if timings:
                msg += f"\nवेळा: {timings}"
            if entry_details:
                msg += f"\nप्रवेश माहिती: {entry_details}"
            msg += "\nअधिक बागांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"Here's a beautiful garden in Pune: '{title}'"
            if address:
                msg += f"\nAddress: {address}"
            if description:
                msg += f"\n{description}"
            if key_attractions:
                msg += f"\nKey Attractions: {key_attractions}"
            if timings:
                msg += f"\nTimings: {timings}"
            if entry_details:
                msg += f"\nEntry Details: {entry_details}"
            msg += "\nAsk me for more garden suggestions!"
            return msg.strip()
    elif type_ == 'hoarding':
        if lang == 'mr':
            msg = f"PMC अधिकृत होर्डिंग: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if external_link:
                msg += f"\nलिंक: {external_link}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Authorised Hoarding: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if external_link:
                msg += f"\nLink: {external_link}"
            msg += "\nLet me know if you want more details!"
            return msg.strip()
    elif type_ == 'hospital':
        if lang == 'mr':
            msg = f"PMC रुग्णालय: '{title}'"
            if address:
                msg += f"\nपत्ता: {address}"
            if department:
                msg += f"\nविभाग: {department}"
            msg += "\nअधिक रुग्णालयांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Hospital: '{title}'"
            if address:
                msg += f"\nAddress: {address}"
            if department:
                msg += f"\nDepartment: {department}"
            msg += "\nAsk me for more hospitals or info!"
            return msg.strip()
    elif type_ == 'news':
        if lang == 'mr':
            msg = f"PMC कडून एक ताज्या बातमी: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nअधिक माहिती: {best_link()}"
            msg += "\nअधिक बातम्यांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"Here's a recent news update from PMC: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nMore info: {best_link()}"
            msg += "\nAsk me for more news or details!"
            return msg.strip()
    elif type_ == 'policy':
        if lang == 'mr':
            msg = f"PMC धोरण: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if pdf_url:
                msg += f"\nPDF: {pdf_url}"
            if description:
                msg += f"\n{description}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Policy: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if pdf_url:
                msg += f"\nPDF: {pdf_url}"
            if description:
                msg += f"\n{description}"
            msg += "\nLet me know if you want more details!"
            return msg.strip()
    elif type_ == 'project':
        if lang == 'mr':
            msg = f"PMC प्रकल्प: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nअधिक माहिती: {best_link()}"
            msg += "\nअधिक प्रकल्पांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Project: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nMore info: {best_link()}"
            msg += "\nAsk me for more projects or details!"
            return msg.strip()
    elif type_ == 'scheme':
        if lang == 'mr':
            msg = f"PMC योजना: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nअधिक माहिती: {best_link()}"
            msg += "\nअधिक योजनांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Scheme: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nMore info: {best_link()}"
            msg += "\nAsk me for more schemes or details!"
            return msg.strip()
    elif type_ == 'school':
        if lang == 'mr':
            msg = f"PMC शाळा: '{title}'"
            if address:
                msg += f"\nपत्ता: {address}"
            if department:
                msg += f"\nविभाग: {department}"
            msg += "\nअधिक शाळांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC School: '{title}'"
            if address:
                msg += f"\nAddress: {address}"
            if department:
                msg += f"\nDepartment: {department}"
            msg += "\nAsk me for more schools or info!"
            return msg.strip()
    elif type_ == 'service':
        if lang == 'mr':
            msg = f"PMC सेवा/ऑनलाइन सेवा: '{title}'"
            if department:
                msg += f"\nविभाग: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nथेट लिंक: {best_link()}"
            msg += "\nइतर सेवांसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Service/Online Service: '{title}'"
            if department:
                msg += f"\nDepartment: {department}"
            if description:
                msg += f"\n{description}"
            if best_link():
                msg += f"\nDirect link: {best_link()}"
            msg += "\nAsk me for more services or help!"
            return msg.strip()
    else:
        if lang == 'mr':
            msg = f"PMC माहिती: '{title}'"
            if description:
                msg += f"\n{description}"
            msg += "\nअधिक माहितीसाठी विचारू शकता!"
            return msg.strip()
        else:
            msg = f"PMC Information: '{title}'"
            if description:
                msg += f"\n{description}"
            msg += "\nLet me know if you want more details!"
            return msg.strip()

# Helper: Load all normalized records into a dict by id (for fast lookup)
def load_normalized_records():
    records = {}
    with open('data/pmc_data_normalized.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            try:
                rec = json.loads(line)
                if 'id' in rec:
                    records[rec['id']] = rec
            except Exception:
                continue
    return records

normalized_records = load_normalized_records()

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
            top_id = top.get('id')
            full_record = normalized_records.get(top_id, top)
            answer = format_answer_by_type(full_record, lang)
            print(f"Bot: {answer}")
            last_answer_metadata = full_record
            last_answer_title = full_record.get('title', '')
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