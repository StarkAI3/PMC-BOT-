import os
import re
import json
import requests
from tqdm import tqdm
import urllib3
import hashlib

# Suppress only the single InsecureRequestWarning from urllib3 needed.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Input files
ENG_LINKS = 'eng_links'
MR_LINKS = 'mr_links'
# Output files
OUT_ENG = 'data/pmc_data_en.jsonl'
OUT_MR = 'data/pmc_data_mr.jsonl'

# Regex patterns for extraction
PDF_PATTERN = re.compile(r'https?://[^\s]+\.pdf')
PHONE_PATTERN = re.compile(r'\b(\+91[-\s]?)?[0]?[6789]\d{9}\b')
MAP_PATTERN = re.compile(r'https?://(goo\.gl|maps\.google\.com|www\.google\.com/maps)[^\s]*')

# Recursive cleaner for null/empty values
def clean_obj(obj):
    if isinstance(obj, dict):
        return {k: clean_obj(v) for k, v in obj.items() if v not in [None, '', [], {}] and clean_obj(v) not in [None, '', [], {}]}
    elif isinstance(obj, list):
        return [clean_obj(v) for v in obj if v not in [None, '', [], {}] and clean_obj(v) not in [None, '', [], {}]]
    else:
        return obj

# Helper to extract fields from JSON
def extract_fields(obj):
    cleaned = clean_obj(obj)
    text = json.dumps(cleaned, ensure_ascii=False)
    pdfs = list(set(PDF_PATTERN.findall(text)))
    phones = list(set(PHONE_PATTERN.findall(text)))
    maps = list(set(MAP_PATTERN.findall(text)))
    # You can add more extraction logic here (external links, dates, etc.)
    return {
        'pdf_links': pdfs,
        'phone_numbers': phones,
        'map_links': maps,
        'raw': cleaned
    }

def get_existing_ids(output_file):
    ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    try:
                        item = json.loads(line)
                        text = json.dumps(item['raw'], ensure_ascii=False)
                        source_url = item.get('source_url', '')
                        uid = hashlib.sha256((text + item.get('lang', '') + source_url).encode('utf-8')).hexdigest()
                        ids.add(uid)
                    except Exception:
                        continue
    return ids

def process_links(input_file, output_file, lang):
    existing_ids = get_existing_ids(output_file)
    with open(input_file, 'r', encoding='utf-8') as f:
        links = [line.strip() for line in f if line.strip()]
    success_log = f'success_links_{lang}.txt'
    fail_log = f'failed_links_{lang}.txt'
    successes = []
    failures = []
    with open(output_file, 'a', encoding='utf-8') as out, \
         open(success_log, 'w', encoding='utf-8') as success_f, \
         open(fail_log, 'w', encoding='utf-8') as fail_f:
        for url in tqdm(links, desc=f'Processing {lang} links'):
            attempt = 0
            success = False
            while attempt < 3 and not success:
                attempt += 1
                try:
                    resp = requests.get(url, timeout=15, verify=False)
                    resp.raise_for_status()
                    data = resp.json()
                    # Some APIs return lists, some dicts
                    if isinstance(data, dict) and 'data' in data and isinstance(data['data'], list):
                        wrote = False
                        for item in data['data']:
                            fields = extract_fields(item)
                            fields['source_url'] = url
                            fields['lang'] = lang
                            text = json.dumps(fields['raw'], ensure_ascii=False)
                            uid = hashlib.sha256((text + lang + url).encode('utf-8')).hexdigest()
                            if uid in existing_ids:
                                continue
                            existing_ids.add(uid)
                            out.write(json.dumps(fields, ensure_ascii=False) + '\n')
                            wrote = True
                        if wrote:
                            success = True
                    else:
                        fields = extract_fields(data)
                        fields['source_url'] = url
                        fields['lang'] = lang
                        text = json.dumps(fields['raw'], ensure_ascii=False)
                        uid = hashlib.sha256((text + lang + url).encode('utf-8')).hexdigest()
                        if uid not in existing_ids:
                            existing_ids.add(uid)
                            out.write(json.dumps(fields, ensure_ascii=False) + '\n')
                        success = True
                except Exception as e:
                    if attempt == 3:
                        print(f"Error processing {url} after 3 attempts: {e}")
            if success:
                successes.append(url)
                success_f.write(url + '\n')
            else:
                failures.append(url)
                fail_f.write(url + '\n')
    print(f"\nSummary for {lang} links:")
    print(f"  Total links: {len(links)}")
    print(f"  Successes: {len(successes)} (see {success_log})")
    print(f"  Failures: {len(failures)} (see {fail_log})")

def main():
    process_links(ENG_LINKS, OUT_ENG, 'en')
    process_links(MR_LINKS, OUT_MR, 'mr')

if __name__ == '__main__':
    main() 