import os
import json
import hashlib
from tqdm import tqdm

def get_id(fields):
    """Generate a unique hash for a record based on key fields."""
    concat = '|'.join([str(fields.get(k, '')) for k in ['title', 'date', 'department', 'pdf_url', 'source_url']])
    return hashlib.sha256(concat.encode('utf-8')).hexdigest()

def clean_dict(d):
    return {k: v for k, v in d.items() if v not in [None, '', [], {}]}

def extract_from_node(node, source_url, lang):
    # Try to extract key fields for a single item
    title = node.get('title') or node.get('name') or node.get('subject')
    date = node.get('date') or node.get('display_date')
    department = node.get('department')
    pdf_url = node.get('file') or node.get('pdf_url')
    # Some APIs may use 'file' as a list or dict
    if isinstance(pdf_url, list):
        pdf_url = pdf_url[0] if pdf_url else None
    if isinstance(pdf_url, dict):
        pdf_url = pdf_url.get('url')
    url = node.get('url') or node.get('link')
    # Only output if all key fields are present
    if title and date and pdf_url:
        record = {
            'title': title,
            'date': date,
            'department': department,
            'pdf_url': pdf_url,
            'url': url,
            'source_url': source_url,
            'lang': lang
        }
        record = clean_dict(record)
        record['id'] = get_id(record)
        return record
    return None

def process_file(input_path, lang):
    output = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'Processing {input_path}'):
            try:
                entry = json.loads(line)
                source_url = entry.get('source_url')
                # Try to find nodes in raw.data.nodes or similar
                nodes = []
                if 'raw' in entry:
                    raw = entry['raw']
                    if isinstance(raw, dict):
                        if 'data' in raw and 'nodes' in raw['data']:
                            nodes = raw['data']['nodes']
                        elif 'budgets' in raw:
                            nodes = raw['budgets']
                        elif 'election_level1' in raw:
                            nodes = raw['election_level1']
                        else:
                            nodes = [raw]
                    else:
                        nodes = [raw]
                else:
                    nodes = [entry]
                for node in nodes:
                    record = extract_from_node(node, source_url, lang)
                    if record:
                        output.append(record)
            except Exception as e:
                print(f'Error processing line: {e}')
    return output

def main():
    data_dir = 'data'
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl') and not f.endswith('normalized.jsonl')]
    all_records = []
    for fname in files:
        lang = 'en' if 'en' in fname else 'mr' if 'mr' in fname else 'unknown'
        records = process_file(os.path.join(data_dir, fname), lang)
        for rec in records:
            all_records.append(rec)
    # Write normalized output
    out_path = os.path.join(data_dir, 'pmc_data_normalized.jsonl')
    with open(out_path, 'w', encoding='utf-8') as out:
        for rec in all_records:
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'Wrote {len(all_records)} normalized records to {out_path}')

if __name__ == '__main__':
    main() 