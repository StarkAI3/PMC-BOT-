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
    # Try to extract all possible relevant fields
    title = node.get('title')
    description = node.get('description') or node.get('short_description') or node.get('body')
    date = node.get('display_date') or node.get('publish_date') or node.get('date') or node.get('changed')
    department = node.get('department')
    pdf_url = node.get('file') or node.get('pdf_url')
    # Some nodes have pdf_files as a list
    if not pdf_url and 'pdf_files' in node and node['pdf_files']:
        pdf_url = node['pdf_files'][0].get('file_url')
    # Some nodes have a list of pdf_links
    if not pdf_url and 'pdf_links' in node and node['pdf_links']:
        pdf_url = node['pdf_links'][0]
    # Other possible fields
    phone_numbers = node.get('phone_numbers')
    location = node.get('location')
    url = node.get('url')
    # Compose record
    record = {
        'id': None,  # to be filled after
        'title': title,
        'description': description,
        'date': date,
        'department': department,
        'pdf_url': pdf_url,
        'phone_numbers': phone_numbers,
        'location': location,
        'url': url,
        'source_url': source_url,
        'lang': lang
    }
    record = clean_dict(record)
    record['id'] = get_id(record)
    return record

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
                    rec = extract_from_node(node, source_url, lang)
                    if rec.get('title') and (rec.get('pdf_url') or rec.get('url')):
                        output.append(rec)
            except Exception as e:
                print(f'Error processing line: {e}')
    return output

def main():
    data_dir = 'data'
    files = [f for f in os.listdir(data_dir) if f.endswith('.jsonl') and not f.endswith('normalized.jsonl')]
    all_records = {}
    for fname in files:
        lang = 'en' if 'en' in fname else 'mr' if 'mr' in fname else 'unknown'
        records = process_file(os.path.join(data_dir, fname), lang)
        for rec in records:
            if rec['id'] not in all_records:
                all_records[rec['id']] = rec
    # Write normalized output
    out_path = os.path.join(data_dir, 'pmc_data_normalized.jsonl')
    with open(out_path, 'w', encoding='utf-8') as out:
        for rec in all_records.values():
            out.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f'Wrote {len(all_records)} normalized records to {out_path}')

if __name__ == '__main__':
    main() 