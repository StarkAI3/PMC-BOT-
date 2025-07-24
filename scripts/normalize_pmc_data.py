import os
import json
import hashlib
from tqdm import tqdm

# Add this mapping at the top of the file
RAW_TYPE_TO_MAIN_TYPE = {
    # Circulars, Notices, Orders
    'circular': 'circular', 'Circular': 'circular',
    # Committee Decisions, Meeting Agendas
    'committee_decision': 'committee_decision', 'Committee Decisions': 'committee_decision',
    # Citizen Charters, Service Charters
    'citizen_charter': 'citizen_charter', 'Citizen Charter': 'citizen_charter',
    # Awards, Recognitions
    'award': 'award', 'Awards': 'award', 'Awards & Recognition': 'award',
    # Gardens, Parks
    'garden': 'garden', 'Gardens In Pune': 'garden', 'Garden List': 'garden', 'Garden Rules': 'garden', 'Garden Department Sub': 'garden',
    # Departments, Contacts, Officers
    'department': 'department', 'Department': 'department', 'Contact Numbers of Officials from the Electrical Department': 'department', 'PMC Officers Directory': 'department',
    # Projects
    'project': 'project', 'Project Documents': 'project', 'Project Glimpses': 'project', 'Projects': 'project',
    # Events
    'event': 'event', 'Events': 'event', 'Competitions': 'event', 'Exhibitions': 'event',
    # News
    'news': 'news', 'News & Updates': 'news', 'Press Note': 'news',
    # Hospitals
    'hospital': 'hospital', 'Hospital': 'hospital', 'Blood Bank': 'hospital', 'Eye Bank': 'hospital',
    # Schools
    'school': 'school', 'School List': 'school', 'Model Schools': 'school', 'Primary and Technical Education School List': 'school',
    # Crematoriums
    'crematorium': 'crematorium', 'Crematoriums': 'crematorium', 'List of cemeteries': 'crematorium',
    # Fire Brigade
    'fire_brigade': 'fire_brigade', 'Fire Brigade Stations': 'fire_brigade', 'Fire Safety': 'fire_brigade',
    # Hoardings
    'hoarding': 'hoarding', 'Hoarding Rules': 'hoarding', 'Authorised Hoardings': 'hoarding',
    # Policies, Guidelines, Acts, Regulations
    'policy': 'policy', 'Policies': 'policy', 'Guidelines': 'policy', 'Acts and Regulations': 'policy',
    # Schemes, Welfare, Subsidies
    'scheme': 'scheme', 'Schemes': 'scheme', 'Welfare': 'scheme',
    # Ward Offices
    'ward_office': 'ward_office', 'Ward Offices': 'ward_office',
    # FAQ, Help, Grievance
    'faq': 'faq', 'FAQ': 'faq', 'Frequently Asked Questions': 'faq', 'Grievance': 'faq',
    # Online Services, Applications
    'service': 'service', 'Online Services': 'service', 'E-Services': 'service', 'Applications': 'service',
}

# Helper to map raw type/title/section to main type

def map_to_main_type(record, source_url=None):
    # Try to use explicit record_type/content_type/content_type_name
    for key in ['record_type', 'content_type', 'content_type_name', 'type']:
        val = record.get(key)
        if val and val in RAW_TYPE_TO_MAIN_TYPE:
            return RAW_TYPE_TO_MAIN_TYPE[val]
    # Try to use title/section
    title = record.get('title', '')
    for raw, main in RAW_TYPE_TO_MAIN_TYPE.items():
        if raw.lower() in title.lower():
            return main
    # Try to use source_url
    if source_url:
        for raw, main in RAW_TYPE_TO_MAIN_TYPE.items():
            if raw.lower() in source_url.lower():
                return main
    # Fallback
    return 'other'

def get_id(fields):
    """Generate a unique hash for a record based on key fields."""
    concat = '|'.join([str(fields.get(k, '')) for k in ['title', 'date', 'department', 'pdf_url', 'source_url']])
    return hashlib.sha256(concat.encode('utf-8')).hexdigest()

def clean_dict(d):
    return {k: v for k, v in d.items() if v not in [None, '', [], {}]}

def extract_from_node(node, source_url, lang):
    # Try to extract key fields for a single item
    title = node.get('title') or node.get('name') or node.get('subject')
    # Only include if at least a title/name/subject is present
    if not title:
        return None
    # Extract as many fields as possible
    record = {
        'title': title,
        'date': node.get('date') or node.get('display_date') or node.get('publish_date') or node.get('changed'),
        'department': node.get('department'),
        'ward_name': node.get('ward_name'),
        'crematorium_type': node.get('crematorium_type'),
        'address': node.get('address'),
        'image': node.get('image'),
        'alt': node.get('alt'),
        'pdf_url': node.get('file') or node.get('pdf_url'),
        'external_link': node.get('external_link'),
        'link': node.get('link'),
        'url': node.get('url'),
        'description': node.get('description') or node.get('desc') or node.get('short_description'),
        'operator_contact': node.get('operator_contact'),
        'operator_name': node.get('operator_name'),
        'tag': node.get('tag'),
        'status': node.get('status'),
        'other_details': node.get('other_details'),
        'source_url': source_url,
        'lang': lang,
        'raw': node  # Retain the full node for fallback
    }
    # Add type assignment using mapping
    main_type = map_to_main_type(node, source_url)
    record['record_type'] = main_type
    record['type'] = main_type  # Alias for clarity
    record = clean_dict(record)
    record['id'] = get_id(record)
    return record

def process_file(input_path, lang):
    output = []
    with open(input_path, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc=f'Processing {input_path}'):
            try:
                entry = json.loads(line)
                source_url = entry.get('source_url') if isinstance(entry, dict) else None
                nodes = []
                # If the entry is a list, treat each item as a node
                if isinstance(entry, list):
                    nodes = entry
                elif 'raw' in entry:
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
                    elif isinstance(raw, list):
                        nodes = raw
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