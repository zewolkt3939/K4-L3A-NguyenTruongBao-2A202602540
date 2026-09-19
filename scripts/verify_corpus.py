import csv
import re
import sys
from pathlib import Path

def check_corpus(dir_path: str):
    D = Path(dir_path)
    REQ = ['doc_id', 'title', 'source_url', 'retrieved_at', 'document_version', 'audience']
    mds = sorted(D.glob('*.md'))
    sources_csv = D / 'sources.csv'
    if not sources_csv.exists():
        print(f"Error: {sources_csv} does not exist!")
        return False
    rows = list(csv.DictReader(open(sources_csv, encoding='utf-8')))
    ids = []
    auds = {}
    all_ok = True
    for p in mds:
        content = p.read_text(encoding='utf-8')
        parts = content.split('---')
        if len(parts) < 3:
            print(f"{p.name:35} FAIL (no frontmatter)")
            all_ok = False
            continue
        fm = dict(re.findall(r'^(\w+):\s*(.+)$', parts[1], re.M))
        ids.append(fm.get('doc_id'))
        aud = fm.get('audience')
        auds[aud] = auds.get(aud, 0) + 1
        is_ok = all(k in fm for k in REQ) and fm.get('doc_id') == p.stem
        if not is_ok:
            all_ok = False
            missing = [k for k in REQ if k not in fm]
            print(f"{p.name:35} FAIL (missing {missing} or doc_id mismatch)")
        else:
            print(f"{p.name:35} OK")
    print('so file :', len(mds), '(can 5-10)')
    csv_status = 'khop' if sorted(r['doc_id'] for r in rows) == sorted(ids) else 'LECH'
    print('csv     :', csv_status)
    print('audience:', auds)
    return all_ok and csv_status == 'khop' and len(mds) >= 5 and len(auds) >= 2

if __name__ == '__main__':
    dir_path = sys.argv[1] if len(sys.argv) > 1 else 'data/khao-thi-phuc-khao'
    success = check_corpus(dir_path)
    sys.exit(0 if success else 1)
