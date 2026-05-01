import json, re
from pathlib import Path
from collections import Counter

records = {r['chunk_id']: r for r in [json.loads(l) for l in Path('data/index_tagged_chunks/records.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]}
queries = [json.loads(l) for l in Path('data/qa/test_qa_tagged.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]
misses = [json.loads(l) for l in Path('outputs/rag_sem_retrieval_eval/qrel_misses.jsonl').read_text(encoding='utf-8').splitlines() if l.strip()]

def normalize(t):
    return re.sub(r'\s+', ' ', str(t or '').lower().strip())

def token_recall(evidence, text):
    ev_tokens = set(normalize(evidence).split())
    tx_tokens = set(normalize(text).split())
    if not ev_tokens:
        return 0.0
    return len(ev_tokens & tx_tokens) / len(ev_tokens)

img_queries = [q for q in queries if q.get('type') == 'image']
print(f'Image queries: {len(img_queries)}')
print(f'Misses total: {len(misses)}')

# Đếm loại query bị miss
miss_ids = {m['query_id'] for m in misses}
miss_types = []
for q in queries:
    qid = str(q.get('query_id', ''))
    if qid in miss_ids:
        miss_types.append(q.get('type', '?'))
print(f'Miss types: {Counter(miss_types)}')
print()

# Thử match thủ công query image đầu tiên
q = img_queries[0]
print(f'Query: {q.get("question", "")[:100]}')
print(f'Evidence: {q.get("evidence", "")[:200]}')
print(f'Page: {q.get("page")}, source_pdf: {q.get("source_pdf")}')
page = q.get('page')
src = q.get('source_pdf')

same_pdf = [(cid, r) for cid, r in records.items() if (r.get('metadata') or r).get('source_pdf') == src]
same_page = [(cid, r) for cid, r in same_pdf if (r.get('metadata') or r).get('page') == page]
print(f'Chunks same PDF: {len(same_pdf)}, same page: {len(same_page)}')

for cid, r in same_page[:5]:
    mod = (r.get('metadata') or r).get('modality')
    txt = str(r.get('text', ''))
    sc = token_recall(q.get('evidence', ''), txt)
    print(f'  [{mod}] token_recall={sc:.2f} | {txt[:180]}')

print()
# Xem top token_recall trên toàn bộ records (không lọc page)
scored = []
for cid, r in records.items():
    meta = r.get('metadata') or r
    if meta.get('source_pdf') != src:
        continue
    txt = str(r.get('text', ''))
    sc = token_recall(q.get('evidence', ''), txt)
    if sc > 0:
        scored.append((sc, meta.get('modality'), meta.get('page'), txt[:200]))
scored.sort(reverse=True)
print(f'Top matches (same PDF, any page):')
for sc, mod, pg, txt in scored[:5]:
    print(f'  [{mod}] page={pg} token_recall={sc:.2f} | {txt[:180]}')
