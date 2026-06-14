import json, sys
nb = json.load(open(sys.argv[1]))
start = int(sys.argv[2]) if len(sys.argv) > 2 else 0
for i, c in enumerate(nb['cells']):
    if i < start:
        continue
    src = ''.join(c['source'])
    print(f"--- cell {i} [{c['cell_type']}] id={c.get('id','?')} ---")
    print(src[:1200])
    print()
