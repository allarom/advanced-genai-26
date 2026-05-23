import json, sys
nb = json.load(open(sys.argv[1]))
start = int(sys.argv[2])
end = int(sys.argv[3])
for i, c in enumerate(nb['cells']):
    if i < start or i > end: continue
    if c['cell_type'] != 'markdown': continue
    src = ''.join(c['source'])
    print(f'=== CELL {i} [{c.get("id","?")}] ===')
    print(src[:5000])
    print()
