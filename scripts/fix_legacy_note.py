#!/usr/bin/env python3
import json

NB = '/Users/dongyuangao/Desktop/advanced-genai-26/Step_2_Reliability_Aware_Design.ipynb'
with open(NB) as f:
    nb = json.load(f)

# Find the cell with principle 1 and update it
for cell in nb['cells']:
    if cell['cell_type'] == 'markdown' and 'Reuse, don' in ''.join(cell['source']):
        for i, line in enumerate(cell['source']):
            if 'Reuse, don' in line:
                cell['source'][i] = (
                    '1. **Reuse legacy code, do not rewrite**: The retrieval pipeline '
                    'from the baseline reproduction lives in `legacy_retrieval_engine.py`. '
                    'It is loaded as an internal library (via `%run`) in Step 3. '
                    'It is **not** a deliverable in the current Step 1-4 workflow '
                    '-- it is background infrastructure we build on top of.\n'
                )
                break
        break

with open(NB, 'w') as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print('Updated principle 1 to clarify legacy status.')
