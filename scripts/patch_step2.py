#!/usr/bin/env python3
"""
Patch Step_2_Reliability_Aware_Design.ipynb with small edits.
Reads the notebook, modifies cells, and writes it back.
"""

import json
from pathlib import Path

NB = Path("/Users/dongyuangao/Desktop/advanced-genai-26/Step_2_Reliability_Aware_Design.ipynb")

with open(NB, "r", encoding="utf-8") as f:
    nb = json.load(f)

# --- Edit 1: Add "Known gap" note to §2.1 Key design principles ---
# Find the cell that contains "### Key design principles"
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "### Key design principles" in "".join(cell["source"]):
        # Find the line with "Graceful degradation"
        for i, line in enumerate(cell["source"]):
            if "Graceful degradation" in line:
                # Insert after that line
                cell["source"].insert(i + 1, "\n")
                cell["source"].insert(
                    i + 2,
                    "5. **Known gap — answer synthesis**: The current Step 3 prototype uses a simplified answer generator (`docs[0].page_content[:250]`). "
                    "The production implementation will use the full `AnswerSynthesizerAgent` from `multi-agent-step-2_strategy-A.ipynb`.\n"
                )
                break
        break

# --- Edit 2: Add §2.6 "Integration with Legacy Step 2" before the Summary ---
# Find the Summary cell and insert before it
for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "markdown" and "## 2.6 Summary" in "".join(cell["source"]):
        new_cell = {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "---\n",
                "\n",
                "## 2.6 Integration with Legacy Step 2\n",
                "\n",
                "The reliability layer does **not** replace the existing multi-agent pipeline — it **wraps** it.\n",
                "\n",
                "### Data flow\n",
                "\n",
                "```\n",
                "User Query\n",
                "    │\n",
                "    ▼\n",
                "┌────────────────────────────────────────────┐\n",
                "│  Legacy Step 2 (strategy-A)                │\n",
                "│  • QueryUnderstandingAgent                  │\n",
                "│  • ConfidenceOrchestrator / Waterfall /     │\n",
                "│    VotingOrchestrator                       │\n",
                "│  • Fusion → ReRank → AnswerSynthesizer      │\n",
                "│  → produces: docs + draft_answer + trace    │\n",
                "└────────────────────────────────────────────┘\n",
                "    │\n",
                "    ▼\n",
                "┌────────────────────────────────────────────┐\n",
                "│  Step 3 Reliability Layer                   │\n",
                "│  • Sufficiency / Groundedness / Contradict  │\n",
                "│  • Trust → Abstention / Critic / Recovery   │\n",
                "│  → produces: final_decision + unified_trace │\n",
                "└────────────────────────────────────────────┘\n",
                "```\n",
                "\n",
                "### Code reuse\n",
                "\n",
                "- **Step 3 loads strategy-A** via `%run multi-agent-step-2_strategy-A.ipynb`.\n",
                "- The orchestrators (`ConfidenceOrchestrator`, `WaterfallOrchestrator`, `VotingOrchestrator`) are reused unchanged.\n",
                "- The answer synthesizer will be reused once the placeholder in Step 3 is replaced (see §2.1, principle 5).\n",
                "- All reliability agents are **new** and live in `Step_3_Reliable_Adaptive_Agentic_RAG.ipynb`.\n",
                "\n",
                "### Why wrap instead of rewrite?\n",
                "\n",
                "- **Minimal risk**: We keep the proven retrieval pipeline intact.\n",
                "- **Modular testing**: We can evaluate strategy-A alone (baseline) and with the reliability layer (new system) side-by-side.\n",
                "- **Clear ablation**: In Step 4, `ReliableAdaptiveRAG(ablate=[...])` disables individual reliability checks while keeping the same retrieval pipeline.\n"
            ]
        }
        nb["cells"].insert(idx, new_cell)
        break

# --- Edit 3: Rename old "2.6 Summary" to "2.7 Summary" ---
for cell in nb["cells"]:
    if cell["cell_type"] == "markdown" and "## 2.6 Summary" in "".join(cell["source"]):
        for i, line in enumerate(cell["source"]):
            if line.startswith("## 2.6 Summary"):
                cell["source"][i] = "## 2.7 Summary\n"
                break
        break

# Save
with open(NB, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Notebook patched successfully.")
print("Changes:")
print("  1. Added 'Known gap — answer synthesis' note to §2.1 design principles")
print("  2. Added §2.6 'Integration with Legacy Step 2' (wrapper diagram + reuse explanation)")
print("  3. Renamed old §2.6 Summary → §2.7 Summary")
