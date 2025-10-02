# ENHANCING GRAPH-BASED RETRIEVAL-AUGMENTED GENERATION VIA QUERY-AWARE PATH REASONING
This repository contains the official implementation of the paper:
"Enhancing Graph-based Retrieval-Augmented Generation via Query-Aware Path Reasoning"

Our method introduces query-aware path reasoning into the graph-based retrieval-augmented generation (RAG) framework, aiming to improve knowledge selection and reasoning by leveraging structured relational paths.

---
## Installation
```bash
cd QPathRAG
pip install -e .
```
---

## Quick Start
The end-to-end pipeline is organized into multiple steps under the reproduce/ folder.

### Step 0: Preprocessing
- Convert your raw .jsonl dataset into a .json file, where each entry contains a list of document segments.

### Step 1: Knowledge Graph Construction
- Utilize the model’s capability to extract structured knowledge from text.
- Transform the segmented documents produced in Step 0 into a knowledge graph representation.

### Step 2: Question Generation
- Generate summaries from groups of document segments.
- Concatenate them as context and prompt a large language model to create 125 diverse questions.

### Step 3: Knowledge-Guided Question Answering
- Use the knowledge graph obtained in Step 1.
- Let the model answer the 125 questions generated in Step 2 based on structured reasoning over the graph.
- Save the results for evaluation.

### Evaluation
- Run
```bash
python reproduce/batch_eval.py
```
to evaluate the model’s responses.

- Finally, compute the win rate with:
```bash
python reproduce/calc_win_rate.py
```

---

## Usage Notes
- Before running, make sure to set your API key if using LLMs:
```bash
api_key = "sk-..."
```
- Prepare your own dataset in .jsonl format and follow the pipeline from Step 0 → Step 3 → Evaluation.