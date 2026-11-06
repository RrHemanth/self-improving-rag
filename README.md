# Self-Improving RAG

A local, document-grounded question-answering system that combines dense and keyword retrieval, query rewriting, cross-encoder reranking, LLM-based evaluation, and experience memory. Instead of using a fixed retrieval strategy for every question, the agent records past outcomes and uses similar queries to decide which strategy to try first.

The project includes a Streamlit interface for uploading documents, asking questions, viewing source citations, and inspecting retrieval diagnostics.

## How it works

```text
PDF / TXT / MD / DOCX files
          |
    Document loader
          |
  Context-aware chunking
          |
   +------+-------+
   |              |
Dense embeddings   BM25 index
   |              |
 FAISS         Keyword search
   +------+-------+
          |
  Retrieval strategies <------ Query rewriting
          |
 Cross-encoder reranking
          |
  Context + LLM answer
          |
 LLM-based evaluation
          |
 Optional regeneration
          |
 Answer + source citations
          |
 Experience memory ------> Preferred strategy for future queries
```

### Retrieval strategies

The agent compares four strategies:

| Strategy | Retrieval input | Retriever |
| --- | --- | --- |
| `hybrid_rewritten` | Rewritten question | FAISS + BM25, fused with reciprocal rank fusion |
| `hybrid_original` | Original question | FAISS + BM25, fused with reciprocal rank fusion |
| `dense_rewritten` | Rewritten question | FAISS |
| `bm25_rewritten` | Rewritten question | BM25 |

Retrieved candidates are reranked with a cross-encoder before answer generation.

### Feedback and memory

1. **Explore:** When there is no sufficiently similar past query, evaluate all four strategies.
2. **Evaluate:** Score retrieval relevance, answer relevance, and faithfulness. If an answer is weak, regenerate it once and retain the new answer only if its combined relevance and faithfulness improve.
3. **Remember:** Save the query, strategy attempts, scores, and best-performing strategy or tied strategies in a corpus-specific JSON file.
4. **Exploit:** For a similar future query, try the memory-preferred strategy first and stop early if it meets the answer-quality thresholds. If it fails, try the remaining strategies.

The system **does not fine-tune model weights**. Its improvement comes from reusing feedback to choose retrieval strategies more efficiently.

The overall attempt score is:

```text
0.4 × retrieval relevance + 0.3 × answer relevance + 0.3 × faithfulness
```

An attempt is considered successful when answer relevance is at least **0.75** and faithfulness is at least **0.80**. These scores come from an LLM evaluator, so they are estimates rather than independent human judgments.

## Tech stack

| Component | Implementation |
| --- | --- |
| Embeddings | Sentence Transformers (`all-MiniLM-L6-v2`) |
| Dense retrieval | FAISS, normalized embeddings and inner-product search |
| Sparse retrieval | BM25 (`rank-bm25`) |
| Hybrid retrieval | Reciprocal rank fusion (RRF) |
| Reranking | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| Query rewriting and answer generation | Ollama `qwen2.5:3b` |
| Retrieval and answer evaluation | Ollama `qwen2.5:7b` |
| LLM integration | LangChain Core and LangChain Ollama |
| Interface | Streamlit |
| Experience storage | Local JSON files, separated by document corpus |

## Getting started

### Prerequisites

- Python 3 and `pip`
- [Ollama](https://ollama.com/) installed and running locally
- Enough disk space and memory to run the Ollama and Hugging Face models

From the repository root, create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
```

Download the Ollama models:

```bash
ollama pull qwen2.5:3b
ollama pull qwen2.5:7b
```

Make sure the Ollama service is running, then start the app:

```bash
python3 -m streamlit run app.py
```

Open the local address printed by Streamlit (usually `http://localhost:8501`). On first use, Sentence Transformers may download the embedding and reranking models from Hugging Face.

### Using the app

1. Upload one or more `.pdf`, `.txt`, `.md`, or `.docx` documents.
2. Select **Process documents** to build the retrieval indexes.
3. Ask a question in the chat interface.
4. Read the generated answer and its source citations.
5. Expand **Retrieval diagnostics** to inspect the selected strategy, attempted strategies, retrieval relevance, answer relevance, faithfulness, and overall scores.

Experience memory is stored in `data/memory/app/<corpus_id>_<config_hash>.json`. The corpus ID is derived from the uploaded file names and contents, so a different document collection gets separate memory.

## Evaluation

### 64-question ablation study

The system was evaluated on 64 held-out questions, with 40 prior experiences available for memory-guided strategy selection.

| Metric | Dense baseline | Explore all | Self-improving |
| --- | ---: | ---: | ---: |
| MRR | 0.646 | 0.751 | **0.752** |
| Answer relevance | 0.797 | 0.875 | **0.875** |
| Faithfulness | 0.758 | 0.977 | **0.977** |
| Average latency | 13.62s | 93.47s | **65.77s** |
| Strategy attempts | 1.00 | 4.00 | **3.11** |

**Key results:**

- **16.4% higher MRR** compared with dense retrieval.
- **22.3% fewer strategy attempts** compared with exhaustive exploration.
- **29.6% lower average latency** compared with exhaustive exploration.
- Maintained the same measured answer relevance and faithfulness as exhaustive exploration.

Original benchmark artifacts are stored in `data/evaluation/`.

### Experience-memory demonstrations

Controlled experiments showed that the agent could reuse successful retrieval strategies for repeated and semantically similar questions.

| Metric | Full exploration | Memory-guided |
| --- | ---: | ---: |
| Strategy attempts | 4 | **1** |
| Evaluation score | 1.0 | **1.0** |
| Repeated-query latency | 21.27s | **4.75s** |
| Paraphrased-query latency | 20.65s | **5.04s** |

This demonstrated a **75% reduction in strategy attempts**, with measured latency reductions of **77.7%** and **75.6%**, respectively. The paraphrased query matched experience memory with a cosine similarity of **0.861**.

### Large-document retrieval

The pipeline was also tested on a 411-page Python programming PDF. It successfully retrieved relevant context and answered a question about the steps required for a terminating while loop.

The successful attempt achieved an overall evaluation score of **0.867**, with answer relevance and faithfulness both scoring **1.0**.

These measurements describe specific local evaluations rather than guaranteed improvements on arbitrary queries. Answer-quality scores were assigned by an LLM evaluator.

### Reproducing the benchmark

Run `python3 benchmarks/benchmark_rag.py` from the repository root. The script compares exhaustive exploration and memory-guided execution and saves results to `benchmarks/results.json`.

## Testing

Run the automated unit tests:

```bash
python3 -m unittest discover -s tests -v
```

The current suite contains **19 passing tests**, covering:

- Experience-memory persistence and embedding caching
- Exploration, exploitation, forced exploration, and tied strategies
- Answer-quality thresholds, regeneration, and failure diagnosis
- Routing to dense, BM25, and hybrid retrieval strategies

Manual end-to-end checks have also covered single-document questions, memory reuse across runs, multi-document retrieval, corpus-specific memory, and the Streamlit interface with citations and diagnostics.

## Repository layout

```text
app.py                         # Streamlit interface
requirements.txt               # Python dependencies
src/
  app/pipeline.py              # Build document indexes and the RAG agent
  ingestion/                   # Document loading and chunking
  retrieval/                   # FAISS, BM25, embeddings, hybrid retrieval
  reranking/reranker.py        # Cross-encoder reranking
  agent/                       # Query rewriting and agent control flow
  generation/generator.py      # Answer generation and citations
  evaluation/rag_evaluator.py  # Retrieval and answer evaluation
  memory/experience_memory.py  # Persistent experience and strategy selection
tests/                         # Automated unit tests
data/evaluation/               # Benchmark results
data/memory/app/                # Corpus-specific runtime memory
```

Some additional modules and root-level test scripts are retained from earlier experiments; the Streamlit application uses the pipeline assembled in `src/app/pipeline.py`.

## Limitations

- **Local inference can be slow.** A new query may explore four strategies and make multiple LLM evaluation calls.
- **Evaluator scores are imperfect.** An LLM judge may overestimate answer quality or faithfulness.
- **Memory is strategy-level adaptation, not model training.** It can help with similar queries but does not guarantee better results for unseen topics.
- **Evaluation scope is limited.** The reported benchmark uses a Python-programming document corpus and should not be treated as a cross-domain result.
- **Deployment is local.** The current application depends on a running Ollama service and has not been validated as a hosted production service.
- **Dependency versions are not pinned** in the current `requirements.txt`; reproducibility across future package releases may require a lockfile or pinned versions.

## License

No license has been specified yet. Add a `LICENSE` file before describing the repository as open source under a particular license.
