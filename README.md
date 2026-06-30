# RAGnRoll 🎵

**A local-first music retrieval and ranking system that maps natural-language mood/activity prompts to audio-feature constraints, retrieves songs from a 1.1M-track corpus, and ranks them using semantic similarity and mood fit.**

🔗 **[Live Demo](https://ragnroll-cool-url-1234.streamlit.app/)**

🎥 **[Video Demo](https://drive.google.com/file/d/1Tbd5m7RU0ExTcc_OQv7vNjVjBlx1EniF/view?usp=sharing)**

Note: Takes 2-3 minutes on first start. Runs seamlessly after that for every new prompt due to improved caching and data management

---

## What This Is

RAGnRoll is a **RAG pipeline for music** : the LLM only parses intent. All retrieval, filtering, ranking, and diversity logic is custom-built and runs on a local corpus of 1.14 million Spotify tracks (2000–2023).

---

## Pipeline

```
User prompt
    ↓
parse_intent()          →  structured intent: moods, energy, valence,
                           acousticness, genres, playlist_length
    ↓
rebuild_retrieval_query() →  audio-feature label string + mood tags + genres
    ↓
FAISS IndexFlatIP search  →  top-k semantically similar tracks
    ↓
filter_songs()            →  hard filter on [energy + valence] (±20% margin)
    ↓
rank_songs()              →  mood_fit_score (0.8) + norm_similarity (0.2)
    ↓
deduplicate_by_name()     →  collapse remix/version duplicates
    ↓
apply_mmr()               →  audio-feature diversity reranking
    ↓
explainer()               →  single LLM call: playlist-level explanation
    ↓
evaluate_playlist()       →  mood fit, artist/genre diversity, feature averages
    ↓
Streamlit UI
```

---

## Key Engineering Decisions

**Retrieval query is not the raw prompt.**
The raw prompt is never passed to FAISS. It is first parsed into structured intent, then converted to an audio-feature label string (`"low energy high acousticness melancholic cinematic"`). This prevents surface token matching (e.g., "exploring" retrieving tracks named "Explorer").

**Hard filters on energy + valence only.**
`acousticness`, `instrumentalness`, `speechiness`, and `danceability` are bimodal or heavily skewed near 0. Hard filtering on them cascades to near-zero results. They contribute as soft signals in `mood_fit_score` instead.

**Ranking is not cosine similarity.**
`final_score = 0.8 × mood_fit_score + 0.2 × normalized_similarity`. Pure cosine similarity rewards lexical overlap with the retrieval query string, not true semantic fit. Mood fit directly measures distance of each audio feature from the LLM-inferred target range midpoint.

**LLM only does intent parsing.**
`parse_intent()` runs once per query via Groq (Llama 3.1 8B Instruct). It outputs structured JSON: moods, activities, feature ranges, genre tags, playlist length. All downstream logic is deterministic Python.

**Genre tags improve retrieval without hard filtering.**
The parser outputs genre tags from a fixed controlled vocabulary. These are appended to the FAISS query string as soft semantic signals — not used as hard filters, because dataset genre labels are inconsistent.

**Memory-efficient artifact loading.**
`song_text` column (~330MB) dropped from deployed metadata. Production parquet (`track_metadata_slim.parquet`) retains only 12 columns. FAISS index loaded with `mmap` (`IO_FLAG_MMAP | IO_FLAG_READ_ONLY`) to avoid RAM spikes on Streamlit Cloud. Fallback to full RAM load if mmap unavailable.

**Artifacts hosted on Hugging Face Hub.**
FAISS index and metadata parquet are stored in a private HF dataset repo. Downloaded once via `hf_hub_download` and cached in `tempfile.gettempdir()`. No re-download on repeated queries.

**Explainer is one LLM call per playlist.**
Per-song LLM calls hit Groq rate limits on the free tier. A single call receives the full ranked playlist with feature values and returns a 2–3 sentence playlist-level explanation grounded in actual audio feature data.

---

## Stack

| Layer | Tool |
|---|---|
| Embeddings | `BAAI/bge-small-en-v1.5` (384-dim) via SentenceTransformers |
| Vector index | FAISS `IndexFlatIP` + L2 normalization (1,144,452 vectors) |
| Intent parsing | Groq API — Llama 3.1 8B Instruct via LangChain |
| Artifact hosting | Hugging Face Hub dataset repo |
| Frontend | Streamlit Cloud |
| Diversity | MMR (λ=0.7, 6 audio features) + name deduplication |
| Evaluation | Custom offline metrics — mood fit, diversity, feature averages |

---

## Repo Structure

```
RAGnRoll/
├── app/
│   └── streamlit_app.py       # UI, session state, spinners
├── src/
│   ├── config_loader.py       # loads harcoded paths from config.yaml
│   ├── prepare_data.py        # loads csv data from dataset, does preliminary cleaning
│   ├── indexing.py            # creates the faiss.index embeddings, metadata.parquet & metadata_slim.parquet
│   ├── parser.py              # parse_intent() — Groq + LangChain
│   ├── retriever.py           # retrieve(), filter_songs(), rebuild_retrieval_query()
│   ├── ranker.py              # rank_songs(), mood_fit_score()
│   ├── diversity_mmr.py       # deduplicate_by_name(), apply_mmr()
│   ├── explainer.py           # playlist-level LLM explanation
│   ├── evaluation.py          # evaluate_playlist()
│   └── logger.py              # get_logger(__name__)
├── configs/
│   ├── config.yaml
├── requirements.txt
└── setup.py
```

---

## Running Locally

```bash
git clone https://github.com/RigAgravanshi/RAGnRoll
cd RAGnRoll
pip install -e .
```

Create `.env`:
```
GROQ_API_KEY = groq_key
HF_KEY = hf_token
```

```bash
streamlit run app/streamlit_app.py
```

Artifacts download automatically from HF Hub on first run and are cached locally.

---

## Dataset

1,144,452 Spotify tracks (2000–2023). Audio features normalized to [0, 1] per Spotify's scale. Embeddings generated offline from `song_text` field using `BAAI/bge-small-en-v1.5`. `song_text` dropped from production metadata to reduce memory footprint.

---

## What This Is Not

- Not a Spotify recommendation API wrapper
- Not a collaborative filter or matrix factorization model
- Not a fine-tuned model
- No user personalization (V2 scope)
- No Spotify export (V2 scope)

---

Self-Note: Initially created and run on my personal hardware (Nvidia RTX 4050) and local LLM (Ollama: qwen 2.5 7b)-->(Came with its own probs: GPU clashes between 2 LLM calls and SentenceTransformers)

Due to web deployment and other constraints (huge amount of data with limited processing and caching on streamlit hosting) shifted to other models. This may have reduced perf. and significantly increased time taken to complete the project & I came face to face with problems encountered during deployment.

Various functions for data caching, metadata slimming, fallbacks defined with the help of Cursor IDE.
