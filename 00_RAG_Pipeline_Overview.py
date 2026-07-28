# Databricks notebook source
# DBTITLE 1,RAG Pipeline Overview
# MAGIC %md
# MAGIC # 🚀 RAG Pipeline - Databricks Implementation
# MAGIC
# MAGIC ## Overview
# MAGIC
# MAGIC This RAG (Retrieval-Augmented Generation) pipeline enables semantic search and question answering over your Orion documentation using:
# MAGIC - **Databricks Foundation Models** for embeddings (BGE) and LLM (Llama 3.3)
# MAGIC - **FAISS** for fast vector similarity search
# MAGIC - **Delta Lake** for data persistence
# MAGIC - **Unity Catalog Volumes** for index storage
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📂 Pipeline Architecture
# MAGIC
# MAGIC The pipeline is split into **two notebooks** for efficient development and execution:
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────────────────────────────────────────────────┐
# MAGIC │                    RAG Pipeline                              │
# MAGIC ├─────────────────────────────────────────────────────────────┤
# MAGIC │                                                              │
# MAGIC │  📓 01_RAG_Build_Embeddings_Index                           │
# MAGIC │  ├─ Load document chunks from Delta                         │
# MAGIC │  ├─ Generate embeddings (Databricks BGE)                    │
# MAGIC │  ├─ Save embeddings to Delta table                          │
# MAGIC │  ├─ Build FAISS index                                       │
# MAGIC │  └─ Persist index to Volume                                 │
# MAGIC │                                                              │
# MAGIC │                        ⬇️                                     │
# MAGIC │                                                              │
# MAGIC │  📓 02_RAG_Query_Retrieval                                  │
# MAGIC │  ├─ Load FAISS index from Volume                            │
# MAGIC │  ├─ Semantic search with FAISS                              │
# MAGIC │  ├─ Retrieve top-k relevant chunks                          │
# MAGIC │  ├─ Augment prompt with context                             │
# MAGIC │  └─ Generate answer with LLM                                │
# MAGIC │                                                              │
# MAGIC └─────────────────────────────────────────────────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔄 Execution Workflow
# MAGIC
# MAGIC ### **Workflow A: Build Index (Run Once)**
# MAGIC
# MAGIC **When to run:**
# MAGIC - ✅ First time setup
# MAGIC - ✅ New documents added
# MAGIC - ✅ Documents updated
# MAGIC - ❌ NOT every query session
# MAGIC
# MAGIC **Notebook:** `01_RAG_Build_Embeddings_Index`
# MAGIC
# MAGIC **What it does:**
# MAGIC 1. Loads chunks from `rag_on_databricks.landing.docs_chunked`
# MAGIC 2. Generates 1024-dim embeddings using `databricks-bge-large-en`
# MAGIC 3. Saves embeddings to `rag_on_databricks.landing.docs_with_embeddings`
# MAGIC 4. Builds FAISS IVFFlat index with cosine similarity
# MAGIC 5. Persists to `/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/`
# MAGIC
# MAGIC **Duration:** ~5-10 minutes (depends on document count)
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ### **Workflow B: Query Documents (Run Frequently)**
# MAGIC
# MAGIC **When to run:**
# MAGIC - ✅ Every time you want to query documents
# MAGIC - ✅ Daily usage
# MAGIC - ✅ Production queries
# MAGIC
# MAGIC **Notebook:** `02_RAG_Query_Retrieval`
# MAGIC
# MAGIC **What it does:**
# MAGIC 1. Loads saved FAISS index from Volume (~5 seconds)
# MAGIC 2. Embeds user query
# MAGIC 3. Retrieves top-k relevant chunks via FAISS search
# MAGIC 4. Creates augmented prompt with context
# MAGIC 5. Generates answer using Llama 3.3 70B
# MAGIC
# MAGIC **Duration:** ~5-10 seconds per query
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📊 Data Flow
# MAGIC
# MAGIC ```
# MAGIC ┌─────────────────┐
# MAGIC │  Source Docs    │ (Volume: Orion_Docs/)
# MAGIC └────────┬────────┘
# MAGIC          │
# MAGIC          ├─ ai_parse_document (Cell 6)
# MAGIC          │
# MAGIC          ▼
# MAGIC ┌─────────────────┐
# MAGIC │  Parsed Docs    │ (Delta: docs_parsed)
# MAGIC └────────┬────────┘
# MAGIC          │
# MAGIC          ├─ LLM Cleaning (Cell 12)
# MAGIC          │
# MAGIC          ▼
# MAGIC ┌─────────────────┐
# MAGIC │  Clean Markdown │ (transformed_df)
# MAGIC └────────┬────────┘
# MAGIC          │
# MAGIC          ├─ Chunking (Cell 13-14)
# MAGIC          │
# MAGIC          ▼
# MAGIC ┌─────────────────┐
# MAGIC │  Chunks         │ (Delta: docs_chunked)
# MAGIC └────────┬────────┘
# MAGIC          │
# MAGIC          ├─ 📓 Notebook 01: Build
# MAGIC          │
# MAGIC          ▼
# MAGIC ┌─────────────────┐
# MAGIC │  Embeddings     │ (Delta: docs_with_embeddings)
# MAGIC └────────┬────────┘
# MAGIC          │
# MAGIC          ├─ FAISS Build
# MAGIC          │
# MAGIC          ▼
# MAGIC ┌─────────────────┐
# MAGIC │  FAISS Index    │ (Volume: faiss_index/)
# MAGIC └────────┬────────┘
# MAGIC          │
# MAGIC          ├─ 📓 Notebook 02: Query
# MAGIC          │
# MAGIC          ▼
# MAGIC ┌─────────────────┐
# MAGIC │  RAG Answers    │
# MAGIC └─────────────────┘
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎯 Quick Start
# MAGIC
# MAGIC ### **First Time Setup:**
# MAGIC
# MAGIC ```python
# MAGIC # Step 1: Run the build notebook
# MAGIC %run ./01_RAG_Build_Embeddings_Index
# MAGIC
# MAGIC # Wait for completion (~5-10 min)
# MAGIC # ✅ Output: FAISS index saved to Volume
# MAGIC ```
# MAGIC
# MAGIC ### **Daily Usage:**
# MAGIC
# MAGIC ```python
# MAGIC # Step 2: Run the query notebook
# MAGIC %run ./02_RAG_Query_Retrieval
# MAGIC
# MAGIC # Change the query in Cell 10 or 11
# MAGIC user_query = "What are the safety features of Orion A1?"
# MAGIC result = RAG(user_query, k=4)
# MAGIC print(result['answer'])
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📦 Dependencies
# MAGIC
# MAGIC **Python Packages:**
# MAGIC - `langchain-text-splitters`
# MAGIC - `databricks_langchain`
# MAGIC - `faiss-cpu`
# MAGIC - `pandas`
# MAGIC - `numpy`
# MAGIC
# MAGIC **Databricks Resources:**
# MAGIC - Catalog: `rag_on_databricks`
# MAGIC - Schema: `landing`
# MAGIC - Volume: `/Volumes/rag_on_databricks/landing/vol_landing/`
# MAGIC
# MAGIC **Foundation Models:**
# MAGIC - Embeddings: `databricks-bge-large-en` (1024 dimensions)
# MAGIC - LLM: `databricks-meta-llama-3-3-70b-instruct`
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🔧 Configuration
# MAGIC
# MAGIC **Key Parameters:**
# MAGIC
# MAGIC | Parameter | Value | Description |
# MAGIC |-----------|-------|-------------|
# MAGIC | `batch_size` | 15 | Chunks per embedding batch |
# MAGIC | `k` | 4 | Number of retrieved chunks |
# MAGIC | `nlist` | auto | FAISS clusters (embeddings // 50) |
# MAGIC | `temperature` | 0.1 | LLM creativity (lower = more factual) |
# MAGIC | `max_tokens` | 500 | LLM response length |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📈 Performance
# MAGIC
# MAGIC **Build (Notebook 01):**
# MAGIC - Embedding: ~2-4 chunks/second
# MAGIC - Index build: <10 seconds
# MAGIC - Total: ~5-10 minutes for ~40 chunks
# MAGIC
# MAGIC **Query (Notebook 02):**
# MAGIC - Index load: ~5 seconds
# MAGIC - Retrieval: <1 second
# MAGIC - LLM generation: ~3-5 seconds
# MAGIC - Total: ~5-10 seconds per query
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## ✅ Benefits of This Architecture
# MAGIC
# MAGIC 1. **Separation of Concerns:**
# MAGIC    - Build index once, query many times
# MAGIC    - No unnecessary rebuilding
# MAGIC
# MAGIC 2. **Persistence:**
# MAGIC    - FAISS index saved to Volume
# MAGIC    - Survives cluster restarts
# MAGIC    - Reusable across sessions
# MAGIC
# MAGIC 3. **Efficiency:**
# MAGIC    - Fast query startup (~5s)
# MAGIC    - No re-embedding on every query
# MAGIC
# MAGIC 4. **Scalability:**
# MAGIC    - Delta tables for data management
# MAGIC    - FAISS IVF for fast search
# MAGIC    - Batch processing for embeddings
# MAGIC
# MAGIC 5. **Production Ready:**
# MAGIC    - Clear build/query separation
# MAGIC    - Easy to schedule (Workflow A weekly, Workflow B on-demand)
# MAGIC    - Monitoring-friendly
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 🎓 Next Steps
# MAGIC
# MAGIC 1. **Run Setup:**
# MAGIC    - Execute `01_RAG_Build_Embeddings_Index` once
# MAGIC    - Verify index saved to Volume
# MAGIC
# MAGIC 2. **Test Queries:**
# MAGIC    - Run `02_RAG_Query_Retrieval`
# MAGIC    - Try different questions
# MAGIC
# MAGIC 3. **Tune Parameters:**
# MAGIC    - Adjust `k` for retrieval count
# MAGIC    - Modify `temperature` for creativity
# MAGIC    - Change `batch_size` for speed
# MAGIC
# MAGIC 4. **Scale Up:**
# MAGIC    - Add more documents to Volume
# MAGIC    - Re-run build notebook
# MAGIC    - Query scales automatically!
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## 📚 Resources
# MAGIC
# MAGIC - **FAISS Documentation:** https://github.com/facebookresearch/faiss/wiki
# MAGIC - **Databricks Foundation Models:** https://docs.databricks.com/en/machine-learning/foundation-models/
# MAGIC - **Unity Catalog Volumes:** https://docs.databricks.com/en/sql/language-manual/sql-ref-volumes.html
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Ready to start?** → Open `01_RAG_Build_Embeddings_Index` and run all cells! 🚀

# COMMAND ----------

