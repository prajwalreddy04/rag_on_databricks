# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,RAG Pipeline - Part 2: Query & Retrieval
# MAGIC %md
# MAGIC # RAG Pipeline - Part 2: Query & Retrieval
# MAGIC
# MAGIC **Purpose:** Load pre-built FAISS index and perform semantic search + answer generation.
# MAGIC
# MAGIC **Prerequisites:**
# MAGIC - ✅ Notebook `01_RAG_Build_Embeddings_Index` must be run first
# MAGIC - ✅ FAISS index must exist at `/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/`
# MAGIC
# MAGIC **Run this notebook:**
# MAGIC - ✅ Every time you want to query documents
# MAGIC - ✅ Fast startup (loads saved index, no rebuild)
# MAGIC
# MAGIC **Features:**
# MAGIC - Load FAISS index from Volume (5 seconds)
# MAGIC - Semantic search with cosine similarity
# MAGIC - RAG pipeline: Retrieve + Augment + Generate
# MAGIC - LLM-powered answer generation

# COMMAND ----------

!pip install -U typing-extensions>=4.13.0
!pip install -U databricks_langchain
!pip install faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from databricks_langchain import DatabricksEmbeddings, ChatDatabricks
import faiss
import pickle
import numpy as np

# COMMAND ----------

# DBTITLE 1,Configure Paths
# Configuration
catalog = "rag_on_databricks"
schema = "landing"

# FAISS index location
index_save_path = "/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/"

print(f"📂 FAISS index path: {index_save_path}")

# COMMAND ----------

# DBTITLE 1,Load Embedding Model
# Load the same embedding model used during indexing
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

# DBTITLE 1,Load FAISS Index from Volume
import time
start_load = time.time()

print("📥 Loading FAISS index from Volume...\n")

# Load FAISS index
index = faiss.read_index(index_save_path.replace("dbfs:", "") + "index.faiss")

# Load chunks metadata
with open(index_save_path.replace("dbfs:", "") + "chunks_data.pkl", "rb") as f:
    chunks_data = pickle.load(f)

load_time = time.time() - start_load

print(f"✅ Loaded FAISS index with {index.ntotal} vectors")
print(f"✅ Loaded metadata for {len(chunks_data)} chunks")
print(f"   Load time: {load_time:.2f}s")
print(f"\n📌 Ready to retrieve!")

# COMMAND ----------

# DBTITLE 1,Define Retrieval Function
def retrieve(query, k=4):
    """
    Retrieve top-k most relevant chunks for a query using FAISS.
    
    Args:
        query (str): User's question
        k (int): Number of chunks to retrieve (default: 4)
    
    Returns:
        list: Top-k chunks with similarity scores, paths, and IDs
    """
    # Embed and normalize query
    query_vector = np.array(embedding_model.embed_query(query), dtype='float32').reshape(1, -1)
    
    # Normalize for cosine similarity
    query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_norm[query_norm == 0] = 1e-12
    normalized_query = query_vector / query_norm
    
    # Search FAISS index
    scores, indices = index.search(normalized_query, k)
    
    # Return results with metadata
    results = []
    for i, idx in enumerate(indices[0]):
        results.append({
            "chunk": chunks_data[idx]['chunk'],
            "path": chunks_data[idx]['path'],
            "id": chunks_data[idx]['id'],
            "score": float(scores[0][i])
        })
    
    return results

print("✅ Retrieval function defined")

# COMMAND ----------

# DBTITLE 1,Test Retrieval
# Test the retrieval function
test_query = "How does the Orion system prevent overheating?"
test_results = retrieve(test_query, k=3)

print(f"🔍 Query: {test_query}")
print(f"\n✅ Retrieved {len(test_results)} chunks:\n")

for i, result in enumerate(test_results, 1):
    print(f"{i}. Score: {result['score']:.4f}")
    print(f"   Path: {result['path'].split('/')[-1]}")
    print(f"   Chunk preview: {result['chunk'][:150]}...\n")

# COMMAND ----------

# DBTITLE 1,Initialize LLM Model
# Initialize Databricks LLM for answer generation
model = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    max_tokens=500,
    temperature=0.1
)


# COMMAND ----------

# DBTITLE 1,Define RAG Pipeline Functions
def create_rag_prompt(retrieved_chunks, question):
    """
    Create augmented prompt with retrieved context for LLM.
    """
    context = "\n\n".join([
        f"Source {i+1} (Score: {chunk['score']:.3f})\n{chunk['chunk']}"
        for i, chunk in enumerate(retrieved_chunks)
    ])
    
    prompt = f"""You are an expert AI assistant.

Use ONLY the information provided in the context below to answer the user's question.

Rules:
- Answer only from the provided context.
- Do not make up information.
- If the answer is not available in the context, reply: "I couldn't find the answer in the provided documents."
- Keep the answer clear, concise, and professional.
- Use bullet points whenever appropriate.

Context:
{context}

Question:
{question}

Answer:"""
    
    return prompt

def RAG(query, k=4):
    """
    Complete RAG pipeline: Retrieve + Augment + Generate.
    
    Args:
        query (str): User's question
        k (int): Number of chunks to retrieve
    
    Returns:
        dict: Question, answer, and sources
    """
    # 1. Retrieve relevant chunks
    retrieved_chunks = retrieve(query, k=k)
    
    # 2. Create augmented prompt
    prompt = create_rag_prompt(retrieved_chunks, query)
    
    # 3. Generate answer with LLM
    response = model.invoke(prompt)
    answer = response.content
    
    return {
        "question": query,
        "answer": answer,
        "sources": retrieved_chunks
    }

print("✅ RAG pipeline functions defined")

# COMMAND ----------

# DBTITLE 1,Run RAG Query
# Example query
user_query = "How does the Orion system prevent overheating?"
#"What are the safety features of the Orion A1?"

print(f"💬 Query: {user_query}\n")
print("="*80)

# Run RAG pipeline
result = RAG(user_query, k=4)

print(f"\n🤖 Answer:\n")
print(result['answer'])
print("\n" + "="*80)

print(f"\n📚 Sources used ({len(result['sources'])}):\n")
for i, source in enumerate(result['sources'], 1):
    print(f"{i}. {source['path'].split('/')[-1]} (Score: {source['score']:.4f})")

# COMMAND ----------

# DBTITLE 1,Interactive Query Cell
# Change the query below to ask your own questions

custom_query = "What is orion robotics ?" 
#"How does the motion controller maintain balance?"

result = RAG(custom_query, k=4)

print(f"💬 Query: {custom_query}\n")
print("="*80)
print(f"\n🤖 Answer:\n")
print(result['answer'])
print("\n" + "="*80)