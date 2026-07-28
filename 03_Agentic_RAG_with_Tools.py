# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
# DBTITLE 1,Install Dependencies
!pip install -U typing-extensions>=4.13.0
!pip install -U databricks_langchain
!pip install -U langgraph
!pip install faiss-cpu

# COMMAND ----------

# DBTITLE 1,Restart Python
dbutils.library.restartPython()

# COMMAND ----------

#AgentBricks

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from databricks_langchain import DatabricksEmbeddings, ChatDatabricks
from langchain.agents import create_agent
from langchain_core.tools import tool
import faiss
import pickle
import numpy as np
import mlflow


# COMMAND ----------

# DBTITLE 1,Configure Paths and Load Index
# Configuration
catalog = "rag_on_databricks"
schema = "landing"
index_save_path = "/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/"

print("Loading FAISS index and metadata...")

# Load FAISS index
index = faiss.read_index(index_save_path + "index.faiss")

# Load chunks metadata
with open(index_save_path + "chunks_data.pkl", "rb") as f:
    chunks_data = pickle.load(f)

print(f"Loaded FAISS index with {index.ntotal} vectors")
print(f"Loaded metadata for {len(chunks_data)} chunks")

# Load embedding model
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

# DBTITLE 1,Define Tool Functions
@tool
def search_documents(query: str, k: int = 3, threshold: float = 0.65) -> str:
    """
    Search for relevant document chunks based on a query.
    Use this tool when you need to find information from the document corpus to answer user questions.
    
    Args:
        query: The search query or question to find relevant documents for
        k: Number of top relevant chunks to retrieve (default: 3)
        threshold: Minimum similarity score to consider a result relevant (default: 0.65)
    
    Returns:
        A formatted string with relevant document chunks and their similarity scores.
        Returns a message if no relevant documents are found above the threshold.
    """
    # Embed and normalize query
    query_vector = np.array(embedding_model.embed_query(query), dtype='float32').reshape(1, -1)
    
    # Normalize for cosine similarity
    query_norm = np.linalg.norm(query_vector, axis=1, keepdims=True)
    query_norm[query_norm == 0] = 1e-12
    normalized_query = query_vector / query_norm
    
    # Search FAISS index
    scores, indices = index.search(normalized_query, k)
    
    # Filter results by threshold
    results = []
    for i, idx in enumerate(indices[0]):
        score = float(scores[0][i])
        
        # Skip results below threshold
        if score < threshold:
            continue
            
        chunk_text = chunks_data[idx]['chunk']
        chunk_path = chunks_data[idx]['path'].split('/')[-1]
        results.append(f"\n[Source {i+1}] (Score: {score:.4f}) - {chunk_path}\n{chunk_text}")
    
    # Return appropriate message based on results
    if not results:
        return f"No relevant documents found above similarity threshold ({threshold}). The query may be outside the scope of the available documents."
    
    return "\n".join(results)


# COMMAND ----------

# DBTITLE 1,Initialize Agent LLM and Tools
# Create list of tools for the agent (only search tool needed)
tools = [search_documents]

# Initialize the LLM for the agent
agent_llm = ChatDatabricks(
    endpoint="databricks-meta-llama-3-3-70b-instruct",
    temperature=0.1
)

print("Agent LLM initialized")
print(f"Tools available: {[tool.name for tool in tools]}")

# COMMAND ----------

# DBTITLE 1,Create Agent with System Prompt
# System prompt for the agent
system_prompt = """You are an AI assistant with document search access.

Workflow:
1. Use search_documents to find relevant information
2. Generate answers based ONLY on retrieved documents
3. Search multiple times if needed
4. If no relevant documents are found (tool returns threshold message), respond: "I don't have information about this topic in my document corpus."
5. NEVER use your pre-trained knowledge - only use information from documents"""

# Create the agent using LangChain's create_agent
agent = create_agent(
    model=agent_llm,
    tools=tools,
    system_prompt=system_prompt
)

print("Agent created using create_agent")

# COMMAND ----------

# DBTITLE 1,Enable MLflow Tracing
# Enable MLflow tracing to see agent's reasoning and tool calls
mlflow.langchain.autolog()
print("MLflow tracing enabled")

# COMMAND ----------

# DBTITLE 1,Test Agent - Example Query
# Test the agent
test_query = "How does the Orion system prevent overheating?"
# "How does the Orion system prevent overheating?"

print(f"User: {test_query}")
print("="*80)

# Run agent
result = agent.invoke({"messages": [{"role": "user", "content": test_query}]})
answer = result["messages"][-1].content

print("\n" + "="*80)
print("Answer:\n")
print(answer)
print("="*80)

# COMMAND ----------

# DBTITLE 1,Interactive Query - Ask Your Own Questions
# Ask your own questions
custom_query = "What are the safety features of Orion A1?"

print(f"User: {custom_query}")
print("="*80)

result = agent.invoke({"messages": [{"role": "user", "content": custom_query}]})
answer = result["messages"][-1].content

print("\n" + "="*80)
print("Answer:\n")
print(answer)
print("="*80)