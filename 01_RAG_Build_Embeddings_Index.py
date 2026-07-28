# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "5"
# ///
!pip install -U langchain-text-splitters
!pip install -U databricks_langchain
!pip install faiss-cpu

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,Import Required Libraries
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pyspark.sql import functions as F
from databricks_langchain import DatabricksEmbeddings
import faiss
import numpy as np
import pickle
import time

# COMMAND ----------

# DBTITLE 1,Configure Catalog and Schema
# Configuration
catalog = "rag_on_databricks"
schema = "landing"

# Table names
parsed_table = f"{catalog}.{schema}.docs_parsed"
chunked_table = f"{catalog}.{schema}.docs_chunked"
embeddings_table = f"{catalog}.{schema}.docs_with_embeddings"

#PDFs/Docs Landing path
vol_landing_path="/Volumes/rag_on_databricks/landing/vol_landing/Orion_Docs/"

# FAISS index save path
index_save_path = "/Volumes/rag_on_databricks/landing/vol_landing/faiss_index/"

print(f"Catalog: {catalog}")
print(f"Schema: {schema}")
print(f"Output table: {embeddings_table}")
print(f"Index save path: {index_save_path}")

# COMMAND ----------

# DBTITLE 1,List the files in your volumes path.
spark.sql(f"LIST '{vol_landing_path}'").display()

# COMMAND ----------

# DBTITLE 1,Parsing Documents with Python
# Read all files from the documents volume
docs_df = spark.read.format("binaryFile").load(vol_landing_path)

# Parse each document using ai_parse_document (use expr to call the SQL AI function)
parsed_df = docs_df.withColumn("parsed_content", 
                           F.expr(f"""ai_parse_document(content, map(
                                "version", "2.0",
                                "imageOutputPath", "{vol_landing_path}parsed_images/"
                               ))""")
                          )
# Drop binary content
parsed_df = parsed_df.drop("content")

# Display a sample of the parsed results
display(parsed_df)


# COMMAND ----------

# DBTITLE 1,Save parsed_df to Delta table
# Save the parsed results as a Delta table for easy querying and sharing
parsed_df.write.format("delta").mode("overwrite").saveAsTable(parsed_table)

print(f"Parsed results saved to Delta table: {parsed_table}")

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from rag_on_databricks.landing.docs_parsed

# COMMAND ----------

# DBTITLE 1,LLM-Powered Semantic Cleaning with ai_query
# Choose a Databricks foundation model (or your own serving endpoint name)
ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"

# Example prompt for the LLM
prompt_prefix = '''
You are a helpful assistant. Given a JSON object representing a parsed document (with pages, elements, and metadata), convert the content into clean, readable markdown. Use "== page ==" to separate each page. Preserve important structure such as headers, tables, and captions. Do not include any JSON or code blocks in the output—just the clean markdown text.

JSON:

'''

# Apply ai_query to batch process the parsed JSON text
# Note: Claude models do not support responseFormat type "text"; omit it for plain-text output.
transformed_df = (
    parsed_df.withColumn(
        "clean_markdown_text",
        F.expr(f"""
          ai_query(
            '{ENDPOINT}',
            CONCAT('{prompt_prefix}', CAST(parsed_content AS STRING))
          )
        """)
    )
)

display(transformed_df.select("path", "clean_markdown_text"))

# COMMAND ----------

# DBTITLE 1,Split Clean Markdown into Chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set chunking parameters
CHUNK_SIZE = 2000
CHUNK_OVERLAP = 200

# Build the text splitter (similar to Cell 16)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    separators=["\n== page ==\n", "== page ==", "\n\n", "\n", " ", ""]
)

# Collect the cleaned documents (similar to Cell 17 approach)
docs_list = transformed_df.select("path", "clean_markdown_text").collect()

# Split each document into chunks
chunks = []
for doc in docs_list:
    path = doc["path"]
    text = doc["clean_markdown_text"]
    
    if text and text.strip():
        # Split the text into chunks
        chunks_subset = splitter.split_text(text)
        
        # Create chunk records with document path only
        for chunk in chunks_subset:
            if chunk.strip():
                chunks.append({
                    "path": path,
                    "chunk": chunk
                })

print(f"Total chunks created: {len(chunks)}")

# Convert back to Spark DataFrame
df_chunks = spark.createDataFrame(chunks)

# Display the resulting chunked DataFrame
display(df_chunks)

# COMMAND ----------

# DBTITLE 1,Save chunked DataFrame to Delta table
# Add a unique, incremental id column before saving
df_chunks = df_chunks.withColumn("id", F.monotonically_increasing_id())

# Save the chunked data with id to the Delta table for retrieval and embedding
df_chunks.write.format("delta").mode("overwrite").option("mergeSchema", "true").saveAsTable(chunked_table)

display(spark.read.table(chunked_table))

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from rag_on_databricks.landing.docs_chunked

# COMMAND ----------

# DBTITLE 1,Load Embedding Model
# Load Databricks BGE embedding model
embedding_model = DatabricksEmbeddings(endpoint="databricks-bge-large-en")

# COMMAND ----------

# DBTITLE 1,Load Chunks from Delta Table
# Load chunks from Delta table
df_chunks = spark.table(chunked_table).toPandas()
chunk_list = df_chunks["chunk"].tolist()

print(f"Loaded {len(chunk_list)} chunks from {chunked_table}")
print(f"Columns: {df_chunks.columns.tolist()}")

# COMMAND ----------

# DBTITLE 1,Generate Embeddings (Batch Processing)
# Generate embeddings in batches
batch_size = 15
all_embeddings = []

print(f"Processing {len(chunk_list)} chunks (batch_size={batch_size})\n")
start_time = time.time()

for i in range(0, len(chunk_list), batch_size):
    batch = chunk_list[i:i + batch_size]
    batch_embeddings = embedding_model.embed_documents(batch)
    all_embeddings.extend(batch_embeddings)
    print(f"   Processed {min(i + batch_size, len(chunk_list))}/{len(chunk_list)} chunks")
    time.sleep(2)  # Rate limit protection

embeddings = all_embeddings
elapsed = time.time() - start_time

print(f"\nGenerated {len(embeddings)} embeddings")
print(f"Duration: {elapsed:.1f}s ({elapsed/60:.1f} min)")
print(f"Dimensions: {len(embeddings[0])}")

# COMMAND ----------

# DBTITLE 1,Save Embeddings to Delta Table
# Add embeddings to dataframe
df_chunks['embedding'] = embeddings

# Convert to Spark DataFrame and save
spark_df = spark.createDataFrame(df_chunks)
spark_df.write.format("delta").mode("overwrite").saveAsTable(embeddings_table)

print(f"\nSaved {len(df_chunks)} chunks with embeddings")
print(f" Table: {embeddings_table}")
print(f" Columns: {spark_df.columns}")

display(spark.table(embeddings_table).limit(5))

# COMMAND ----------

# DBTITLE 1,Build FAISS Index
# Load embeddings from Delta table
df_embedded = spark.table(embeddings_table).toPandas()

# Prepare data for FAISS
chunks_data = df_embedded[['id', 'path', 'chunk']].to_dict('records')
embeddings_array = np.array(df_embedded['embedding'].tolist(), dtype='float32')

print(f"Loaded {len(embeddings_array)} embeddings")
print(f"Embedding dimensions: {embeddings_array.shape[1]}")

# Normalize embeddings for cosine similarity
norms = np.linalg.norm(embeddings_array, axis=1, keepdims=True)
norms[norms == 0] = 1e-12
normalized_embeddings = embeddings_array / norms

print("\nBuilding FAISS index...")

# Build FAISS IVF index
dimension = embeddings_array.shape[1]
nlist = max(1, len(embeddings_array) // 10)  # Adaptive cluster size

quantizer = faiss.IndexFlatIP(dimension)
index = faiss.IndexIVFFlat(quantizer, dimension, nlist, faiss.METRIC_INNER_PRODUCT)

# Train and add vectors
index.train(normalized_embeddings)
index.add(normalized_embeddings)

print(f"\nFAISS index built successfully!")
print(f" Total vectors: {index.ntotal}")
print(f" Clusters: {nlist}")
print(f" Metric: Inner Product (Cosine Similarity)")
print(f" Type: IVFFlat")

# COMMAND ----------

# DBTITLE 1,Save FAISS Index to Volume
# Create directory if it doesn't exist
dbutils.fs.mkdirs(index_save_path)

# Save FAISS index
faiss.write_index(index, index_save_path.replace("dbfs:", "") + "index.faiss")

# Save chunks metadata
with open(index_save_path.replace("dbfs:", "") + "chunks_data.pkl", "wb") as f:
    pickle.dump(chunks_data, f)

print(f"FAISS index saved to: {index_save_path}index.faiss")
print(f"Metadata saved to: {index_save_path}chunks_data.pkl")
print(f"\n Persisted files:")
print(f"   - index.faiss ({index.ntotal} vectors)")
print(f"   - chunks_data.pkl ({len(chunks_data)} chunks metadata)")