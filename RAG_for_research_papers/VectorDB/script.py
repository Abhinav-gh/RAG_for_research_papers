from pathlib import Path
import chromadb
import pickle
import os
from dotenv import load_dotenv
load_dotenv()

multiquery_rag_output_path = "../RAG Results/multiquery_rag_results.txt"
Relative_Database_path = "./chroma_Data"
Absolute_Database_path = Path(Relative_Database_path).resolve()
file_path = "../Chunking/Chunk_files/harry_potter_chunks_semantic.pkl"
# Create a new collection with a unique name
collection_name = "harry_potter_collection"
# Set API key
# os.environ["GOOGLE_API_KEY"] = os.environ.get("GEMINI_API_KEY")

# Initialize the persistent client
client = chromadb.PersistentClient(path=Absolute_Database_path)
print(f"[INFO] ChromaDB client initialized at: {Absolute_Database_path}")

# List existing collections
existing_collections = client.list_collections()
print(f"Existing collections: {[c.name for c in existing_collections]}")


# No need for fitz or RecursiveCharacterTextSplitter here, as we are loading from a file.


loaded_docs = []

try:
    with open(file_path, "rb") as f: # 'rb' mode for reading in binary
        loaded_docs = pickle.load(f)
    print(f"Successfully loaded {len(loaded_docs)} chunks from '{file_path}'.")
except FileNotFoundError:
    print(f"Error: The file '{file_path}' was not found.")
except Exception as e:
    print(f"Error loading file: {e}")

# Now you can inspect the loaded documents to verify.
print("\nHere is the metadata of a loaded chunk:")
if loaded_docs:
    print(loaded_docs[0].metadata)

# Install if needed
# !pip install sentence_transformers

# Set up embedding function
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
embedding_function = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
print("Embedding function initialized with model: all-MiniLM-L6-v2")

from datetime import datetime

# FORCE DELETE the collection if it exists
try:
    client.delete_collection(name=collection_name)
    print(f"[INFO] Deleted existing collection '{collection_name}'")
except Exception as e:
    print(f"[INFO] No existing collection named '{collection_name}' to delete.")

# Create a FRESH collection
collection = client.create_collection(
    name=collection_name,
    embedding_function=embedding_function,
    metadata={
        "description": "Julius Caesar Chunks collection for RAG",
        "created": str(datetime.now())
    }
)

print(f"[SUCCESS] Fresh collection '{collection_name}' created successfully")
print(f"Current count in collection: {collection.count()}")

import uuid

# Prepare documents for ChromaDB
ids = []
documents = []
metadatas = []

# Process each loaded document chunk
for i, doc in enumerate(loaded_docs):
    # Generate a unique ID (you could use a more deterministic approach if needed)
    doc_id = f"hp_chunk_{i}"
    
    # Get the document text
    document_text = doc.page_content
    
    # Get the document metadata and add id and ischunk fields
    metadata = doc.metadata.copy()
    metadata["id"] = doc_id
    metadata["ischunk"] = True
    
    # Add to our lists
    ids.append(doc_id)
    documents.append(document_text)
    metadatas.append(metadata)

# Add documents in batches to avoid memory issues
batch_size = 500
total_added = 0

for i in range(0, len(ids), batch_size):
    end_idx = min(i + batch_size, len(ids))
    
    # collection.update(
    #     ids=ids[i:end_idx],
    #     documents=documents[i:end_idx],
    #     metadatas=metadatas[i:end_idx]
    # )
    collection.add(
        ids=ids[i:end_idx],
        documents=documents[i:end_idx],
        metadatas=metadatas[i:end_idx]
    )
    
    total_added += end_idx - i
    print(f"Added batch: {i} to {end_idx-1} ({end_idx-i} items)")

print(f"Successfully added {total_added} documents to collection '{collection_name}'")

