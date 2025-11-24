import chromadb
from chromadb.config import Settings
import os
import numpy as np

CHROMA_DB_PATH = "/home/tanish/ANLP_Proj/RAG_for_research_papers/VectorDB/chroma_Data_with_Fine_tuned_BERT"
CHROMA_COLLECTION_NAME = "HP_Chunks_BERT_Finetuned_collection"

class ChromaDBClient:
    def __init__(self, persist_directory=CHROMA_DB_PATH):
        print(">>> DEBUG: Initializing ChromaDBClient")

        try:
            print(">>> Using persist dir:", persist_directory)
            self.client = chromadb.PersistentClient(
                path=persist_directory,
                settings=Settings(anonymized_telemetry=False)
            )
        except Exception as e:
            print(">>> ERROR creating PersistentClient:", e)
            raise

        try:
            collections = self.client.list_collections()
            print(">>> Collections:", [c.name for c in collections])
        except Exception as e:
            print(">>> ERROR listing collections:", e)
            raise

        # Ensure chosen collection exists
        if CHROMA_COLLECTION_NAME not in [c.name for c in collections]:
            raise RuntimeError(
                f"[FATAL] Collection '{CHROMA_COLLECTION_NAME}' not found. "
                f"Available: {[c.name for c in collections]}"
            )

        try:
            self.collection = self.client.get_collection(CHROMA_COLLECTION_NAME)
            print(f">>> Loaded collection: {CHROMA_COLLECTION_NAME}")
        except Exception as e:
            print(">>> ERROR loading collection:", e)
            raise

        # Get sample embedding
        try:
            sample = self.collection.get(include=["embeddings"], limit=1)
            embs = sample.get("embeddings", [])
            if embs:
                print(">>> Sample emb shape:", np.array(embs[0]).shape)
            else:
                print(">>> No embeddings in DB")
        except Exception as e:
            print(">>> ERROR reading sample embeddings:", e)

    def query(self, query_embedding, top_k=10):
        try:
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where={"ischunk": True}  
            )
            return results["documents"][0], results["ids"][0]
        except Exception as e:
            print(">>> ERROR in collection.query:", e)
            raise e
