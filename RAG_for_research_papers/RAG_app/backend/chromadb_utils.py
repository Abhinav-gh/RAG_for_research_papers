import chromadb
from chromadb.config import Settings

CHROMA_DB_PATH = "../../VectorDB/chroma_Data_with_Fine_tuned_BERT"

class ChromaDBClient:
    def __init__(self, persist_directory=CHROMA_DB_PATH):
        self.client = chromadb.Client(Settings(persist_directory=persist_directory))
        # You may need to specify the collection name if not default
        self.collection = self.client.list_collections()[0] if self.client.list_collections() else self.client.get_or_create_collection(name="default")

    def query(self, query_embedding, top_k=10):
        # ChromaDB expects embeddings as list of lists
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k
        )
        # Return top-k chunks and their ids
        return results['documents'][0], results['ids'][0]
