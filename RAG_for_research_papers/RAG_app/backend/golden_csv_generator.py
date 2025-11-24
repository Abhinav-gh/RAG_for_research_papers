import csv
import chromadb
from chromadb.config import Settings

INPUT_CSV = "../../LLM Caller/generated_pairs_without_commas.csv"       # Modify if needed
OUTPUT_CSV = "golden.csv"     # Output file
CHROMA_DIR = "../../VectorDB/chroma_Data_with_Fine_tuned_BERT"    # Path to Chroma persistence directory
COLLECTION_NAME = "HP_Chunks_BERT_Finetuned_collection" # Your Chroma collection name


def load_chroma_collection():
    """Load persistent Chroma collection."""
    client = chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(anonymized_telemetry=False)
    )

    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        raise RuntimeError(f"Collection '{COLLECTION_NAME}' not found in ChromaDB.")

    return collection


def fetch_chunk_content(collection, chunk_id):
    """Retrieve the chunk text from ChromaDB using chunk_id."""
    try:
        result = collection.get(ids=[chunk_id])
        documents = result.get("documents", [])
        if documents and len(documents) > 0:
            return documents[0]
    except Exception as e:
        print(f"Warning: Could not fetch chunk '{chunk_id}': {e}")

    return ""  # return empty string if missing


def create_golden_csv():
    """Read the input CSV, fetch chunk content, and create golden.csv"""

    # Load collection
    collection = load_chroma_collection()

    rows = []

    # Read input CSV: query, chunk_id
    with open(INPUT_CSV, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            query = row["query"]
            chunk_id = row["chunk_id"]

            # Fetch answer from ChromaDB
            answer_text = fetch_chunk_content(collection, chunk_id)

            rows.append({"query": query, "answer": answer_text})

    # Write golden.csv
    with open(OUTPUT_CSV, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["query", "answer"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Golden dataset created: {OUTPUT_CSV}")


if __name__ == "__main__":
    create_golden_csv()