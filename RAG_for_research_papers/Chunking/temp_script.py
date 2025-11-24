from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceEmbeddings
import re

def is_valid_chunk_for_bert(text):
    """
    Check if a chunk is valid for BERT pre-training.
    - Should have at least 2 complete sentences
    - Should not be a half-cut sentence
    - Should have minimum length for meaningful content
    """
    # Remove extra whitespace
    text = text.strip()
    
    # Check minimum length (at least 100 characters for meaningful content)
    if len(text) < 100:
        return False
    
    # Count sentences (look for sentence endings)
    sentence_endings = re.findall(r'[.!?]+', text)
    if len(sentence_endings) < 2:
        return False
    
    # Check if text ends with a complete sentence
    if not re.search(r'[.!?]\s*$', text):
        return False
    
    # Check if text starts properly (not a fragment)
    if text[0].islower():  # Likely a sentence fragment
        return False
    
    return True

# File path to your PDF
file_path = "../harrypotter.pdf"

# A list to store chunks
docs_semantic = []

try:
    with fitz.open(file_path) as pdf_doc:
        # Initialize HuggingFace embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # Initialize semantic chunker with embeddings
        text_splitter = SemanticChunker(embeddings)

        for page_num, page in enumerate(pdf_doc):
            # Extract text from the current page
            page_text = page.get_text()

            # Skip empty pages
            if not page_text.strip():
                continue

            # Split the text into semantic chunks
            page_chunks = text_splitter.create_documents([page_text])

            # Filter and add metadata to each valid chunk
            for chunk in page_chunks:
                # Validate chunk for BERT pre-training
                if is_valid_chunk_for_bert(chunk.page_content):
                    chunk.metadata.update({
                        "source": file_path, 
                        "page_number": page_num + 1,
                        "c": "semantic",  # Added metadata field 'c'
                        "ischunk": True  # Added ischunk field
                    })
                    docs_semantic.append(chunk)

    print("✅ Successfully loaded and chunked the book content from the PDF with semantic awareness + page numbers.")
    print(f"Filtered chunks for BERT pre-training quality.")
except FileNotFoundError:
    print(f"❌ Error: The file '{file_path}' was not found. Please make sure the file exists.")
    exit()

# Print some information about the chunks to verify
print(f"Total number of valid chunks created: {len(docs_semantic)}")
print("\nHere is the content of the first chunk:")
print("---------------------------------------")
print(docs_semantic[0].page_content)
print("---------------------------------------")
print(f"First chunk metadata: {docs_semantic[0].metadata}")