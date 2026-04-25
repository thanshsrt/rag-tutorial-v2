import argparse
import os
import shutil
from pathlib import Path

from langchain_community.document_loaders import (
    PyPDFDirectoryLoader,
    DirectoryLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from get_embedding_function import get_embedding_function
from langchain_chroma import Chroma


CHROMA_PATH = "chroma"
DATA_PATH = "data"


def main():

    # Check if the database should be cleared (using the --clear flag).
    parser = argparse.ArgumentParser()
    parser.add_argument("--reset", action="store_true", help="Reset the database.")
    args = parser.parse_args()
    if args.reset:
        print("✨ Clearing Database")
        clear_database()

    # Create (or update) the data store.
    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)


def load_documents():
    """Load PDFs AND code files."""
    all_documents = []
    
    # 1. Load PDFs
    if any(f.endswith('.pdf') for f in os.listdir(DATA_PATH)):
        print("📄 Loading PDFs...")
        pdf_loader = PyPDFDirectoryLoader(DATA_PATH)
        all_documents.extend(pdf_loader.load())
        
    # 2. Load Python files
    py_path = os.path.join(DATA_PATH, "code")
    if os.path.exists(py_path):
        print("🐍 Loading Python files...")
        # Use TextLoader for code files with proper encoding
        code_loader = DirectoryLoader(
            py_path,
            glob="**/*.py",
            loader_cls=TextLoader,
            loader_kwargs={'encoding': 'utf-8'},
            recursive=True
        )
        all_documents.extend(code_loader.load())
        
    # 3. Load other text files (markdown, etc.) (NEW)
    for ext in ['*.md', '*.txt', '*.js', '*.ts', '*.json']:
        try:
            text_loader = DirectoryLoader(
                DATA_PATH,
                glob=f"**/{ext}",
                loader_cls=TextLoader,
                loader_kwargs={'encoding': 'utf-8'},
                recursive=True
            )
            docs = text_loader.load()
            if docs:
                print(f"📄 Loaded {len(docs)} {ext} files")
                all_documents.extend(docs)
        except Exception as e:
            print(f"⚠️  Could not load {ext}: {e}")
    
    print(f"📚 Total documents loaded: {len(all_documents)}")
    return all_documents

def split_documents(documents: list[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)


def add_to_chroma(chunks: list[Document]):
    # Load the existing database.
    db = Chroma(
        persist_directory=CHROMA_PATH, embedding_function=get_embedding_function()
    )

    # Calculate Page IDs.
    chunks_with_ids = calculate_chunk_ids(chunks)

    # Add or Update the documents.
    existing_items = db.get(include=[])  # IDs are always included by default
    existing_ids = set(existing_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

    # Only add documents that don't exist in the DB.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        db.add_documents(new_chunks, ids=new_chunk_ids)
        # db.persist()
    else:
        print("✅ No new documents to add")


def calculate_chunk_ids(chunks):

    # This will create IDs like "data/monopoly.pdf:6:2"
    # Page Source : Page Number : Chunk Index

    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        
        # For PDFs: use page number
        if 'page' in chunk.metadata:
            page = chunk.metadata.get("page")
            current_page_id = f"{source}:{page}"
        else:
            # For code files: use line number approximation
            # or just file-based chunking
            current_page_id = source
            
        # If the page ID is the same as the last one, increment the index.
        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        # Calculate the chunk ID.
        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        # Add it to the page meta-data.
        chunk.metadata["id"] = chunk_id

    return chunks


def clear_database():
    if os.path.exists(CHROMA_PATH):
        shutil.rmtree(CHROMA_PATH)


if __name__ == "__main__":
    main()
