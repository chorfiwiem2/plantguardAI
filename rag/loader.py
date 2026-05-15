from langchain_community.document_loaders import (
    PyMuPDFLoader,
    DirectoryLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_documents(path="data/documents/"):

    """
    Charge tous les PDFs du dossier et les découpe en chunks.
    """

    # Vérifie que le dossier existe et contient des PDFs
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dossier {path} introuvable")

    all_docs = []

    # Charge les PDFs
    pdf_files = [f for f in os.listdir(path) if f.endswith('.pdf')]
    if pdf_files:
        print(f"Chargement de {len(pdf_files)} PDFs...")
        pdf_loader = DirectoryLoader(
            path,
            glob="**/*.pdf",
            loader_cls=PyMuPDFLoader,
            show_progress=True
        )
        all_docs.extend(pdf_loader.load())

    # Charge les TXTs
    txt_files = [f for f in os.listdir(path) if f.endswith('.txt')]
    if txt_files:
        print(f"Chargement de {len(txt_files)} TXTs...")
        for txt_file in txt_files:
            txt_path = os.path.join(path, txt_file)
            loader = TextLoader(txt_path, encoding="utf-8")
            all_docs.extend(loader.load())

    if not all_docs:
        raise ValueError(f"Aucun document PDF ou TXT trouve dans {path}")

    print(f"Total : {len(all_docs)} documents charges")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,  
        chunk_overlap=150,
        separators=["\n\n", "\n", ".", "!", "?", " "]
    )

    chunks = splitter.split_documents(all_docs)
    print(f"Total : {len(chunks)} chunks crees")
    return chunks