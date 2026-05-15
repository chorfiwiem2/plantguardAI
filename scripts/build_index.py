import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rag.vectorstore import build_vectorstore

"""
Script à lancer UNE SEULE FOIS pour construire la base FAISS.
Relancer uniquement si vous ajoutez de nouveaux PDFs."""


if __name__ == "__main__":
    print("=" * 50)
    print("Construction de la base vectorielle FAISS")
    print("=" * 50)
    build_vectorstore()
    print("=" * 50)
    print("Terminé ! Vous pouvez maintenant lancer app.py")
    print("=" * 50)
