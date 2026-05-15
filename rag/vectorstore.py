from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from rag.loader import load_documents
import os

def get_embeddings():

    """
    Crée le modèle d'embeddings.
    sentence-transformers convertit le texte en vecteurs numériques.
    all-MiniLM-L6-v2 est gratuit et performant.
    """
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True} #normalise les vecteurs pour que la recherche cosine soit plus précise.
    )

def build_vectorstore(docs_path="data/documents/", 
                      index_path="data/faiss_index"):
    
    """
    Construit la base vectorielle FAISS depuis les PDFs.
    À lancer UNE SEULE FOIS (ou quand vous ajoutez des PDFs).
    """

    print("Construction de la base FAISS...")

     # Charge et découpe les PDFs
    chunks = load_documents(docs_path)

    # Crée les embeddings (chaque chunk en vecteur)
    embeddings = get_embeddings()

    # Construit FAISS depuis les chunks
    print("Creation des vecteurs...")

    #transforme chaque chunk en vecteur puis stocke tous les vecteurs
    #dasn une structure FAISS 
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Sauvegarde base FAISS sur disque
    vectorstore.save_local(index_path) #cree deux fichiers .faiss (vecteurs) et .pk1(métadonnees)(sasn ça tout sera perdu au redemarrage)
    print(f"Base FAISS sauvegardee dans {index_path}")
    return vectorstore

def load_vectorstore(index_path="data/faiss_index"):
    """
    Charge la base FAISS existante depuis le disque.
    Beaucoup plus rapide que de la reconstruire.
    """

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"Base FAISS introuvable. Lancez d abord: py scripts/build_index.py"
        )
    embeddings = get_embeddings()

    vectorstore = FAISS.load_local(
        index_path, 
        embeddings, 
        allow_dangerous_deserialization=True #mesure de sécurité bligatoire avec les nouvelles versions de LangChain
    )
    
    print("Base FAISS chargee")
    return vectorstore