from langchain_community.retrievers import BM25Retriever
from rag.vectorstore import load_vectorstore
from langchain_core.documents import Document
from typing import List

#Elle va stocker le retriever une fois créé pour ne pas le recréer à chaque question. 
_retriever_cache = None

class SimpleHybridRetriever:
    #Le constructeur de la classe — appelé automatiquement quand on crée un objet SimpleHybridRetriever. 
    # Il reçoit les deux retrievers en paramètres.
    def __init__(self, faiss_retriever, bm25_retriever):
        self.faiss = faiss_retriever
        self.bm25 = bm25_retriever

    # Elle prend une question (query) en texte et retourne une liste de Documents.
    def invoke(self, query: str) -> List[Document]:
        
        try:
            faiss_docs = self.faiss.invoke(query)
        except:
            faiss_docs = []
        try:
            bm25_docs = self.bm25.invoke(query)
        except:
            bm25_docs = []

        seen = set() #détecter les répétitions. 
        combined = []

        for doc in faiss_docs + bm25_docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                combined.append(doc)
        return combined[:6] #Retourne les 6 premiers documents de la liste combinée.

def get_hybrid_retriever():
    """Crée le retriever hybride une seule fois et le met en cache dans _retriever_cache.
      Ainsi FAISS n'est chargé qu'une seule fois au démarrage, pas à chaque question"""

    global _retriever_cache

    #Sans ça, FAISS serait rechargé depuis le disque à chaque question,
    # ce qui prendrait 10-20 secondes à chaque fois.
    if _retriever_cache is not None:
        return _retriever_cache
    

    print("Chargement du retriever...")
    vectorstore = load_vectorstore()


    faiss_retriever = vectorstore.as_retriever( #Transforme le vectorstore FAISS en retriever utilisable
        search_type="similarity", #cherche par similarité de sens (cosine similarity entre vecteurs)
        search_kwargs={"k": 4} #4 documents les plus proches.
    )

    #Récupère 200 documents de FAISS 
    docs = vectorstore.similarity_search("maladie plante symptomes", k=200)

    #Crée le retriever BM25 depuis ces 200 documents.
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 4

    #Crée l'objet hybride en lui passant les deux retrievers,
    # et le stocke dans le cache global.
    _retriever_cache = SimpleHybridRetriever(faiss_retriever, bm25_retriever)
    print("Retriever pret")
    return _retriever_cache