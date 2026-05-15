""" "cherche les infos" → prend le nom de la maladie identifiée, 
cherche dans FAISS les informations détaillées sur cette maladie, 
retourne les infos avec les sources."""

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.retriever import get_hybrid_retriever
import os
from dotenv import load_dotenv

load_dotenv()

_llm_cache = None

#La première fois que get_llm() est appelée, 
#elle crée le LLM et le stocke dans _llm_cache. Les fois suivantes, 
#elle retourne directement le LLM déjà créé sans en créer un nouveau.
def get_llm():
    global _llm_cache
    if _llm_cache is None:
        _llm_cache = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=1024
        )
    return _llm_cache

@tool
def recherche_base_agricole(query: str) -> str:
    """Recherche des informations confirmees sur une maladie dans la base agricole PDF."""
    try:
        retriever = get_hybrid_retriever()

        #cherche avec le nom de la maladie
        docs = retriever.invoke(query)

        if not docs:
            return "Aucun document trouve pour cette requete."

        #Limite à 400 caractères par doc
        context = "\n\n".join([d.page_content[:400] for d in docs[:4]])
        sources = list(set([
            os.path.basename(d.metadata.get("source", "Base agricole"))
            for d in docs
        ]))

        prompt = ChatPromptTemplate.from_template(
            "Expert phytopathologie. Contexte:\n{context}\n\nQuestion: {question}\n\nReponds en 5 lignes max en francais."
        )

        chain = prompt | get_llm() | StrOutputParser()
        answer = chain.invoke({"context": context, "question": query})

        return f"INFO:\n{answer}\n\nSOURCES: {', '.join(sources)}"

    except Exception as e:
        return f"Erreur recherche base: {str(e)}"