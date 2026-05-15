"""prend la description de l'agriculteur, 
cherche dans FAISS les documents qui correspondent aux symptômes, 
et identifie la maladie."""

from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from rag.retriever import get_hybrid_retriever
import os
from dotenv import load_dotenv

load_dotenv()

_llm = None

#La première fois que get_llm() est appelée, 
#elle crée le LLM et le stocke dans _llm_cache. Les fois suivantes, 
#elle retourne directement le LLM déjà créé sans en créer un nouveau. 
def get_llm():
    global _llm
    if _llm is None:
        _llm = ChatGroq(
            model="llama-3.1-8b-instant",
            temperature=0,
            api_key=os.getenv("GROQ_API_KEY"),
            max_tokens=300
        )
    return _llm

@tool
def analyser_symptomes(description: str) -> str:
    """
    Analyse les symptomes en cherchant dans la base de documents
    agricoles pour identifier la maladie.
    """
    try:
        retriever = get_hybrid_retriever()

        # cherche dans FAISS avec la description complète des symptômes
        docs = retriever.invoke(description)

        if not docs:
            return "MALADIE: Non identifiée\nCONFIANCE: 0%\nAucun document trouvé."

        # Prend les 4 meilleurs documents
        context = "\n\n---\n\n".join([
            f"[Source: {os.path.basename(d.metadata.get('source','?'))}]\n{d.page_content}"
            for d in docs[:4]
        ])

        prompt = ChatPromptTemplate.from_template("""
Tu es expert en phytopathologie.

Voici des extraits de documents agricoles :
{context}

Un agriculteur décrit ces symptômes :
"{description}"

RÈGLE ABSOLUE : Base ton diagnostic UNIQUEMENT sur les documents fournis.
Si les documents parlent de la même maladie que les symptômes décrits, c'est cette maladie.
Ne jamais inventer une maladie qui n'est pas dans les documents.

Réponds en français :
MALADIE: [nom exact trouvé dans les documents]
CULTURE: [plante concernée]
SIGNES CLES: [symptômes des documents qui correspondent]
CONFIANCE: [%]
SOURCE: [nom du fichier source]
""")

        chain = prompt | get_llm() | StrOutputParser() #le prompt est formaté → envoyé au LLM → la réponse du LLM est parsée en texte simple.
        return chain.invoke({
            "context": context,
            "description": description
        })

    except Exception as e:
        return f"Erreur: {str(e)}"