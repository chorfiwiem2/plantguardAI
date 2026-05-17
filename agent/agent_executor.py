from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from agent.tools.vision_tool import analyser_image_plante
from agent.tools.symptom_tool import analyser_symptomes
from agent.tools.rag_tool import recherche_base_agricole
from agent.tools.fiche_tool import generer_fiche_traitement
import os
from dotenv import load_dotenv

load_dotenv()

# Mémoire persistante pour toutes les sessions
memory = MemorySaver()

def create_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=4096,
    )

    tools = [
        analyser_image_plante,
        analyser_symptomes,
        recherche_base_agricole,
        generer_fiche_traitement
    ]
    
    system_prompt = """Tu es PlantGuard AI, un expert phytopathologiste. Tu aides les agriculteurs à identifier et traiter les maladies des plantes.

RÈGLES STRICTES:
1. Pour toute description de symptômes ou image, tu DOIS utiliser la séquence:
   - analyser_symptomes (ou analyser_image_plante si image) → identifier la maladie
   - recherche_base_agricole → chercher les infos dans la base
   - generer_fiche_traitement → générer la fiche complète

2. Si l'utilisateur demande "le traitement" ou "la fiche" et qu'une maladie a déjà été identifiée dans la conversation, utilise directement generer_fiche_traitement avec cette maladie.

3. Ne jamais inventer d'informations. Base-toi uniquement sur les outils.

4. Réponds toujours en français, de manière structurée et professionnelle."""

    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=memory  # Active la mémoire conversationnelle
    )
    
    return agent