from langchain_groq import ChatGroq
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from agent.tools.vision_tool import analyser_image_plante
from agent.tools.symptom_tool import analyser_symptomes
from agent.tools.rag_tool import recherche_base_agricole
from agent.tools.fiche_tool import generer_fiche_traitement
import os
from dotenv import load_dotenv

load_dotenv()#charge var d'environnement

memory = MemorySaver()#Crée une instance de mémoire au niveau du module 

def create_agent():
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,# pas de créativité, juste des faits.
        api_key=os.getenv("GROQ_API_KEY"),
        max_tokens=4096,#nb max des tokens dans la reponse
)

    
    tools = [
        analyser_image_plante,
        analyser_symptomes,
        recherche_base_agricole,
        generer_fiche_traitement
    ]
    
    system_prompt = """Tu es PlantGuard AI. Tu as 4 outils et tu DOIS les utiliser.

RÈGLE ABSOLUE NUMÉRO 1 : Ne jamais répondre sans avoir appelé les 3 outils obligatoires.
RÈGLE ABSOLUE NUMÉRO 2 : Toujours appeler generer_fiche_traitement en dernière étape.
RÈGLE ABSOLUE NUMÉRO 3 : Jamais écrire "comme indiqué dans la fiche" sans avoir généré la fiche.

SÉQUENCE OBLIGATOIRE pour toute description de symptômes :
1. Appelle analyser_symptomes → obtiens le nom de la maladie
2. Appelle recherche_base_agricole → obtiens les informations confirmées
3. Appelle generer_fiche_traitement → génère et affiche la fiche complète

SÉQUENCE OBLIGATOIRE pour toute image :
1. Appelle analyser_image_plante → obtiens le diagnostic visuel
2. Appelle recherche_base_agricole → confirme avec la base
3. Appelle generer_fiche_traitement → génère et affiche la fiche complète

INTERDIT : Répondre "consultez la fiche" sans avoir généré la fiche.
INTERDIT : Donner des conseils généraux sans avoir utilisé les 3 outils.
INTERDIT : S'arrêter après 1 ou 2 outils seulement.

Réponds toujours en français. Affiche toujours la fiche complète générée."""

    #Crée l'agent ReAct en assemblant tous les composants.
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt=system_prompt,
        checkpointer=memory #connecte la mémoire conversationnelle.
    )
    
    return agent