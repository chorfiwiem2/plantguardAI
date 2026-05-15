"""
Outil 4 — Générateur de fiche de traitement
Produit une fiche structurée et complète prête à l'emploi.
"""
from langchain.tools import tool
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def generer_fiche_traitement(maladie: str) -> str:
    """
    Génère une fiche de traitement complète, structurée et 
    professionnelle pour une maladie de plante identifiée.
    Utiliser TOUJOURS en dernière étape après confirmation 
    du diagnostic par la recherche dans la base agricole.
    L'argument doit être le nom précis de la maladie.
    """
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
        api_key=os.getenv("GROQ_API_KEY")
    )
    
    prompt = f"""Tu es un expert phytosanitaire.
Génère une fiche de traitement complète pour : **{maladie}**

Utilise exactement cette structure :

═══════════════════════════════════════
🌿 FICHE DE TRAITEMENT — {maladie.upper()}
═══════════════════════════════════════

📋 IDENTIFICATION
- Pathogène responsable : [champignon/bactérie/virus + nom latin]
- Niveau de gravité : [faible / moyen / élevé]
- Cultures touchées : [liste des cultures]

🔍 SYMPTÔMES CARACTÉRISTIQUES
- [symptôme 1]
- [symptôme 2]
- [symptôme 3]

💊 TRAITEMENT CHIMIQUE
- Produit 1 : [nom commercial] — dosage : [X g/L ou mL/L]
- Produit 2 : [nom commercial] — dosage : [X g/L ou mL/L]

🌱 TRAITEMENT BIOLOGIQUE (alternatif)
- [produit bio] — dosage : [X g/L]

📅 CALENDRIER D'APPLICATION
- Fréquence : [ex: toutes les 7-10 jours]
- Durée : [ex: 3 à 4 applications]
- Période conseillée : [ex: début de saison, avant floraison]

🛡️ MESURES PRÉVENTIVES
- [mesure 1]
- [mesure 2]
- [mesure 3]

⚠️ PRÉCAUTIONS DE SÉCURITÉ
- EPI requis : [gants, masque, lunettes...]
- Délai avant récolte : [X jours]
- Stockage : [conditions de stockage du produit]

═══════════════════════════════════════

Sois très précis sur les dosages. Réponds en français."""
    
    response = llm.invoke(prompt)
    return response.content