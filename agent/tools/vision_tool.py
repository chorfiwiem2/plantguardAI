from langchain_core.tools import tool
import google.generativeai as genai
import PIL.Image
import os
from dotenv import load_dotenv

load_dotenv()

@tool
def analyser_image_plante(image_path: str) -> str:
    """Analyse visuellement une photo de plante pour identifier une maladie."""
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            return "Clé GEMINI_API_KEY manquante dans le fichier .env"

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-1.5-flash")

        clean_path = image_path.strip().strip('"').strip("'")
        if not os.path.exists(clean_path):
            return f"Image introuvable : {clean_path}"

        image = PIL.Image.open(clean_path)

        response = model.generate_content([
            """Tu es expert en phytopathologie. Analyse cette image de plante.

Reponds en francais :
1. PLANTE : quelle plante vois-tu ?
2. SYMPTOMES : decris exactement ce que tu vois
3. MALADIE PROBABLE : nom precis

Guide :
- Poudre BLANCHE sur feuilles = OIDIUM
- Taches JAUNES angulaires delimitees par nervures = MILDIOU
- Taches BRUNES avec anneaux concentriques = ALTERNARIOSE
- Pustules ORANGE sous feuilles = ROUILLE
- Fletrissement + jaunissement = FUSARIOSE
- Pourriture molle grise = BOTRYTIS

4. CONFIANCE : %""",
            image
        ])
        return response.text

    except Exception as e:
        return f"Erreur vision: {str(e)}"