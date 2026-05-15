"""
app.py
------
PlantGuard AI — Pipeline manuel sans agent ReAct.
Séquence forcée : symptômes → RAG → fiche.
"""

import os
import sys
import tempfile
from pathlib import Path
from dotenv import load_dotenv
import gradio as gr
from PIL import Image as PILImage
import numpy as np

# Imports directs des outils (pas d'agent)
from agent.tools.symptom_tool import analyser_symptomes
from agent.tools.rag_tool import recherche_base_agricole
from agent.tools.fiche_tool import generer_fiche_traitement
from agent.tools.vision_tool import analyser_image_plante

load_dotenv()

print("Initialisation de PlantGuard AI (mode pipeline manuel)...")

# ─── CSS personnalisé ─────────────────────────────────────────────────────────

CUSTOM_CSS = """
:root {
    --green-dark:   #1a3d1f;
    --green-mid:    #2d6a37;
    --green-light:  #4caf50;
    --green-pale:   #e8f5e9;
    --green-accent: #81c784;
    --earth:        #5d4037;
    --cream:        #fafdf7;
    --text-dark:    #1b2e1d;
    --text-mid:     #4a6741;
    --border:       #c8e6c9;
    --shadow:       rgba(26, 61, 31, 0.12);
}

body, .gradio-container {
    background: var(--cream) !important;
    font-family: 'Georgia', 'Cambria', serif !important;
}

#plantguard-header {
    background: linear-gradient(135deg, var(--green-dark) 0%, var(--green-mid) 60%, #3d8b40 100%);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px var(--shadow);
}

#plantguard-header::before {
    content: "🌿";
    position: absolute;
    font-size: 120px;
    right: -10px;
    top: -20px;
    opacity: 0.12;
}

#plantguard-header h1 {
    color: white !important;
    font-size: 2rem !important;
    font-weight: 700 !important;
    margin: 0 0 6px 0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.3);
}

#plantguard-header p {
    color: var(--green-accent) !important;
    font-size: 1rem !important;
    margin: 0 !important;
}

#chatbot-box {
    border: 1.5px solid var(--border) !important;
    border-radius: 14px !important;
    background: white !important;
    box-shadow: 0 2px 12px var(--shadow) !important;
}

.message.bot {
    background: var(--green-pale) !important;
    color: #000000 !important;              /* texte noir */
    border-left: 4px solid var(--green-light) !important;
    border-radius: 0 12px 12px 0 !important;
}



.message.bot strong {
    color: #1a1a1a !important;
    font-weight: 700 !important;
}

.message.user {
    background: var(--green-dark) !important;
    color: white !important;
    border-radius: 12px 0 0 12px !important;
}

#input-row textarea {
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    background: white !important;
    font-family: 'Georgia', serif !important;
    font-size: 0.95rem !important;
    color: var(--text-dark) !important;
    padding: 12px !important;
}

#send-btn {
    background: var(--green-mid) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 12px 24px !important;
    cursor: pointer !important;
    transition: background 0.2s !important;
    box-shadow: 0 2px 8px rgba(45, 106, 55, 0.3) !important;
}

#send-btn:hover {
    background: var(--green-dark) !important;
}

#clear-btn {
    background: white !important;
    color: var(--earth) !important;
    border: 1.5px solid #d7ccc8 !important;
    border-radius: 10px !important;
    font-weight: 500 !important;
    padding: 10px 18px !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
}

#clear-btn:hover {
    background: #efebe9 !important;
}

#image-upload {
    border: 2px dashed var(--green-accent) !important;
    border-radius: 12px !important;
    background: var(--green-pale) !important;
}

.example-btn {
    background: white !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 8px !important;
    color: var(--text-mid) !important;
    font-size: 0.85rem !important;
    padding: 8px 14px !important;
    cursor: pointer !important;
    text-align: left !important;
    transition: all 0.2s !important;
}

.example-btn:hover {
    background: var(--green-pale) !important;
    border-color: var(--green-light) !important;
    color: var(--green-dark) !important;
}

@media (max-width: 768px) {
    #plantguard-header h1 { font-size: 1.5rem !important; }
}
"""

# ─── Questions d'exemple ─────────────────────────────────────────────────────

EXAMPLE_QUESTIONS = [
    "🍅 Mes tomates ont des taches brunes avec un duvet blanc dessous les feuilles",
    "🍇 Les feuilles de ma vigne sont couvertes d'une poudre blanche",
    "🌾 Mon blé présente des épis blanchis avec des sporodoches oranges",
    "🥔 Mes pommes de terre ont des taches avec des anneaux concentriques",
    "🍓 Mes fraises pourrissent avec un duvet gris, que faire ?",
    "🌿 Quelles précautions prendre pour appliquer un traitement cuivre ?",
    "💊 Quel est le traitement biologique contre le mildiou ?",
]

# ─── PIPELINE MANUEL (remplace l'agent ReAct) ───────────────────────────────

def run_pipeline_texte(description: str) -> str:
    """
    Pipeline forcé pour description textuelle :
    1. analyser_symptomes
    2. recherche_base_agricole
    3. generer_fiche_traitement
    """
    # Étape 1 : Diagnostic (.invoke metjode langchaine pour appeler un tool)
    diagnostic = analyser_symptomes.invoke({"description": description})
    
    # Extraction du nom de maladie (ligne commençant par MALADIE:)
    maladie = "Maladie non identifiée"

    for line in diagnostic.split("\n"): #Extrait le nom de la maladie depuis la réponse du tool.
        if line.strip().upper().startswith("MALADIE:"):
            maladie = line.split(":", 1)[1].strip() #Pour chaque ligne, si elle commence par "MALADIE:",
                                                    #on prend tout ce qui est après les : et on le stocke dans maladie
            break
    
    #Appelle les deux tools suivants avec le nom de la maladie trouvée.
    # Étape 2 : Recherche base agricole
    recherche = recherche_base_agricole.invoke({"query": maladie})
    
    # Étape 3 : Fiche traitement
    fiche = generer_fiche_traitement.invoke({"maladie": maladie})
    

    # Assemblage final (Assemble les 3 résultats en un seul texte)
    resultat = f"""🔍 **DIAGNOSTIC**
{diagnostic}

📚 **RECHERCHE BASE AGRICOLE**
{recherche}

📋 **FICHE DE TRAITEMENT**
{fiche}

---
*PlantGuard AI — Diagnostic assisté par IA. Consultez un agronome pour validation.*"""
    
    return resultat


def run_pipeline_image(image_path: str, description: str = "") -> str:
    """
    Pipeline forcé pour image :
    1. analyser_image_plante
    2. recherche_base_agricole
    3. generer_fiche_traitement
    """
    # Étape 1 : Analyse visuelle
    diagnostic_visuel = analyser_image_plante.invoke({"image_path": image_path})
    
    # Extraction du nom de maladie
    maladie = "Maladie non identifiée"
    for line in diagnostic_visuel.split("\n"):
        if "MALADIE PROBABLE" in line.upper() or line.strip().upper().startswith("4."):
            maladie = line.split(":", 1)[1].strip() if ":" in line else line.strip()
            break
    
    # Étape 2 : Recherche
    recherche = recherche_base_agricole.invoke({"query": maladie})
    
    # Étape 3 : Fiche
    fiche = generer_fiche_traitement.invoke({"maladie": maladie})
    
    resultat = f"""🔍 **ANALYSE VISUELLE**
{diagnostic_visuel}

📚 **RECHERCHE BASE AGRICOLE**
{recherche}

📋 **FICHE DE TRAITEMENT**
{fiche}

---
*PlantGuard AI — Diagnostic visuel assisté par IA. Consultez un agronome pour validation.*"""
    
    return resultat


# ─── Logique du chatbot ─────────────────────────────────────────────────────
#Fonction centrale qui décide quel pipeline lancer 

def chat_with_agent(message: str, history: list, image):

    if not message.strip() and image is None:
        return "Veuillez décrire les symptômes ou uploader une photo."

    # Sauvegarder l'image temporairement
    image_path = None
    if image is not None:
        if hasattr(image, 'save'):
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            image.save(tmp.name)
            image_path = tmp.name
        else:
            pil_img = PILImage.fromarray(image.astype('uint8'))
            tmp = tempfile.NamedTemporaryFile(suffix=".jpg", delete=False)
            pil_img.save(tmp.name)
            image_path = tmp.name

    try:
        if image_path:
            response = run_pipeline_image(image_path, message)
        else:
            response = run_pipeline_texte(message)

    #s'exécute toujours, même si une erreur survient —
    # il supprime le fichier temporaire de l'image pour ne pas saturer le disque.
    finally: 
        if image_path:
            try:
                os.unlink(image_path)
            except Exception:
                pass

    return response


def clear_conversation():
    return [], None, ""


def fill_example(example: str):
    clean = example.split(" ", 1)[1] if " " in example else example
    return clean


# ─── Générateur de réponse Gradio ────────────────────────────────────────────

def respond(message, history, image):
    if not message.strip() and image is None:
        return history, ""   #si l'utilisateur a envoyé un message vide sans image.

    history = history[-4:] if len(history) > 4 else history
    history.append({"role": "user", "content": message})
    yield history, ""

    bot_response = chat_with_agent(message, history, image)
    
    history.append({"role": "assistant", "content": bot_response})
    yield history, ""


# ─── Interface Gradio ────────────────────────────────────────────────────────

def build_interface():
    with gr.Blocks(
        title="PlantGuard AI – Diagnostic des Maladies des Plantes",
    ) as demo:

        gr.HTML("""
        <div id="plantguard-header">
            <h1>🌿 PlantGuard AI</h1>
            <p>Agent Intelligent de Détection et Traitement des Maladies des Plantes</p>
        </div>
        """)

        with gr.Row():
            with gr.Column(scale=3):
                chatbot = gr.Chatbot(
                    value=[],
                    elem_id="chatbot-box",
                    label="Conversation",
                    height=480,
                    render_markdown=True,
                    avatar_images=(
                        None,
                        "https://em-content.zobj.net/source/twitter/376/seedling_1f331.png",
                    ),
                )

                with gr.Row(elem_id="input-row"):
                    msg_input = gr.Textbox(
                        placeholder="Décrivez les symptômes observés sur votre plante...",
                        label="",
                        lines=2,
                        max_lines=4,
                        scale=5,
                        show_label=False,
                    )

                with gr.Row():
                    send_btn = gr.Button("🔍 Analyser", variant="primary", elem_id="send-btn", scale=2)
                    clear_btn = gr.Button("🗑️ Nouvelle conversation", elem_id="clear-btn", scale=1)

            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="📷 Photo de la plante (optionnel)",
                    type="pil",
                    elem_id="image-upload",
                    height=180,
                    sources=["upload", "clipboard"],
                )

                gr.Markdown("### 💬 Exemples de questions")
                for example in EXAMPLE_QUESTIONS:
                    ex_btn = gr.Button(example, size="sm", elem_classes=["example-btn"])
                    ex_btn.click(fn=fill_example, inputs=[gr.State(example)], outputs=[msg_input])

        with gr.Accordion("ℹ️ Comment utiliser PlantGuard AI ?", open=False):
            gr.Markdown("""
**PlantGuard AI** analyse vos plantes en 3 étapes automatiques :

1. **🔍 Diagnostic** : Identifie la maladie (texte ou photo)
2. **📚 Recherche** : Consulte la base agricole
3. **📋 Rapport** : Génère la fiche de traitement

**⚠️ Disclaimer** : Outil d'aide à la décision. Consultez un agronome agréé.
            """)

        # Événements
        send_btn.click(fn=respond, inputs=[msg_input, chatbot, image_input], outputs=[chatbot, msg_input])
        msg_input.submit(fn=respond, inputs=[msg_input, chatbot, image_input], outputs=[chatbot, msg_input])
        clear_btn.click(fn=clear_conversation, outputs=[chatbot, image_input, msg_input])

    return demo


if __name__ == "__main__":
    demo = build_interface()
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
        theme=gr.themes.Soft(
            primary_hue="green",
            secondary_hue="emerald",
            neutral_hue="slate",
        ),
    )