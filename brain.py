import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class Brain:
    def __init__(self):
        keys_str = os.getenv("GEMINI_API_KEYS") or os.getenv("GEMINI_API_KEY")
        if not keys_str:
            print("🔴 ERREUR : Pas de clé API trouvée dans .env")
            self.api_keys = []
        else:
            self.api_keys = [k.strip() for k in keys_str.split(',')]
            
        self.current_key_index = 0
        self.history = [] 
        self.system_prompt = "Tu es un assistant utile."
        self.model = None
        self.chat = None
        self.model_name = "gemini-2.5-flash"
        self.init_model()

    def update_persona(self, name, scenario, behavior):
        """Met à jour l'identité de l'IA"""
        self.system_prompt = f"""
        Tu incarnes {name}.
        SCÉNARIO : {scenario}
        COMPORTEMENT : {behavior}
        
        RÈGLES :
        - Réponses conversationnelles et naturelles.
        - Ne dis jamais que tu es une IA.
        - Réponses courtes (1-2 phrases) sauf demande contraire.
        """
        print(f"🧠 Persona mise à jour : {name}")
        self.history = [] 
        self.init_model() 

    def init_model(self):
        if not self.api_keys: return

        genai.configure(api_key=self.api_keys[self.current_key_index])
        try:
            # --- CORRECTION ICI : On utilise le modèle 1.5 Flash (Stable) ---
            self.model = genai.GenerativeModel(
                model_name="gemini-2.5-flash", 
                system_instruction=self.system_prompt
            )
            self.chat = self.model.start_chat(history=self.history)
            print("🧠 Modèle Gemini 2.5 Flash Lite.")
        except Exception as e:
            print(f"🔴 Erreur chargement modèle : {e}")

    def think_streaming(self, user_text):
        if not self.chat: return
        try:
            response = self.chat.send_message(user_text, stream=True)
            buffer = ""
            for chunk in response:
                text = chunk.text
                buffer += text
                if any(p in text for p in [".", "?", "!", "\n"]):
                    if len(buffer) > 5:
                        yield buffer
                        buffer = ""
            if buffer: yield buffer
        except Exception as e:
            yield "Désolé, j'ai un petit bug de cerveau."
            print(f"🔴 Erreur Chat : {e}")


    def analyze_pitch(self, prompt_context):
        """Analyse le pitch selon la méthode QQOQCP + Structure Idéale"""
        print(f"📊 Envoi à {self.model_name} pour analyse structurée...")
        
        analysis_prompt = f"""
        Tu es un expert en Pitch de Startup (Type Y-Combinator).
        Analyse ce pitch en vérifiant la présence des 9 points clés de la structure idéale :
        
        1. POURQUOI (Le problème/Accroche)
        2. QUOI (La solution)
        3. QUI (La cible)
        4. COMMENT (Le fonctionnement)
        5. OÙ/QUAND (Contexte/Marché)
        6. POURQUOI TOI (Différenciation)
        7. ARGENT (Modèle éco - Optionnel mais bon à savoir)
        8. APPEL À L'ACTION (Ce que tu veux)
        
        CONTEXTE ET STATS DU PITCHEUR :
        {prompt_context}
        
        CONSIGNES DE RÉPONSE (JSON STRICT) :
        Tu dois noter SÉVÈREMENT. Si un point clé est absent, dis-le.
        
        Réponds UNIQUEMENT avec ce JSON :
        {{
            "note": "Note/100",
            "accroche_probleme": "Analyse du WHY et du problème (1 phrase)",
            "solution_cible": "Analyse du QUOI et QUI (1 phrase)",
            "unicite_business": "Analyse du POURQUOI TOI et du MODÈLE ÉCO (1 phrase)",
            "cta_action": "Analyse de l'APPEL À L'ACTION (1 phrase)",
            "elements_manquants": "Liste les points oubliés parmi les 9 (ex: 'Manque le Business Model, Manque le CTA...')",
            "conseil": "Le conseil prioritaire pour améliorer la structure"
        }}
        """
        
        try:
            response = self.model.generate_content(analysis_prompt)
            clean_json = response.text.replace("```json", "").replace("```", "").strip()
            return clean_json
        except Exception as e:
            print(f"🔴 CRASH ANALYSE : {e}")
            return '{"note": "0", "accroche_probleme": "Erreur", "solution_cible": "Erreur", "unicite_business": "Erreur", "cta_action": "Erreur", "elements_manquants": "Erreur", "conseil": "Erreur technique"}'