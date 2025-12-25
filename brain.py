import os
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

class Brain:
    def __init__(self):
        # Récupération de la clé API
        # On cherche GEMINI_API_KEY ou GOOGLE_API_KEY
        keys_str = os.getenv("GEMINI_API_KEY")
        
        if not keys_str:
            print("🔴 ERREUR : Pas de clé API trouvée dans .env")
            self.api_keys = []
        else:
            self.api_keys = [k.strip() for k in keys_str.split(',')]
            
        self.current_key_index = 0
        self.history = [] 
        
        # Instruction de base
        self.persona_instruction = "Tu es un assistant utile."
        self.model = None
        self.chat = None
        
        # On utilise FLASH 1.5 : C'est le meilleur compromis Vitesse/Prix/Multimodal
        self.model_name = "gemini-2.5-flash"
        
        self.init_model()

    def update_persona(self, name, scenario, behavior):
        """Met à jour l'identité de l'IA"""
        self.persona_instruction = f"""
        Tu incarnes {name}.
        SCÉNARIO : {scenario}
        COMPORTEMENT : {behavior}
        
        RÈGLES IMPORTANTES :
        - Tu es dans une conversation ORALE.
        - Ne fais JAMAIS de listes à puces ou de formatage complexe (gras, italique).
        - Fais des phrases courtes, claires et percutantes.
        - Réagis directement à ce qu'on te dit (ou au ton de la voix).
        """
        print(f"🧠 Persona mise à jour : {name}")
        self.history = [] 
        self.init_model()

    def init_model(self):
        if not self.api_keys: return

        genai.configure(api_key=self.api_keys[self.current_key_index])
        try:
            self.model = genai.GenerativeModel(
                model_name=self.model_name, 
                system_instruction=self.persona_instruction
            )
            # On garde un historique vide au début
            self.chat = self.model.start_chat(history=[])
            print(f"🧠 Cerveau prêt : {self.model_name}")
        except Exception as e:
            print(f"🔴 Erreur chargement modèle : {e}")

    def think_from_audio(self, audio_path):
        """
        Reçoit un chemin de fichier audio (mp3/wav/webm),
        L'envoie à Gemini pour qu'il l'écoute,
        Et retourne la réponse textuelle.
        """
        try:
            print(f"👂 Brain écoute le fichier : {audio_path}")
            
            # 1. Upload du fichier vers les serveurs Google (c'est très rapide)
            # Note: Le mime_type peut être 'audio/mp3', 'audio/wav', 'audio/webm'
            audio_file = genai.upload_file(path=audio_path)
            
            # 2. Génération de la réponse
            # On envoie le fichier audio + le prompt système implicite (défini dans init_model)
            response = self.model.generate_content([
                "Écoute cet audio attentivement et réponds-moi en suivant ton persona.", 
                audio_file
            ])
            
            # 3. Nettoyage (Bonne pratique : on ne garde pas les fichiers chez Google)
            # (Optionnel, Google les supprime auto après 48h, mais on peut le faire ici)
            # genai.delete_file(audio_file.name)
            
            print(f"🧠 Réponse générée : {response.text[:50]}...")
            return response.text

        except Exception as e:
            print(f"🔴 Erreur Brain Audio : {e}")
            return "Désolé, je n'ai pas bien entendu. Peux-tu répéter ?"
         
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