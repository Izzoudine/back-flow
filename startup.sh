#!/bin/bash

# 1. Gestion du Virtual Environment
if [ ! -d "venv" ]; then
    echo "🐍 Création de l'environnement virtuel Python..."
    python3 -m venv venv
else
    echo "✅ Environnement virtuel déjà présent."
fi

source venv/bin/activate

# 2. Installation des dépendances Python
echo "📦 Vérification des dépendances..."
pip install -r requirements.txt > /dev/null 2>&1

# 3. Installation de Ngrok (Seulement si absent)
if ! command -v ngrok &> /dev/null
then
    echo "📡 Ngrok non trouvé. Installation en cours..."
    curl -s https://ngrok-agent.s3.amazonaws.com/ngrok.asc | sudo tee /etc/apt/trusted.gpg.d/ngrok.asc >/dev/null
    echo "deb https://ngrok-agent.s3.amazonaws.com buster main" | sudo tee /etc/apt/sources.list.d/ngrok.list
    sudo apt update && sudo apt install ngrok -y
else
    echo "✅ Ngrok est déjà installé."
fi

# 4. Nettoyage des processus existants (pour éviter les doublons)
echo "🧹 Nettoyage des anciens processus..."
pkill -f "python3 -u server.py"
pkill -f "ngrok"

# 5. Lancement du Serveur Python
echo "🚀 Démarrage de server.py..."
nohup python3 -u server.py > server.log 2>&1 &
SERVER_PID=$!
echo "   ↳ Serveur lancé (PID: $SERVER_PID)"

# 6. Lancement de Ngrok
echo "🌍 Démarrage du tunnel Ngrok..."
# On utilise > /dev/null car on va récupérer l'URL proprement via l'API locale de Ngrok
nohup ngrok http 8001 > /dev/null 2>&1 &

# 7. ATTENTE ET AFFICHAGE DE L'URL (La partie magique)
echo "⏳ Attente de la génération de l'URL..."
sleep 5

# On interroge l'API locale de Ngrok pour choper l'URL publique
PUBLIC_URL=$(curl -s localhost:4040/api/tunnels | grep -o '"public_url":"[^"]*' | grep -o 'https://[^"]*')

echo "-----------------------------------------------------"
if [ -z "$PUBLIC_URL" ]; then
    echo "🔴 Erreur : Impossible de récupérer l'URL Ngrok."
    echo "   Vérifie que tu as bien mis ton token : ngrok config add-authtoken TON_TOKEN"
else
    echo "✅ SUCCÈS ! Tout tourne."
    echo ""
    echo "👉 COPIE CETTE URL DANS TON INDEX.HTML (PROD_API) :"
    echo ""
    echo "   $PUBLIC_URL"
    echo ""
    echo "-----------------------------------------------------"
fi