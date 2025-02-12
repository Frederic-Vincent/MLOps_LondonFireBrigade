"""
---------------------------------------------------------------------------------------------------
Nom du script : api.py

Dernière mise à jour : mercredi 12 février 2025

Fonction de ce script : API pour la prédiction du temps de réponse des pompiers

Tâches réalisées par ce script :
 - Configuration de l'API FastAPI
 - Définition des routes
 - Gestion des requêtes de prédiction
---------------------------------------------------------------------------------------------------
"""

""""
---------------------------------------------------------------------------------------------------
                            Import des bibliothèques
---------------------------------------------------------------------------------------------------
"""

import os
import time
import logging
from pathlib import Path
from datetime import datetime, timedelta
import pytz
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
#from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm

#from jose import JWTError, jwt

from src.api.models import PredictionRequest
# from src.api.security import (
#     Token, User, authenticate_user, create_access_token,
#     fake_users_db, ACCESS_TOKEN_EXPIRE_MINUTES, SECRET_KEY, ALGORITHM
# )
from src.ml.predict import predict

""""
---------------------------------------------------------------------------------------------------
                            Configuration des chemins et des logs
---------------------------------------------------------------------------------------------------
"""
# Chemins
STATIC_DIR = Path("src/api/static")
LOG_DIR = Path("logs")
LOG_FILE = LOG_DIR / "api.log"

class ParisTimeFormatter(logging.Formatter):
    """Formateur personnalisé pour les logs avec le fuseau horaire de Paris."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.tz = pytz.timezone('Europe/Paris')

    def formatTime(self, record, datefmt=None):
        """Surcharge de la méthode de formatage du temps pour utiliser le fuseau horaire de Paris."""
        dt = datetime.fromtimestamp(record.created, self.tz)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

def setup_logging():
    """Configure et initialise le système de logging."""
    # Création du dossier de logs s'il n'existe pas
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Création du formateur avec le fuseau horaire de Paris
    formatter = ParisTimeFormatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S %z'
    )

    # Configuration des handlers
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    # Configuration de base du logging
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

    # Obtention du logger
    logger = logging.getLogger(__name__)

    # Activation de la capture des warnings
    logging.captureWarnings(True)

    return logger

# Initialisation du logger
logger = setup_logging()

""""
---------------------------------------------------------------------------------------------------
                            Création et configuration de l'API
---------------------------------------------------------------------------------------------------
"""
# Création de l'application FastAPI
app = FastAPI(
    title="London Fire Brigade Response Time API",
    description="API pour prédire le temps de réponse des pompiers de Londres",
    version="1.0.0"
)

""""
---------------------------------------------------------------------------------------------------
                            Routes de l'API
---------------------------------------------------------------------------------------------------
"""

@app.on_event("startup")
async def startup_event():
    """Exécuté au démarrage de l'application."""
    logger.info(
        "\n" + "="*80 + "\n" +
        "                    Démarrage de l'API London Fire Brigade Response Time\n" +
        "="*80 + "\n"
    )

@app.on_event("shutdown")
async def shutdown_event():
    """Exécuté à l'arrêt de l'application."""
    logger.info(
        "\n" + "="*80 + "\n" +
        "                    Arrêt de l'API London Fire Brigade Response Time\n" +
        "="*80 + "\n\n"
    )

# Montage des fichiers statiques
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Fichier statique (Favicon)
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    """Route pour servir le favicon."""
    return FileResponse(STATIC_DIR / 'favicon.ico')

# Vérification
@app.get('/verify')
def verify():
    """Vérifie que l'API est fonctionnelle."""
    logger.info("Vérification du statut de l'API")
    return {"message": "L'API est fonctionnelle."}

# Prédiction
@app.post('/predict')
def prediction(request: PredictionRequest):
    """
    Route pour la prédiction du temps de réponse.
    
    Args:
        request (PredictionRequest): Données de la requête de prédiction
        
    Returns:
        dict: Résultats de la prédiction
    """
    start_time = time.time()
    
    # Log de la requête reçue
    logger.info(
        "\nNouvelle requête de prédiction reçue :\n"
        f" - Adresse: {request.address}\n"
        f" - Heure d'appel: {request.HourOfCall}\n"
        f" - Type d'incident: {request.IncidentGroup}\n"
        f" - Type de propriété: {request.PropertyCategory}\n"
    )
    
    try:
        # Calcul de la prédiction
        result = predict(
            request.address,
            request.HourOfCall,
            request.IncidentGroup,
            request.PropertyCategory
        )
        
        # Log du résultat
        processing_time = time.time() - start_time
        logger.info(
            "\nPrédiction réussie :\n"
            f" - Station: {result['station']}\n"
            f" - Distance: {result['DistanceToStation']:.3f} m\n"
            f" - Temps prédit: {result['prediction']:.1f} secondes\n"
            f" - Temps de traitement: {processing_time:.3f} secondes\n\n"
            + "-"*80  + "\n\n"
        )
        
        return result
        
    except Exception as e:
        # Log de l'erreur
        logger.error(f"Erreur lors de la prédiction : {str(e)}")
        raise HTTPException(
            status_code=500,
            detail="Erreur lors du calcul de la prédiction"
        )