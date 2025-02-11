from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

from src.api.models import PredictionRequest
from src.ml.predict import predict

# Création de l'application FastAPI
app = FastAPI(
    title="London Fire Brigade Response Time API",
    description="API pour prédire le temps de réponse des pompiers de Londres",
    version="1.0.0"
)

# Montage des fichiers statiques
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Route pour le favicon
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join('src/api/static', 'favicon.ico'))

# Route de vérification
@app.get('/verify')
def verify():
    """Vérifie que l'API est fonctionnelle."""
    return {"message": "L'API est fonctionnelle."}

# Route de prédiction
@app.post('/predict')
def prediction(request: PredictionRequest):
    """
    Prédit le temps de réponse des pompiers.
    
    Args:
        request (PredictionRequest): Données de la requête
        
    Returns:
        dict: Prédiction et informations associées
    """
    prediction = predict(
        request.address,
        request.HourOfCall,
        request.IncidentGroup,
        request.PropertyCategory
    )
    
    return prediction