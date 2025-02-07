""""
-------------------------------------------------------------------------------
                            Import des bibliothèques
-------------------------------------------------------------------------------
"""

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os

from src.predict.predict import predict


""""
-------------------------------------------------------------------------------
                            Création de l'API
                        Route pour servir le favicon
                    Modèle de données pour les requêtes de prédiction
-------------------------------------------------------------------------------
"""

app = FastAPI()

# Monte le dossier static
app.mount("/static", StaticFiles(directory="src/api/static"), name="static")

# Route pour servir le favicon
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join('src/api/static', 'favicon.ico'))

# Modèle de données pour les requêtes de prédiction
class X_predict(BaseModel):
    address: str
    HourOfCall: int
    IncidentGroup: str
    PropertyCategory: str

#----------------------------------------------------------------#
# Point de terminaison pour vérifier que l'API est fonctionnelle #
#----------------------------------------------------------------#
    
@app.get('/verify')
def get_verify():
    return {"message": "L'API est fonctionnelle."}

@app.post('/predict')
def post_predict(
    x_predict: X_predict):

    # TESTS UNITAIRES
    # ....
    print(f"Adresse: {x_predict.address}")

    # TRAITEMENT - Prédiction du temps de réponse
    prediction = predict(x_predict.address,
                   x_predict.HourOfCall,
                   x_predict.IncidentGroup,
                   x_predict.PropertyCategory
                   )
    
    # LOGS
    # ...

    return prediction