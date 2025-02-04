
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os

app = FastAPI()

# Monte le dossier static
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route pour servir le favicon
@app.get('/favicon.ico', include_in_schema=False)
async def favicon():
    return FileResponse(os.path.join('static', 'favicon.ico'))


#----------------------------------------------------------------#
# Point de terminaison pour vérifier que l'API est fonctionnelle #
#----------------------------------------------------------------#
    
@app.get('/')
def get_verify():
    return {"message": "L'API est fonctionnelle."}
