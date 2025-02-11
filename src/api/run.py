import uvicorn

if __name__ == "__main__":
    # Configuration du serveur
    uvicorn.run(
        "src.api.api:app",  # Chaîne d'importation au lieu de l'import direct
        host="0.0.0.0",
        port=8000,
        reload=True  # Activer le rechargement automatique pendant le développement
    )