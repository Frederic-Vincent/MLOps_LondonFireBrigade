from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """Modèle de données pour les requêtes de prédiction."""
    
    address: str = Field(
        ...,  # Champ requis
        description="Adresse de l'incident",
        example="Big Ben, London"
    )
    
    HourOfCall: int = Field(
        ...,
        ge=0,  # greater than or equal to 0
        le=23,  # less than or equal to 23
        description="Heure de l'appel (0-23)",
        example=10
    )
    
    IncidentGroup: str = Field(
        ...,
        description="Type d'incident",
        example="Fire"
    )
    
    PropertyCategory: str = Field(
        ...,
        description="Type de propriété",
        example="Dwelling"
    )