# Prédiction du temps de réponse des pompiers de Londres

Projet de prédiction du temps de réponse des pompiers de Londres utilisant le Machine Learning.

## Structure du Projet

```
.
├── data/                           # Données du projet
│   ├── 1_raw/                     # Données brutes
│   │   ├── Incident data          # Données d'incidents (2009-2024)
│   │   ├── Mobilisation data      # Données de mobilisation (2009-2024)
│   │   └── Metadata               # Métadonnées des fichiers
│   │
│   ├── 2_CSV/                     # Données converties en CSV
│   │   ├── incident_*.csv         # Fichiers d'incidents
│   │   └── mobilisation_*.csv     # Fichiers de mobilisation
│   │
│   ├── 3_external/                # Données externes
│   │   └── final_stations_list.csv    # Liste des casernes
│   │
│   └── 4_processed_CSV/           # Données traitées
│       └── df_modelisation.csv    # Dataset final pour la modélisation
│
├── logs/                          # Logs de l'application
│   ├── api.log                    # Logs de l'API
│   ├── model-XGB.log             # Logs de l'entraînement
│   └── preprocess.log            # Logs du preprocessing
│
├── models/                        # Modèles et encodeurs
│   ├── encoders.json             # Encodeurs au format JSON
│   ├── model-XGB.json            # Modèle XGBoost au format natif
│   └── *.pkl                     # Anciennes versions (déprécié)
│
├── requirements/                  # Dépendances Python
│   ├── base.txt                  # Dépendances communes
│   ├── api.txt                   # Dépendances API
│   └── frontend.txt              # Dépendances Frontend
│
├── src/                          # Code source
│   ├── api/                      # API FastAPI
│   │   ├── api.py               # Point d'entrée de l'API
│   │   ├── models.py            # Modèles Pydantic
│   │   └── Dockerfile           # Configuration Docker
│   │
│   ├── frontend/                 # Interface utilisateur
│   │   ├── streamlit_app.py     # Application Streamlit
│   │   └── Dockerfile           # Configuration Docker
│   │
│   ├── ml/                       # Machine Learning
│   │   ├── preprocess.py        # Préparation des données
│   │   ├── model-XGB.py         # Entraînement du modèle
│   │   └── predict.py           # Prédiction
│   │
│   └── utils/                    # Utilitaires
│       └── geo_utils.py         # Fonctions géographiques
│
├── docker-compose.yml            # Configuration des conteneurs
└── setup.py                      # Configuration du package Python
```

## Composants Principaux

### API (FastAPI)
- Point d'entrée des prédictions
- Gestion des requêtes HTTP
- Logging des prédictions

### Frontend (Streamlit)
- Interface utilisateur interactive
- Visualisation des prédictions
- Affichage cartographique

### Machine Learning
- Prétraitement des données
- Entraînement du modèle XGBoost
- Prédiction des temps d'intervention

