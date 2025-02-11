
"""
---------------------------------------------------------------------------------------------------
Nom du script : model-XGB.py

Dernière mise à jour : samedi 25 janvier 2025

Fonction de ce script : création d'un modèle XGB pour la prédiction

Tâches réalisées par ce script :
 - 
---------------------------------------------------------------------------------------------------
"""

""""
---------------------------------------------------------------------------------------------------
                            Import des bibliothèques
---------------------------------------------------------------------------------------------------
"""
import time
import datetime
from datetime import datetime
import logging
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor
import pickle




"""
---------------------------------------------------------------------------------------------------
   TRAITEMENT - Capture du temps de début
   TRAITEMENT - Définition des chemins vers les fichiers utilisés (à adapter en fonction de l'architecture locale)
---------------------------------------------------------------------------------------------------
"""

# Capture le temps au début
start_time = time.time()

# logs
path_to_log = './logs/model-XGB.log'

# données de modélisation
path_to_CSV = "./data/4_processed_CSV/df_modelisation.csv"

# sauvegarde du modèle
path_to_model = "./models/model-XGB.pkl"

""""
---------------------------------------------------------------------------------------------------
    LOG - Configuration de la journalisation pour l'écriture de logs
---------------------------------------------------------------------------------------------------
"""

# LOG -  Configurer globalement la journalisation avec basicConfig
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.FileHandler(path_to_log),
                        logging.StreamHandler()
                    ])

# LOG -  Obtenir le logger principal
logger = logging.getLogger(__name__)

# LOG -  Activer la capture des warnings (cela doit être fait après la configuration de logging)
logging.captureWarnings(True)

# LOG - Construction du message à logger
message = ["",
           "",
           "--------------------------------------------------------------------------------------------------------------------------",
           "",
           "                                        Début des traitements de model-XGB.py",
           "",
           "--------------------------------------------------------------------------------------------------------------------------",
           ""
           ]
logger.info("\n".join(message))

""""
---------------------------------------------------------------------------------------------------
    TRAITEMENT - Chargement des données de modélisation (df_modelisation)
---------------------------------------------------------------------------------------------------
"""

# LOG - Construction du message à logger
message = ["",
           "",
           f"TRAITEMENT - Chargement des données de modélisation (df_modelisation) : {path_to_CSV}",
           "...",
           ""
           ]
logger.info("\n".join(message))

# TRAITEMENT - Chargement des données de modélisation
df_modelisation = pd.read_csv(path_to_CSV, index_col=0)

# LOG - Construction du message à logger
message = ["",
           "",
           f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
           "",
           f" - Taille de df_modelisation: {df_modelisation.shape}",
           "",
           "COLONNES de df_modelisation :"
           ]
message.extend([f" - {col}" for col in df_modelisation.columns])
message.append("")
message.append("--------------------------------------------------------------------------------------------------------------------------")
message.append("")
logger.info("\n".join(message))


""""
---------------------------------------------------------------------------------------------------
    TRAITEMENT - Création des variables X (données) et Y (cible)
    TRAITEMENT - Création des jeux d'entraînement et de test
---------------------------------------------------------------------------------------------------
"""

# LOG - Construction du message à logger
message = ["",
           "",
           "TRAITEMENT - Création des variables X (données) et Y (cible)",
           "TRAITEMENT - Création des jeux d'entraînement et de test",
           "...",
           ""
           ]
logger.info("\n".join(message))

# TRAITEMENT - Création des variables X (données) et Y (cible)
X = df_modelisation.drop(columns=['AttendanceTimeSeconds'])
y = df_modelisation['AttendanceTimeSeconds'] 

# TRAITEMENT - Création des jeux d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# LOG - Construction du message à logger
message = ["",
           "",
           f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
           "",
           "--------------------------------------------------------------------------------------------------------------------------",
           ""
           ]
logger.info("\n".join(message))


""""
---------------------------------------------------------------------------------------------------
    TRAITEMENT - Entraînement du modèle XGB
    TRAITEMENT - Prédictions
    TRAITEMENT - Évaluation
    TRAITEMENT - Sauvegarde dans un fichier pickle
---------------------------------------------------------------------------------------------------
"""

# LOG - Construction du message à logger
message = ["",
           "",
           "TRAITEMENT - Entraînement du modèle XGB",
           "TRAITEMENT - Prédictions",
           "TRAITEMENT - Évaluation",
           "TRAITEMENT - Sauvegarde dans un fichier pickle",
           "...",
           ""
           ]
logger.info("\n".join(message))

# TRAITEMENT - Entraînement du modèle XGB
xgb_model = XGBRegressor(subsample=0.8, n_estimators=200, max_depth=7, learning_rate=0.2, random_state=42)
xgb_model.fit(X_train, y_train)

# TRAITEMENT - Prédictions
y_pred = xgb_model.predict(X_test)

# TRAITEMENT - Évaluation
mse_xgb = mean_squared_error(y_test, y_pred)
r2_xgb = r2_score(y_test, y_pred)

# TRAITEMENT - Sauvegarde dans un fichier pickle
pickle.dump(xgb_model, open(path_to_model, 'wb'))

# LOG - Construction du message à logger
message = ["",
           "",
           f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
           "",
           f"Tuned XGBoost MSE: {mse_xgb}",
           "",
           f"Tuned XGBoost R^2: {r2_xgb}",
           "",
           "--------------------------------------------------------------------------------------------------------------------------",
           "",
           "                              Fin des traitements de model-XGB.py",
           "",
           "--------------------------------------------------------------------------------------------------------------------------",
           ""
           ]
logger.info("\n".join(message))