"""
---------------------------------------------------------------------------------------------------
Nom du script : model-XGB.py

Dernière mise à jour : samedi 25 janvier 2025

Fonction de ce script : création d'un modèle XGB pour la prédiction

Tâches réalisées par ce script :
 - Chargement des données
 - Création des jeux d'entraînement et de test
 - Entraînement du modèle XGB
 - Évaluation des performances
 - Sauvegarde du modèle au format natif XGBoost
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
import xgboost as xgb

"""
---------------------------------------------------------------------------------------------------
   TRAITEMENT - Capture du temps de début
   TRAITEMENT - Définition des chemins vers les fichiers utilisés
---------------------------------------------------------------------------------------------------
"""

# Capture le temps au début
start_time = time.time()

# logs
path_to_log = './logs/model-XGB.log'

# données de modélisation
path_to_CSV = "./data/4_processed_CSV/df_modelisation.csv"

# sauvegarde du modèle
path_to_model = "./models/model-XGB.json"  # Format natif XGBoost

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

# LOG -  Activer la capture des warnings
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

# Conversion en DMatrix pour XGBoost
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

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
    TRAITEMENT - Sauvegarde au format natif XGBoost
---------------------------------------------------------------------------------------------------
"""

# LOG - Construction du message à logger
message = ["",
           "",
           "TRAITEMENT - Entraînement du modèle XGB",
           "TRAITEMENT - Prédictions",
           "TRAITEMENT - Évaluation",
           "TRAITEMENT - Sauvegarde au format natif XGBoost",
           "...",
           ""
           ]
logger.info("\n".join(message))

# Paramètres du modèle
params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 7,
    'learning_rate': 0.2,
    'subsample': 0.8,
    'n_estimators': 200,
    'seed': 42
}

# TRAITEMENT - Entraînement du modèle XGB
num_rounds = 200
evallist = [(dtrain, 'train'), (dtest, 'eval')]

model = xgb.train(
    params,
    dtrain,
    num_rounds,
    evallist,
    early_stopping_rounds=10,
    verbose_eval=10
)

# TRAITEMENT - Prédictions
y_pred = model.predict(dtest)

# TRAITEMENT - Évaluation
variance_y_test = y_test.var()
mse_xgb = mean_squared_error(y_test, y_pred)
r2_xgb = r2_score(y_test, y_pred)

# TRAITEMENT - Sauvegarde au format natif XGBoost
model.save_model(path_to_model)

# LOG - Construction du message à logger
message = ["",
           "",
           f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
           "",
           f"Données - Variance des données de test [ Variance(y_test) ] = {variance_y_test}",
           "",
           f"XGBoost - Moyenne des moindres carrés (MSE) = {mse_xgb}",
           "",
           f"XGBoost - R^2 = [ 1 - ( MSE / Variance(y_test) ) ] = {r2_xgb}",
           "",
           "--------------------------------------------------------------------------------------------------------------------------",
           "",
           "                              Fin des traitements de model-XGB.py",
           "",
           "--------------------------------------------------------------------------------------------------------------------------",
           ""
           ]
logger.info("\n".join(message))