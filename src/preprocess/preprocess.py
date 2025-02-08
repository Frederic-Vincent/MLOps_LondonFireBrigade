"""
---------------------------------------------------------------------------------------------------
Nom du script : preprocess.py

Dernière mise à jour : dimanche 26 janvier 2025

Fonction de ce script : préparation des données pour la modélisation et la prédiction

Tâches réalisées par ce script :
 - Chargement des données d'incidents
 - Concaténation des données d'incidents : df_incidents
 - Traitement des valeurs manquantes (FirstPumpArriving_AttendanceTime)
 - Chargement des données de mobilisation
 - Concaténation des données de mobilisation : df_mobilisation
 - Jointure des données d'incidents et de mobilisations (colonne 'IncidentNumber')  : df_incidents_mobilisations
 - Sélection de colonnes
 - Calcul de IncidentLatitude et IncidentLongitude à partir des données Easting_rounded et Northing_rounded
 - Suppression des colonnes 'Northing_rounded', 'Easting_rounded', 'Latitude', 'Longitude'
 - Suppression des lignes avec des valeurs manquantes
 - Chargement des données des stations : df_stations
 - Jointure de df_incidents_mobilisations avec df_stations
 - ... (left_on='DeployedFromStation_Name', right_on='Station')
 - ... df_modelisation
 - Suppression des colonnes 'Station', 'IncidentNumber'
 - Suppression des lignes avec des valeurs manquantes
 - Calcul de la distance entre l'incident et la station avec la fonction d'Haversine
 - Détection des colonnes de type 'object'
 - Encodage des colonnes de type 'object'
 - Calcul des quantiles Q1 et Q3, et de l'écart interquantile (IQR = Q3 - Q1)
 - Définition des limites (lower_bound, upper_bound) pour filtrer les outliers
 - Filtre de df_modelisation avec la condition : lower_bound <= AttendanceTimeSeconds <= upper_bound
 - Réinitialisation de l'index de df_modelisation
 - Sauvegarde de df_modelisation dans un fichier CSV

 Fonctions locales
  - haversine
  - preprocess
---------------------------------------------------------------------------------------------------
"""



""""
---------------------------------------------------------------------------------------------------
                            Import des bibliothèques
---------------------------------------------------------------------------------------------------
"""

import time
import logging
import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.preprocessing import LabelEncoder
import pickle

from src.utils.geo_utils import haversine



"""
---------------------------------------------------------------------------------------------------
TRAITEMENT - Définition des chemins vers les fichiers utilisés (à adapter en fonction de l'architecture locale)
---------------------------------------------------------------------------------------------------
"""

# données d'incidents
path_incident_1 = "./data/2_CSV/incident_2009_2017.csv"
path_incident_2 = "./data/2_CSV/incident_2018_2024_07.csv"

# TEST UNITAIRE - Le fichier incident_2018_2024_07_TEST contient un dataframe avec une colonne en moins
# TEST UNITAIRE - Décommenter la ligne suivante pour exécuter le test
# path_incident_2 = "./data/2_CSV/incident_2018_2024_07_TEST.csv"

# données de mobilisations
path_mobilisation_1 = "./data/2_CSV/mobilisation_2009_2014.csv"
path_mobilisation_2 = "./data/2_CSV/mobilisation_2015_2020.csv"
path_mobilisation_3 = "./data/2_CSV/mobilisation_2021_2024.csv"

# données des stations
path_to_stations="./data/3_external/final_stations_list.csv"

# logs
path_to_log = './logs/preprocess.log'

# sauvegardes
path_to_encoders = "./models/label_encoders.pkl"
path_to_CSV = "./data/4_processed_CSV/df_modelisation.csv"


""""
---------------------------------------------------------------------------------------------------
    
    FONCTION LOCALE : preprocess()
    
    - Chargement des données d'incidents
    - Concaténation des données d'incidents : df_incidents
    - Traitement des valeurs manquantes (FirstPumpArriving_AttendanceTime)
    - Chargement des données de mobilisation
    - Concaténation des données de mobilisation : df_mobilisation
    - Jointure des données d'incidents et de mobilisations (colonne 'IncidentNumber')  : df_incidents_mobilisations
    - Sélection de colonnes
    - Calcul de IncidentLatitude et IncidentLongitude à partir des données Easting_rounded et Northing_rounded
    - Suppression des colonnes 'Northing_rounded', 'Easting_rounded', 'Latitude', 'Longitude'
    - Suppression des lignes avec des valeurs manquantes
    - Chargement des données des stations : df_stations
    - Jointure de df_incidents_mobilisations avec df_stations
    - ... (left_on='DeployedFromStation_Name', right_on='Station')
    - ... df_modelisation
    - Suppression des colonnes 'Station', 'IncidentNumber'
    - Suppression des lignes avec des valeurs manquantes
    - Calcul de la distance entre l'incident et la station avec la fonction d'Haversine
    - Détection des colonnes de type 'object'
    - Encodage des colonnes de type 'object'
    - Calcul des quantiles Q1 et Q3, et de l'écart interquantile (IQR = Q3 - Q1)
    - Définition des limites (lower_bound, upper_bound) pour filtrer les outliers
    - Filtre de df_modelisation avec la condition : lower_bound <= AttendanceTimeSeconds <= upper_bound
    - Réinitialisation de l'index de df_modelisation
    - Sauvegarde de df_modelisation dans un fichier CSV

---------------------------------------------------------------------------------------------------
"""


def preprocess(path_incident_1, 
               path_incident_2, 
               path_mobilisation_1,
               path_mobilisation_2,
               path_mobilisation_3,
               path_to_stations,
               path_to_log,
               path_to_CSV
               ):
    
    # temps de début
    start_time = time.time()

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
               "                                        Début des traitements de preprocess.py",
               "",
               "--------------------------------------------------------------------------------------------------------------------------",
               ""
               ]
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Chargement des données d'incidents
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Chargement des données d'incidents :",
               f" - df_incidents_1 : {path_incident_1}",
               f" - df_incidents_2 : {path_incident_2}",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Chargement des données d'incidents
    df_incidents_1 = pd.read_csv(path_incident_1, low_memory=False)
    df_incidents_2 = pd.read_csv(path_incident_2, low_memory=False)

    # LOG - Construction du message à logger
    message = ["",
               "",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               f" - Taille de df_incidents_1: {df_incidents_1.shape}",
               f" - Taille de df_incidents_2: {df_incidents_2.shape}",
               "",
               "COLONNES de df_incidents_1 :"
               ]
    message.extend([f" - {col}" for col in df_incidents_1.columns])
    message.append("")
    message.append("--------------------------------------------------------------------------------------------------------------------------")
    message.append("")
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Concaténation des données d'incidents : df_incidents
        TRAITEMENT - Traitement des valeurs manquantes (FirstPumpArriving_AttendanceTime)
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Concaténation des données d'incidents : df_incidents",
               "TRAITEMENT - Traitement des valeurs manquantes (FirstPumpArriving_AttendanceTime)",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Concaténation
    # TEST UNITAIRE - Vérifier que df_incidents_1 et df_incidents_2 ont bien le même nombre de colonnes
    if (df_incidents_1.shape[1] == df_incidents_2.shape[1]):
        df_incidents = pd.concat([df_incidents_1, df_incidents_2])
    else:
        message = ["",
                   "",
                   "TEST UNITAIRE - df_incidents_1 et df_incidents_2 n'ont pas le même nombre de colonnes",
                   "TEST UNITAIRE - Arrêt du script",
                   "",
                   "--------------------------------------------------------------------------------------------------------------------------"
                   ]
        logger.info("\n".join(message))
        return

    # TRAITEMENT - Traitement des valeurs manquantes (FirstPumpArriving_AttendanceTime)
    df_incidents = df_incidents.dropna(subset=['FirstPumpArriving_AttendanceTime'])

    # LOG - Construction du message à logger
    message = ["",
               "",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               f" - Taille de df_incidents: {df_incidents.shape}",
               "",
               "--------------------------------------------------------------------------------------------------------------------------",
               ""
               ]
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Chargement des données de mobilisation
        TRAITEMENT - Concaténation des données de mobilisation : df_mobilisation
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Chargement des données de mobilisation :",
               f" - df_mobilisation_1 : {path_mobilisation_1}",
               f" - df_mobilisation_2 : {path_mobilisation_2}",
               f" - df_mobilisation_3 : {path_mobilisation_3}",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Chargement des données de mobilisation
    df_mobilisation_1 = pd.read_csv(path_mobilisation_1, low_memory=False)
    df_mobilisation_2 = pd.read_csv(path_mobilisation_2, low_memory=False)
    df_mobilisation_3 = pd.read_csv(path_mobilisation_3, low_memory=False)

    # TRAITEMENT - Concaténation des données de mobilisation : df_mobilisation
    df_mobilisation = pd.concat([df_mobilisation_1, df_mobilisation_2, df_mobilisation_3])


    # LOG - Construction du message à logger
    message = ["",
               "",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               f" - Taille de df_mobilisation_1: {df_mobilisation_1.shape}",
               f" - Taille de df_mobilisation_2: {df_mobilisation_2.shape}",
               f" - Taille de df_mobilisation_3: {df_mobilisation_3.shape}",
               "",
               f" - Taille de df_mobilisation: {df_mobilisation.shape}",
               "",
               "COLONNES de df_mobilisation :"
               ]
    message.extend([f" - {col}" for col in df_mobilisation.columns])
    message.append("")
    message.append("--------------------------------------------------------------------------------------------------------------------------")
    message.append("")
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Jointure des données d'incidents et de mobilisations (colonne 'IncidentNumber') : df_incidents_mobilisations
        TRAITEMENT - Sélection de colonnes
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Jointure des données d'incidents et de mobilisations (colonne 'IncidentNumber')  : df_incidents_mobilisations",
               "TRAITEMENT - Sélection de colonnes",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Jointure des données d'incidents et de mobilisations (colonne 'IncidentNumber')
    df_incidents_mobilisations = pd.merge(df_incidents, df_mobilisation, how='inner', on='IncidentNumber')

    # TRAITEMENT - Sélection de colonnes
    columns_to_keep = ['IncidentNumber',
                       'HourOfCall_x', 'IncidentGroup',
                       'IncidentStationGround', 'PropertyCategory', 'Northing_rounded',
                       'Easting_rounded', 'IncGeo_BoroughName','Latitude','Longitude',
                       'DeployedFromStation_Name','AttendanceTimeSeconds'
                       ]

    # TRAITEMENT - Sélection de colonnes
    df_incidents_mobilisations = df_incidents_mobilisations[columns_to_keep]

    # LOG - Construction du message à logger
    message = ["",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               "Colonnes sélectionnées:"
               ]
    message.extend([f" - {col}" for col in columns_to_keep])
    message.append("")
    message.append("--------------------------------------------------------------------------------------------------------------------------")
    message.append("")
    logger.info("\n".join(message))

    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Calcul de IncidentLatitude et IncidentLongitude à partir des données Easting_rounded et Northing_rounded
        TRAITEMENT - Suppression des colonnes 'Northing_rounded', 'Easting_rounded', 'Latitude', 'Longitude'
        TRAITEMENT - Suppression des lignes avec des valeurs manquantes
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Calcul de IncidentLatitude et IncidentLongitude à partir des données Easting_rounded et Northing_rounded",
               "TRAITEMENT - Suppression des colonnes 'Northing_rounded', 'Easting_rounded', 'Latitude', 'Longitude'",
               "TRAITEMENT - Suppression des lignes avec des valeurs manquantes",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Création d'un transformateur pour la conversion ...
    # TRAITEMENT - ... (Easting_rounded, Northing_rounded) > (Latitude, Longitude)
    transformer = Transformer.from_crs("epsg:27700", "epsg:4326")

    # TRAITEMENT - Calcul de IncidentLatitude et IncidentLongitude
    df_incidents_mobilisations['IncidentLatitude'], df_incidents_mobilisations['IncidentLongitude'] = zip(
        *df_incidents_mobilisations.apply(lambda row: transformer.transform(row['Easting_rounded'], 
                                                                            row['Northing_rounded']),
                                        axis=1))

    # TRAITEMENT - Suppression des colonnes 'Northing_rounded', 'Easting_rounded', 'Latitude', 'Longitude'
    df_incidents_mobilisations = df_incidents_mobilisations.drop(columns=['Northing_rounded', 'Easting_rounded', 
                                                                        'Latitude', 'Longitude'])

    # TRAITEMENT - Suppression des lignes avec des valeurs manquantes
    df_incidents_mobilisations = df_incidents_mobilisations.dropna()

    # LOG - Construction du message à logger
    message = ["",
               "",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               f"Taille de df_incidents_mobilisations : {df_incidents_mobilisations.shape}",
               "",
               "Colonnes de df_incidents_mobilisations:"
               ]
    message.extend([f" - {col}" for col in df_incidents_mobilisations.columns])
    message.append("")
    message.append("--------------------------------------------------------------------------------------------------------------------------")
    message.append("")
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Chargement des données des stations : df_stations
        TRAITEMENT - Jointure de df_incidents_mobilisations avec df_stations
        TRAITEMENT - ... (left_on='DeployedFromStation_Name', right_on='Station')
        TRAITEMENT - ... df_modelisation
        TRAITEMENT - Suppression des colonnes 'Station', 'IncidentNumber'
        TRAITEMENT - Suppression des lignes avec des valeurs manquantes
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Chargement des données des stations : df_stations",
               "TRAITEMENT - Jointure de df_incidents_mobilisations avec df_stations",
               "TRAITEMENT - ... (left_on='DeployedFromStation_Name', right_on='Station')",
               "TRAITEMENT - ... df_modelisation",
               "TRAITEMENT - Suppression des colonnes 'Station', 'IncidentNumber'",
               "TRAITEMENT - Suppression des lignes avec des valeurs manquantes",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Chargement des données des stations : df_stations
    df_stations = pd.read_csv(path_to_stations)

    # TRAITEMENT - Jointure de df_incidents_mobilisations avec df_stations 
    # TRAITEMENT - ... (left_on='DeployedFromStation_Name', right_on='Station')
    df_modelisation = pd.merge(df_incidents_mobilisations, 
                            df_stations[['Station', 'StationLatitude', 'StationLongitude']], 
                            how='left',
                            left_on='DeployedFromStation_Name',
                            right_on='Station'
                            )

    # TRAITEMENT - Suppression des colonnes 'Station', 'IncidentNumber'
    df_modelisation = df_modelisation.drop(columns=['Station', 'IncidentNumber'])

    # TRAITEMENT - Suppression des lignes avec des valeurs manquantes
    df_modelisation = df_modelisation.dropna()

    # LOG - Construction du message à logger
    message = ["",
               "",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               f"Taille de df_modelisation : {df_modelisation.shape}",
               "",
               "Colonnes de df_modelisation : "
               ]
    message.extend([f" - {col}" for col in df_modelisation.columns])
    message.append("")
    message.append("--------------------------------------------------------------------------------------------------------------------------")
    message.append("")
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Calcul de la distance entre l'incident et la station avec la fonction d'Haversine
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Calcul de la distance entre l'incident et la station avec la fonction Haversine",
               "..."
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Calcul de la distance entre l'incident et la station avec la fonction Haversine
    df_modelisation['DistanceToStation'] = df_modelisation.apply(
        lambda row: haversine(row['IncidentLatitude'], row['IncidentLongitude'], 
                            row['StationLatitude'], row['StationLongitude']
                            ),
        axis=1
    )

    # LOG - Construction du message à logger
    message = ["",
               "",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               f"Taille de df_modelisation : {df_modelisation.shape}",
               "",
               "--------------------------------------------------------------------------------------------------------------------------",
               ""
               ]
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Détection des colonnes de type 'object'
        TRAITEMENT - Encodage des colonnes de type 'object' 
    --------------------------------------------------------------------------------------------------------------------------
    """

    # TRAITEMENT - Détection des colonnes de type 'object'
    non_numeric_cols = df_modelisation.select_dtypes(include=['object']).columns

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Détection des colonnes de type 'object'",
               "",
               "Colonnes de type 'object' :"
               ]
    message.extend([f" - {col}" for col in non_numeric_cols])
    message.append("")
    message.append("TRAITEMENT - Encodage des colonnes de type 'object'")
    message.append(f"TRAITEMENT - Sauvegarde des encodeurs dans {path_to_encoders}")
    message.append("")
    logger.info("\n".join(message))

    # TRAITEMENT - Encodage des colonnes de type 'object'
    label_encoders = {}
    for col in non_numeric_cols:
        le = LabelEncoder()
        df_modelisation[col] = le.fit_transform(df_modelisation[col])
        label_encoders[col] = le

    # TRAITEMENT - Sauvegarde des encodeurs
    with open(path_to_encoders, 'wb') as f:
        pickle.dump(label_encoders, f)

    # LOG - Construction du message à logger
    message = ["",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               f"Taille de df_modelisation : {df_modelisation.shape}",
               "",
               "Colonnes de df_modelisation : "
               ]
    message.extend([f" - {col}" for col in df_modelisation.columns])
    message.append("")
    message.append("--------------------------------------------------------------------------------------------------------------------------")
    message.append("")
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Calcul des quantiles Q1 et Q3, et de l'écart interquantile (IQR = Q3 - Q1)
        TRAITEMENT - Définition des limites (lower_bound, upper_bound) pour filtrer les outliers
        TRAITEMENT - Filtre de df_modelisation avec la condition : lower_bound <= AttendanceTimeSeconds <= upper_bound
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Calcul des quantiles Q1 et Q3, et de l'écart interquantile (IQR = Q3 - Q1)",
               "TRAITEMENT - Définition des limites (lower_bound, upper_bound) pour filtrer les outliers",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Calcul des quantiles Q1 et Q3, et de l'écart interquantile (IQR = Q3 - Q1)
    Q1 = df_modelisation['AttendanceTimeSeconds'].quantile(0.25)
    Q3 = df_modelisation['AttendanceTimeSeconds'].quantile(0.75)
    IQR = Q3 - Q1

    # TRAITEMENT - Définition des bornes (lower_bound, upper_bound) pour filtrer les outliers
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # LOG - Construction du message à logger
    message = ["",
               "",
               f"Quantile Q1 = {Q1}",
               f"Quantile Q3 = {Q3}",
               f"Interquantile IQR = {IQR}",
               f"lower_bound = Q1 - 1.5 * IQR = {lower_bound}",
               "",
               "TRAITEMENT - Filtre de df_modelisation avec la condition : lower_bound <= AttendanceTimeSeconds <= upper_bound",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Filtre de df_modelisation avec la condition : lower_bound <= AttendanceTimeSeconds <= upper_bound
    index = (df_modelisation['AttendanceTimeSeconds'] >= lower_bound) & (df_modelisation['AttendanceTimeSeconds'] <= upper_bound)
    df_modelisation = df_modelisation[index]

    # LOG - Construction du message à logger
    message = ["",
               "",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               f"Taille de df_modelisation : {df_modelisation.shape}",
               "",
               "Colonnes de df_modelisation : "
               ]
    message.extend([f" - {col}" for col in df_modelisation.columns])
    message.append("")
    message.append("--------------------------------------------------------------------------------------------------------------------------")
    message.append("")
    logger.info("\n".join(message))


    """"
    --------------------------------------------------------------------------------------------------------------------------
        TRAITEMENT - Réinitialisation de l'index de df_modelisation
        TRAITEMENT - Sauvegarde de df_modelisation dans un fichier CSV
    --------------------------------------------------------------------------------------------------------------------------
    """

    # LOG - Construction du message à logger
    message = ["",
               "",
               "TRAITEMENT - Réinitialisation de l'index de df_modelisation",
               f"TRAITEMENT - Sauvegarde de df_modelisation dans un fichier CSV : {path_to_CSV}",
               "...",
               ""
               ]
    logger.info("\n".join(message))

    # TRAITEMENT - Réinitialisation de l'index de df_modelisation
    df_modelisation = df_modelisation.reset_index(drop=True)

    # TRAITEMENT - Sauvegarde de df_modelisation dans un fichier CSV
    df_modelisation.to_csv(path_to_CSV)

    # LOG - Construction du message à logger
    message = ["",
               "",
               f"C'est fait. Temps écoulé depuis le lancement : environ {round(time.time() - start_time)} secondes",
               "",
               "--------------------------------------------------------------------------------------------------------------------------",
               "",
               "                              Fin des traitements de preprocess.py",
               "",
               "--------------------------------------------------------------------------------------------------------------------------",
               ""
               ]
    logger.info("\n".join(message))


if __name__ == "__main__":

    # appel de la fonction preprocess
    preprocess(path_incident_1, 
               path_incident_2, 
               path_mobilisation_1,
               path_mobilisation_2,
               path_mobilisation_3,
               path_to_stations,
               path_to_log,
               path_to_CSV
               )
