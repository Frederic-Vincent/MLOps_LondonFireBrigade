import time
import numpy as np
import pandas as pd
import pickle

from src.utils.geo_utils import address_to_lat_long, haversine

""""
---------------------------------------------------------------------------------------------------
    
    FONCTION : predict(address,
                        HourOfCall,
                        IncidentGroup,
                        PropertyCategory)
    
    Fonction qui prédit le temps d'arrivée du premier camion de pompiers

---------------------------------------------------------------------------------------------------
"""

def predict(address,
           HourOfCall,
           IncidentGroup,
           PropertyCategory):
    
    # Récupération du modèle à utiliser pour la prédiction
    with open('./models/model-XGB.pkl','rb') as f:
        xgb_model = pickle.load(f)

    # Récupération des encodeurs pour les variables non-numériques
    with open('./models/label_encoders.pkl', 'rb') as f:
        l_e_ = pickle.load(f)
    
    # Récupération des données géospatiales sur les stations de pompiers
    df_stations = pd.read_csv('./data/3_external/final_stations_list.csv')

    # Colonnes à utiliser 
    columns = ['HourOfCall_x', 
               'IncidentGroup', 
               'IncidentStationGround',
               'PropertyCategory', 
               'IncGeo_BoroughName', 
               'DeployedFromStation_Name',
               'IncidentLatitude', 'IncidentLongitude', 
               'StationLatitude', 'StationLongitude', 
               'DistanceToStation']

    # Récupération de la latitude et de la longitude du lieu de l'incident à partir de l'adresse postale
    latitude, longitude = address_to_lat_long(address)

    # Calcul des distances entre le lieu de l'incident et les stations de pompiers
    df_stations['DistanceToStation'] = df_stations.apply(
        lambda row: haversine(row['StationLatitude'], row['StationLongitude'], 
                              latitude, longitude
                             ),
        axis=1)

    # Identification de la station de pompiers qui est la plus proche du lieu de l'incident
    station = df_stations.iloc[df_stations['DistanceToStation'].idxmin()]

    # Construction des variables à utiliser pour la prédiction
    X_predict = pd.DataFrame(data = [[HourOfCall,
                                      int(l_e_['IncidentGroup'].transform([IncidentGroup])[0]),
                                      int(l_e_['IncidentStationGround'].transform([station['Station']])[0]),
                                      int(l_e_['PropertyCategory'].transform([PropertyCategory])[0]),
                                      int(l_e_['IncGeo_BoroughName'].transform([station['StationBorough']])[0]),
                                      int(l_e_['DeployedFromStation_Name'].transform([station['Station']])[0]),
                                      latitude, longitude,
                                      station["StationLatitude"], station["StationLongitude"],
                                      station["DistanceToStation"]]],
                             columns=columns)
    
    # Calcul de la prédiction
    return {"latitude": latitude,
            "longitude": longitude,
            "station": station['Station'],
            "StationBorough":station['StationBorough'],
            "StationLatitude": station["StationLatitude"],
            "StationLongitude": station["StationLongitude"],
            "prediction": float(xgb_model.predict(X_predict)[0])}
