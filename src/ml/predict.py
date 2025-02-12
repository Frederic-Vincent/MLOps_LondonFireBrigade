import time
import json
import numpy as np
import pandas as pd
import xgboost as xgb

from src.utils.geo_utils import address_to_lat_long, haversine

def predict(address,
           HourOfCall,
           IncidentGroup,
           PropertyCategory):
    
    # Récupération du modèle XGBoost
    xgb_model = xgb.Booster()
    xgb_model.load_model('./models/model-XGB.json')

    # Récupération des encodeurs depuis le JSON
    with open('./models/encoders.json', 'r') as f:
        encoders = json.load(f)
        
    # Création des encodeurs à partir des classes stockées
    l_e_ = {
        category: {value: idx for idx, value in enumerate(values)}
        for category, values in encoders.items()
    }
    
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

    # Récupération de la latitude et de la longitude du lieu de l'incident
    latitude, longitude = address_to_lat_long(address)

    # Calcul des distances entre le lieu de l'incident et les stations
    df_stations['DistanceToStation'] = df_stations.apply(
        lambda row: haversine(row['StationLatitude'], row['StationLongitude'], 
                            latitude, longitude),
        axis=1)

    # Identification de la station la plus proche
    station = df_stations.iloc[df_stations['DistanceToStation'].idxmin()]

    # Construction des variables pour la prédiction
    incident_encoded = l_e_['IncidentGroup'].get(IncidentGroup, 0)
    station_encoded = l_e_['IncidentStationGround'].get(station['Station'], 0)
    property_encoded = l_e_['PropertyCategory'].get(PropertyCategory, 0)
    borough_encoded = l_e_['IncGeo_BoroughName'].get(station['StationBorough'], 0)
    deployed_encoded = l_e_['DeployedFromStation_Name'].get(station['Station'], 0)

    X_predict = pd.DataFrame(data=[[
        HourOfCall,
        incident_encoded,
        station_encoded,
        property_encoded,
        borough_encoded,
        deployed_encoded,
        latitude, longitude,
        station["StationLatitude"], station["StationLongitude"],
        station["DistanceToStation"]
    ]], columns=columns)
    
    # Conversion en DMatrix pour XGBoost
    dmatrix = xgb.DMatrix(X_predict)
    
    # Calcul de la prédiction
    prediction = float(xgb_model.predict(dmatrix)[0])
    
    return {
        "latitude": latitude,
        "longitude": longitude,
        "station": station['Station'],
        "StationBorough": station['StationBorough'],
        "StationLatitude": station["StationLatitude"],
        "StationLongitude": station["StationLongitude"],
        "DistanceToStation": station["DistanceToStation"],
        "prediction": prediction
    }