import numpy as np
from geopy.geocoders import Nominatim

""""
---------------------------------------------------------------------------------------------------
    
    FONCTION : haversine(lat1, lon1, lat2, lon2)
    
    Fonction qui implémente la formule d'Haversine pour calculer la distance entre deux points,
    sur la Terre, le long du grand cercle, à partir de leurs latitudes et leurs longitudes.
    Le résultat est une distance exprimée en mètres

---------------------------------------------------------------------------------------------------
"""

def haversine(lat1, lon1, lat2, lon2):

    # Radius of Earth in meters
    R = 6378137.0
    
    # Convert degrees to radians
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    
    # Differences in latitudes and longitudes
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    
    # Haversine formula
    a = np.sin(dlat / 2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    # Distance in meters
    distance = R * c
    
    return distance


""""
---------------------------------------------------------------------------------------------------
    
    FONCTION : address_to_lat_long(address)
    
    Fonction qui donne la latitude et la longitude à partir d'une addresse postale.

    Exemple : 
    
        latitude, longitude = address_to_lat_long("Big Ben, London")
        print(latitude, longitude)

        51.5007042 -0.1245721

---------------------------------------------------------------------------------------------------
"""

def address_to_lat_long(address):

    # Initialiser le géocodeur
    geolocator = Nominatim(user_agent="ML_Ops_LondonFireBrigade")

    # Géocodage
    location = geolocator.geocode(address)

    return location.latitude, location.longitude