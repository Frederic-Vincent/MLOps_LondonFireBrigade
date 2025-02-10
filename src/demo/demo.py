import streamlit as st
import requests
import folium
from streamlit_folium import folium_static

st.title("Prédiction du temps d'intervention des pompiers de Londres")

# Formulaire de saisie
address = st.text_input("Adresse", "Big Ben, London")
hour = st.number_input("Heure d'appel", min_value=0, max_value=23, value=10)
incident_group = st.selectbox("Type d'incident", ["Fire", "Special Service", "False Alarm"])
property_category = st.selectbox("Type de propriété", ["Dwelling", "Other Residential", "Office"])

# Bouton de prédiction
if st.button("Prédire le temps d'intervention"):
   # Préparer la requête
   url = "http://127.0.0.1:8000/predict"
   headers = {"Content-Type": "application/json"}
   data = {
       "address": address,
       "HourOfCall": hour,
       "IncidentGroup": incident_group,
       "PropertyCategory": property_category
   }
   
   try:
       # Envoyer la requête
       response = requests.post(url, headers=headers, json=data)
       response.raise_for_status()
       
       # Récupérer les résultats
       results = response.json()
       
       # Afficher les résultats
       col1, col2 = st.columns(2)
       
       with col1:
           st.success(f"⏱️ Temps d'intervention estimé : {results['prediction']:.1f} secondes")
           st.info(f"🚒 Caserne : {results['station']}")
           st.info(f"📍 Arrondissement : {results['StationBorough']}")
           
       with col2:
           # Créer la carte
           m = folium.Map(location=[results['latitude'], results['longitude']], zoom_start=13)
           
           # Marquer l'incident
           folium.Marker(
               [results['latitude'], results['longitude']],
               popup="Lieu de l'incident",
               icon=folium.Icon(color='red', icon='info-sign')
           ).add_to(m)

           # Marquer la caserne
           folium.Marker(
               [results['StationLatitude'], results['StationLongitude']],
               popup=f"Caserne {results['station']}",
               icon=folium.Icon(color='blue', icon='info-sign')
           ).add_to(m)

           # Tracer une ligne entre l'incident et la caserne
           folium.PolyLine(
               locations=[
                   [results['latitude'], results['longitude']],
                   [results['StationLatitude'], results['StationLongitude']]
               ],
               weight=2,
               color='red',
               dash_array='10'
           ).add_to(m)
           
           # Afficher la carte
           folium_static(m)
           
   except requests.exceptions.RequestException as e:
       st.error(f"Erreur lors de la requête : {str(e)}")