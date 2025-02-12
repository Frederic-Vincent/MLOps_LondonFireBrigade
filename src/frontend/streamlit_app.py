import os
import streamlit as st
import requests
import folium
from streamlit_folium import folium_static
from streamlit_mermaid import st_mermaid

# Titre principal de l'application
st.title("Prédiction du temps d'intervention des pompiers de Londres")

# Menu de navigation étendu
page = st.sidebar.radio(
    "Navigation",
    ["Contexte", "Exploration", "Modélisation", "Architecture", "Prédiction", "Logs", "En cours de développement"]
)

def show_context():
    """Affiche la section Contexte."""
    st.subheader("Contexte du Projet")
    
    st.markdown("""
    ### London Fire Brigade
    La London Fire Brigade (LFB) est le service d'incendie et de secours de Londres. C'est l'un des plus grands services 
    d'incendie et de secours au monde avec environ :
    - 5 500 pompiers opérationnels et officiers
    - 100 stations de pompiers
    - 6 000 appels d'urgence par mois
    
    ### Objectif du Projet
    Développer un modèle de Machine Learning pour prédire le temps d'intervention des pompiers de Londres en fonction de :
    - La localisation de l'incident
    - L'heure de l'appel
    - Le type d'incident
    - Le type de propriété
    """)

def show_exploration():
    """Affiche la section Exploration."""
    st.subheader("Exploration des Données")
    
    st.markdown("""
    ### Sources de Données
    Les données proviennent de trois sources principales :
    1. **Incidents (2009-2024)**
       - Informations sur chaque incident
       - Localisation, type, date et heure
       
    2. **Mobilisations (2009-2024)**
       - Détails des interventions
       - Temps de réponse, casernes mobilisées
       
    3. **Stations**
       - Informations sur les casernes
       - Localisation, zone de couverture
    
    ### Preprocessing
    Les principales étapes de préparation :
    1. Nettoyage des données manquantes
    2. Conversion des coordonnées géographiques
    3. Calcul des distances entre incidents et stations
    4. Encodage des variables catégorielles
    5. Filtrage des valeurs aberrantes
    """)
    
    # Ici on pourrait ajouter des visualisations

def show_modeling():
    """Affiche la section Modélisation."""
    st.subheader("Modélisation")
    
    st.markdown("""
    ### Choix du Modèle
    Le modèle XGBoost a été sélectionné pour :
    - Sa performance sur les données tabulaires
    - Sa capacité à gérer les relations non-linéaires
    - Sa robustesse aux valeurs aberrantes
    
    ### Variables Utilisées
    Le modèle prend en compte :
    - Heure de l'appel
    - Type d'incident
    - Type de propriété
    - Localisation de l'incident
    - Distance à la caserne la plus proche
    
    ### Métriques de Performance
    - MSE (Mean Squared Error)
    - R² Score
    - Distribution des erreurs
    """)
    
    # Ici on pourrait ajouter des graphiques de performance

def show_architecture():
    """Affiche le diagramme d'architecture du projet."""
    st.subheader("Architecture du Projet")
    
    # Création du diagramme Mermaid
    mermaid_diagram = """
    graph TB
        subgraph Docker Environment
            subgraph Frontend Container
                A[Streamlit App<br>Port 8501] --> B[Requests]
            end
            
            subgraph API Container
                C[FastAPI<br>Port 8000] --> D[Prediction Module]
                D --> E[XGBoost Model]
                D --> F[Label Encoders]
                D --> G[Stations Data]
            end
            
            B --> |HTTP POST<br>/predict| C
        end
        
        subgraph Data Pipeline
            K[Data Processing<br>preprocess.py] --> E
            K --> F
            K --> G
        end
        
        subgraph Shared Volumes
            H[(data/)] -.-> G
            H -.-> K
            I[(models/)] -.-> E
            I -.-> F
            J[(logs/)] -.-> |api.log| C
            J -.-> |preprocess.log| K
        end
        
        style Frontend Container fill:#f9f,stroke:#333,stroke-width:2px
        style API Container fill:#bbf,stroke:#333,stroke-width:2px
        style Shared Volumes fill:#bfb,stroke:#333,stroke-width:2px
        style Data Pipeline fill:#ffb,stroke:#333,stroke-width:2px
    """
    
    # Affichage du diagramme avec st_mermaid
    st_mermaid(mermaid_diagram)
    
    # Description de l'architecture
    st.markdown("""
    #### Description de l'Architecture
    
    Le projet est composé de deux conteneurs Docker principaux :
    
    1. **Frontend (Streamlit)**
       - Interface utilisateur interactive
       - Port 8501
       - Envoie les requêtes de prédiction à l'API
    
    2. **API (FastAPI)**
       - Gère les requêtes de prédiction
       - Port 8000
       - Utilise le modèle XGBoost pour les prédictions
       - Accède aux données des stations
    
    #### Volumes Partagés
    
    Les conteneurs partagent trois volumes principaux :
    - **/data** : Données des stations
    - **/models** : Modèle XGBoost et encodeurs
    - **/logs** : Fichiers de logs
    """)

def show_prediction():
    # Formulaire de saisie
    address = st.text_input("Adresse", "Big Ben, London")
    hour = st.number_input("Heure d'appel", min_value=0, max_value=23, value=10)
    incident_group = st.selectbox("Type d'incident", ["Fire", "Special Service", "False Alarm"])
    property_category = st.selectbox("Type de propriété", ["Dwelling", "Other Residential", "Office"])


    # Bouton de prédiction

    if st.button("Prédire le temps d'intervention"):
        
        # Récupérer l'URL de l'API depuis les variables d'environnement
        # # avec une valeur par défaut pour le développement local
        api_url = os.getenv('API_URL', 'http://127.0.0.1:8000')
        url = f"{api_url}/predict"

        # Préparer la requête        
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
                st.info(f"📏 Distance : {results['DistanceToStation'] / 1000:.3f} km")
                
            with col2:
                # Créer la carte
                m = folium.Map(location=[results['latitude'], results['longitude']], zoom_start=13)
                
                # Marquer l'incident
                incident_popup = f"""
Lieu de l'incident

Adresse : {address}
Latitude : {results['latitude']}
Longitude : {results['longitude']}
"""
                folium.Marker(
                    [results['latitude'], results['longitude']],
                    popup=incident_popup,
                    icon=folium.Icon(color='red', icon='info-sign')
                ).add_to(m)

                # Marquer la caserne
                station_popup = f"""
Caserne {results['station']}

Latitude : {results['StationLatitude']}
Longitude : {results['StationLongitude']}

Distance : {results['DistanceToStation'] / 1000:.3f} km
"""
                folium.Marker(
                    [results['StationLatitude'], results['StationLongitude']],
                    popup=station_popup,
                    icon=folium.Icon(color='blue', icon='info-sign')
                ).add_to(m)

                # Tracer une ligne entre l'incident et la caserne
                distance_popup = f"""
Distance : {results['DistanceToStation'] / 1000:.3f} km
"""
                folium.PolyLine(
                    locations=[
                        [results['latitude'], results['longitude']],
                        [results['StationLatitude'], results['StationLongitude']]
                    ],
                    weight=2,
                    color='red',
                    dash_array='10',
                    popup=distance_popup
                ).add_to(m)
                
                # Afficher la carte
                folium_static(m)
                
        except requests.exceptions.RequestException as e:
            st.error(f"Erreur lors de la requête : {str(e)}")

def show_logs():
    """Affiche la section Logs."""
    st.subheader("Logs du Système")
    
    st.markdown("""
    ### Types de Logs
    Le système maintient plusieurs fichiers de logs :
    1. **Logs API**
       - Requêtes reçues
       - Temps de traitement
       - Erreurs éventuelles
       
    2. **Logs Preprocessing**
       - Étapes de traitement des données
       - Statistiques sur les données
       - Alertes et erreurs
       
    3. **Logs Prédiction**
       - Détails des prédictions
       - Performance du modèle
       - Temps de réponse
    """)
    
    # Option pour afficher les derniers logs
    if st.checkbox("Afficher les derniers logs"):
        try:
            with open("logs/api.log", "r") as f:
                last_logs = f.readlines()[-50:]  # 50 dernières lignes
                for log in last_logs:
                    st.text(log.strip())
        except Exception as e:
            st.error(f"Erreur lors de la lecture des logs : {str(e)}")

def show_development():
    """Affiche la section En cours de développement."""
    st.subheader("En cours de développement")
    
    st.markdown("""
    ### Fonctionnalités à venir
    
    1. **Sécurité**
       - Authentification des utilisateurs
       - Gestion des rôles et permissions
       - Historique des connexions
    
    2. **Amélioration du Modèle**
       - Intégration des conditions météorologiques
       - Prise en compte du trafic routier
       - Mise à jour automatique du modèle
    
    3. **Interface**
       - Dashboard de monitoring
       - Visualisations avancées
       - Export des résultats
    
    4. **Infrastructure**
       - Mise en place de tests automatisés
       - Monitoring des performances
       - Gestion des sauvegardes
    """)



# Affichage de la section sélectionnée
if page == "Contexte":
    show_context()
elif page == "Exploration":
    show_exploration()
elif page == "Modélisation":
    show_modeling()
elif page == "Architecture":
    show_architecture()
elif page == "Prédiction":
    show_prediction()
elif page == "Logs":
    show_logs()
else:  # En cours de développement
    show_development()

