import streamlit as st
import pandas as pd
import joblib  # For loading the model and scaler
from xgboost import Booster

# Load the XGBoost model and scaler
xgb_model = Booster()
xgb_model.load_model('xgb_model_booster.json')  # Updated to load the JSON model
scaler = joblib.load('scaler.pkl')

# Title of the application
st.set_page_config(page_title="Prédiction de l'Attrition Client", page_icon="🔍")
st.title("Prédiction de l'Attrition Client")

# File uploader for the dataset
uploaded_file = st.file_uploader("Choose the E-Commerce Dataset CSV file", type="csv")
if uploaded_file is not None:
    df_renamed = pd.read_csv(uploaded_file)

    # Load encoders for categorical columns
    encoders = {}
    categorical_columns = ['Genre']

    for column in categorical_columns:
        encoders[column] = joblib.load(f'{column}_encoder.pkl')

    # Create inputs for client features
    with st.container():
        anciennete_client = st.number_input("Ancienneté Client (en mois)", min_value=0, key="anciennete_client")
        categorie_ville = st.selectbox("Catégorie Ville", options=df_renamed['Categorie_Ville'].unique(), key="categorie_ville")
        genre = st.selectbox("Genre", options=df_renamed['Genre'].unique(), key="genre")
        heures_passees_sur_app = st.number_input("Heures Passées Sur App", min_value=0, key="heures_passees_sur_app")
        nombre_appareils_enregistres = st.number_input("Nombre d'Appareils Enregistrés", min_value=1, key="nombre_appareils_enregistres")
        nombre_adresses_enregistrees = st.number_input("Nombre d'Adresses Enregistrées", min_value=1, key="nombre_adresses_enregistrees")
        jours_depuis_derniere_commande = st.number_input("Jours Depuis Dernière Commande", min_value=0, key="jours_depuis_derniere_commande")
        nombre_commandes_mois_prec = st.number_input("Nombre de Commandes Mois Précédent", min_value=0, key="nombre_commandes_mois_prec")
        augmentation_commandes_annee_prec = st.number_input("Augmentation Commandes Année Précédente", min_value=0, key="augmentation_commandes_annee_prec")
        montant_cashback_moyen = st.number_input("Montant Cashback Moyen", min_value=0.0, key="montant_cashback_moyen")
        coupons_utilises = st.number_input("Coupons Utilisés", min_value=0, key="coupons_utilises")
        score_satisfaction = st.number_input("Score de Satisfaction (0-10)", min_value=0.0, max_value=10.0, key="score_satisfaction")
        reclamation = st.selectbox("Réclamation (0 ou 1)", options=[0, 1], key="reclamation")

    # Function to predict attrition
    def predict_attrition(input_data):
        input_data_scaled = scaler.transform(input_data)
        prediction = xgb_model.predict(input_data_scaled)
        return prediction

    if st.button("Prédire l'Attrition Client"):
        input_data = pd.DataFrame([[anciennete_client, categorie_ville, genre,
                                     heures_passees_sur_app, nombre_appareils_enregistres,
                                     nombre_adresses_enregistrees, jours_depuis_derniere_commande,
                                     nombre_commandes_mois_prec, augmentation_commandes_annee_prec,
                                     montant_cashback_moyen, coupons_utilises, score_satisfaction,
                                     reclamation]], 
                                   columns=['Anciennete_Client', 'Categorie_Ville', 'Genre',
                                            'Heures_Passees_Sur_App', 'Nombre_Appareils_Enregistres',
                                            'Nombre_Adresses_Enregistrees', 'Jours_Depuis_Derniere_Commande',
                                            'Nombre_Commandes_Mois_Prec', 'Augmentation_Commandes_Annee_Prec',
                                            'Montant_Cashback_Moyen', 'Coupons_Utilises',
                                            'Score_Satisfaction', 'Reclamation'])

        for column in categorical_columns:
            input_data[column] = encoders[column].transform(input_data[column])

        prediction = predict_attrition(input_data)
        Attrition_Client = prediction[0]

        if Attrition_Client == 0:
            message = "Le client est fidèle, il est susceptible de rester."
            css_class = "success"
            icon = "✅"  
        else:
            message = "Le client est à risque de partir."
            css_class = "alert"
            icon = "⚠️"  

        st.markdown(f"<div class='result {css_class}'><span class='icon'>{icon}</span>{message}</div>", unsafe_allow_html=True)  
else:
    st.warning("Veuillez télécharger le fichier CSV pour continuer.")
