import streamlit as st
import pandas as pd
import joblib  # For loading the model and scaler

# Load the XGBoost model and scaler
xgb_model = joblib.load('xgb_model_churn.pkl')
scaler = joblib.load('scaler.pkl')

# Load your dataset (make sure the path is correct)
df_renamed = pd.read_csv('E_Commerce_Dataset_cleaned.csv')  # Replace with the path to your CSV file

# Title of the application
st.set_page_config(page_title="Prédiction de l'Attrition Client", page_icon="🔍")
st.title("Prédiction de l'Attrition Client")

# Load encoders for categorical columns
encoders = {}
categorical_columns = ['Genre']

for column in categorical_columns:
    encoders[column] = joblib.load(f'{column}_encoder.pkl')

# Custom CSS for styling
st.markdown("""
<style>
    .input-container {
        border: 2px solid #4CAF50; /* Green border */
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 20px;
    }
    h2 {
        color: #4CAF50;
        text-align: center;
    }
    .result {
        font-size: 18px;
        font-weight: bold;
        padding: 20px;
        border-radius: 8px;
        background-color: #f8f8f8; /* Light background */
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .alert {
        color: #FF5733; /* Red color for alert */
        border: 2px solid #FF5733; /* Red border */
        background-color: #ffe6e1; /* Light red background */
    }
    .success {
        color: #4CAF50; /* Green color for success */
        border: 2px solid #4CAF50; /* Green border */
        background-color: #e8f5e9; /* Light green background */
    }
    .icon {
        font-size: 40px;
        margin-right: 15px;
    }
</style>
""", unsafe_allow_html=True)

# Create inputs for the features
with st.container():
    # Inputs for client features
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

    # Selectors for attrition and complaints
    reclamation = st.selectbox("Réclamation (0 ou 1)", options=[0, 1], key="reclamation")

# Function to predict attrition
def predict_attrition(input_data):
    # Normalize the data
    input_data_scaled = scaler.transform(input_data)

    # Make prediction
    prediction = xgb_model.predict(input_data_scaled)
    return prediction

# Create a button to make the prediction
if st.button("Prédire l'Attrition Client"):
    # Prepare input data as a DataFrame
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

    # Encode categorical variables
    for column in categorical_columns:
        input_data[column] = encoders[column].transform(input_data[column])

    # Make the prediction
    prediction = predict_attrition(input_data)

    # Interpret the result
    Attrition_Client = prediction[0]
    if Attrition_Client == 0:
        message = "Le client est fidèle, il est susceptible de rester."
        css_class = "success"
        icon = "✅"  # Icon for success
    else:
        message = "Le client est à risque de partir."
        css_class = "alert"
        icon = "⚠️"  # Icon for alert

    # Display the result with an icon
    st.markdown(f"<div class='result {css_class}'><span class='icon'>{icon}</span>{message}</div>", unsafe_allow_html=True)
