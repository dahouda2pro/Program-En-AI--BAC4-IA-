import streamlit as st # pip install streamlit
import numpy as np
import pandas as pd
import joblib
from tensorflow.keras.models import load_model

# Chargement du modèle et du scaler
model_dnn = load_model("model_dnn.h5")
scaler = joblib.load("scaler_ic.pkl")  # scaler entraîné sur les données d'origine

# --- Interface utilisateur ---
st.title("🧠 Classification des sols basée sur l'apprentissage profond")
st.subheader("🔢 Entrez les paramètres du site")

site = st.number_input("Site")
longitude = st.number_input("Longitude")
latitude = st.number_input("Latitude")
altitude = st.number_input("Altitude")
ic = st.number_input("Ic")

if st.button("Prédire la classe"):
    st.write("📍 Données saisies :")
    st.write(f"Site : {site}")
    st.write(f"Longitude : {longitude}")
    st.write(f"Latitude : {latitude}")
    st.write(f"Altitude : {altitude}")
    st.write(f"Ic : {ic}")

    # Préparer l'entrée utilisateur
    user_input = np.array([[site, longitude, latitude, altitude, ic]])
    user_input_scaled = scaler.transform(user_input)
    # st.write("Normalisation ", user_input_scaled)

    # Prédiction
    prediction = model_dnn.predict(user_input_scaled)
    predicted_class = np.argmax(prediction)
    probabilities = np.round(prediction[0] * 100, 2)

    # Dictionnaire des classes
    classe_labels = {
        0: "Dur",
        1: "Liquide",
        2: "Mi-dur",
        3: "Terne"
    }
    predicted_label = classe_labels[predicted_class]

    # Résultats
    st.success(f"🎯 Classe prédite : {predicted_label} ({predicted_class})")
    st.write("📊 Probabilités par classe (%) :")
    for i, prob in enumerate(probabilities):
        st.write(f"{classe_labels[i]} ({i}) : {prob}%")

    # Carte
    st.subheader("🗺️ Localisation sur la carte")
    map_data = pd.DataFrame({'lat': [latitude], 'lon': [longitude]})
    #st.map(map_data)
    st.map(map_data, size=20, color="#0044ff")
