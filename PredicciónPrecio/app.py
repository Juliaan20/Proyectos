
# Carga las librerías necesarias
import streamlit as st
import numpy as np
import joblib

# Cargamos el modelo
model = joblib.load('rfr_model.pkl')

# Título de la aplicación
st.title("Predicción de Precios de Computadoras")
st.divider()

# Descripción de la aplicación
st.write("Ingrese las características de la computadora para predecir su precio.")

# Entradas del usuario

processor_speed = st.number_input("Velocidad del Procesador (GHz)", min_value=1.0, max_value=5.0, value=2.5)
ram = st.number_input("Cantidad de RAM (GB)", min_value=1, max_value=64, value=8)
storage = st.number_input("Almacenamiento (GB)", min_value=128, max_value=2000, value=512)  


# Convertimos las entradas del usuario a un formato adecuado para el modelo
x = [processor_speed, ram, storage]
st.divider()


# Realizamos la predicción
prediction = st.button("Predecir Precio")
st.divider()


if prediction:
    st.balloons()
    x1 = np.array(x)
    prediction_value = model.predict([x1])[0]
    st.success(f"El precio estimado de la computadora es: ${prediction_value:.2f}")
else:
    st.warning("Por favor, complete todos los campos y presione el botón para predecir el precio.")


