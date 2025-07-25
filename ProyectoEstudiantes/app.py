import streamlit as st
import numpy as np
import joblib
import warnings
warnings.filterwarnings("ignore")

# Cargar modelo entrenado
model = joblib.load('best_student_performance_model.pkl')

st.title("Predicción del Rendimiento Estudiantil")

# Entradas del usuario
study_hours = st.slider("Horas de estudio por día", 0.0, 12.0, 2.0)
attendance = st.slider("Porcentaje de asistencia", 0.0, 100.0, 80.0)
mental_health = st.slider("Nivel de salud mental (1-10)", 1, 10, 5)
sleep_hours = st.slider("Horas de sueño por noche", 0.0, 12.0, 7.0)
part_time_job = st.selectbox("¿Trabajas a tiempo parcial?", ["No", "Yes"])

# Codificar variable categórica
ptj_encoded = 1 if part_time_job == "Yes" else 0

# Botón de predicción
if st.button("Predecir rendimiento"):
    input_data = np.array([[study_hours, ptj_encoded, attendance, sleep_hours]])
    prediction = model.predict(input_data)[0]
    prediction = max(0, min(100, prediction))  # Limitar entre 0 y 100

    st.success(f"Predicción del rendimiento estudiantil: {prediction:.2f} puntos")
    st.write("¡Gracias por usar nuestra aplicación!")
