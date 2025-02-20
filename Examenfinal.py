import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from dotenv import load_dotenv
from groq import Groq
from loguru import logger

load_dotenv()

logger.info(os.getenv('GROQ_API_KEY'))

qclient = Groq()

if 'messages' not in st.session_state:
    st.session_state.messages = []

for messages in st.session_state.messages:
    with st.chat_message(messages['role']):
        st.markdown(messages['content'])

# Configuración de la app
st.title("PREDICCIÓN ELECTORAL")
st.subheader("Análisis de votos a partir de comentarios en redes sociales")

# Cargar archivo
uploaded_file = st.file_uploader("Sube tu archivo XLSX", type=["xlsx"])

if uploaded_file:
    df = pd.read_excel(uploaded_file, engine="openpyxl")
    
    # Verificar si la columna 'text' existe
    if 'text' not in df.columns:
        st.error("El archivo no contiene una columna 'text'")
    else:
        # Muestra de datos
        if st.checkbox("Mostrar una muestra aleatoria de los comentarios"):
            sample_size = st.slider("Selecciona la cantidad de muestras", 1, 20, 5)
            st.write(df['text'].sample(sample_size))
        else:
            st.write(df[['text']].head(10))

        # Clasificación de votos
        def clasificar_voto(texto):
            texto = str(texto).lower()
            if "noboa" in texto:
                return "Voto Noboa"
            elif "luisa" in texto:
                return "Voto Luisa"
            else:
                return "Voto Nulo"

        df['Clasificación'] = df['text'].apply(clasificar_voto)
        
        # Mostrar conteo de votos
        conteo_votos = df['Clasificación'].value_counts()
        st.write("### Distribución de votos")
        st.bar_chart(conteo_votos)
        
        # Mostrar cantidad de votos nulos y conclusión
        votos_nulos = conteo_votos.get("Voto Nulo", 0)
        st.write(f"### Votos Nulos: {votos_nulos}")
        if votos_nulos > max(conteo_votos.get("Voto Noboa", 0), conteo_votos.get("Voto Luisa", 0)):
            st.warning("Los votos nulos superan a los candidatos, posible problema de indecisión o rechazo")
        else:
            ganador = conteo_votos.idxmax()
            st.success(f"El candidato con más votos es: {ganador}")


        
        # Guardar resultados
        df.to_csv("resultados_prediccion.csv", index=False)
        st.download_button(
            label="Descargar resultados",
            data=df.to_csv(index=False),
            file_name="resultados_prediccion.csv",
            mime="text/csv"
        )

def process_data(chat_completion) -> str:
    for chunk in chat_completion:
        if chunk.choices[0].delta.content: yield chunk.choices[0].delta.content


if prompt := st.chat_input('Insert questions'):
    with st.chat_message('user'):
        st.markdown(prompt)

    st.session_state.messages.append({'role': 'user', 'content': prompt})

    with st.chat_message('assistant'):
        # stream_response = client.chat.completions.create(
        stream_response = qclient.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "Necesito que con la clasificacion de los resultados respondas la pregunta del usuario",
                },
                {
                    "role": "user",
                    "content": prompt,
                },
            ],
            model="deepseek-r1-distill-llama-70b",

            stream=True
        )

        response = process_data(stream_response)

        response = st.write_stream(response)

    st.session_state.messages.append({'role': 'asistant', 'content': response})
