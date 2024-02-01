import streamlit as st
import pandas as pd
import numpy as np
import joblib
from neuron import Neuron



def mostrar_bloques(etiqueta,letra,cantidad):
    # Crea columnas para organizar los bloques en una fila
    columns = st.columns(cantidad)

    # Lista para almacenar los valores ingresados
    valores_ingresados = []

    for idx, col in enumerate(columns):
        # Muestra el número de entrada para la columna
        key = f"{etiqueta}_W_input_{idx}"  # Genera una clave única
        number = col.number_input(f"W{idx}", key=key, value=0.00)
        
        # Almacena el valor ingresado en la lista
        valores_ingresados.append(number)

    # Muestra el valor ingresado para todas las columnas
    st.write(f"{letra} = {[val for i, val in enumerate(valores_ingresados)]}")
    return valores_ingresados


def main():
    st.set_page_config(layout="wide")
    st.image('fotoNeurona.jpg', width=200)
    st.title("Simulador de neurona")

    numero_entradas = st.slider('Elige el número de entradas/pesos que tendrá la neurona', 1, 10, 1)
   
    st.title("Pesos")
    valores_ingresados1=mostrar_bloques("Pesos","w",numero_entradas)
    st.title("Entradas")
    valores_ingresados2=mostrar_bloques("Entradas","x",numero_entradas)

    col1, col2 = st.columns(2)
    col1.write("Sesgo")
    col2.write("Función de activación")
    sesgo = col1.number_input(f"Introduce el valor del sesgo", value=0.00)
    funcion = col2.selectbox(
    'Elige la función de activación',
    ('Sigmoide', 'ReLu', 'TangenteHiperbólica'))
    
    if st.button('Calcular'):

        n1=Neuron(weights=valores_ingresados1, bias=sesgo,func=funcion)
        x = valores_ingresados2 
        output= n1.run(input_data=x)
        st.write(output)

if __name__ == "__main__":
    main()
