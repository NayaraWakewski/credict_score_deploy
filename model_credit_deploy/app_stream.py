import streamlit as st
from joblib import load
import numpy as np

# Carregar o modelo
model = load('model.jbl')

# Personalização do Streamlit
st.set_page_config(
    page_title="Modelo de Crédito",
    page_icon="💳",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Título e descrição do aplicativo
st.title('Modelo de Crédito 💳')
st.markdown("""
    ### Bem-vindo ao nosso aplicativo de modelo de crédito!
    Este aplicativo permite que você insira algumas características financeiras e obtenha uma previsão 
    sobre a aprovação de crédito com base em nosso modelo de machine learning.
""")

# Criar um contêiner para o formulário de entrada
with st.form(key='credit_form'):
    st.header("Insira suas informações financeiras")
    
    # Campos de entrada
    feature1 = st.number_input('Digite a taxa de débito em relação a renda:', format="%.2f")
    feature2 = st.number_input('Digite a taxa de mantimentos em relação a renda:', format="%.2f")
    feature3 = st.number_input('Digite a taxa de renda:', format="%.2f")

    # Botão de submissão
    submit_button = st.form_submit_button(label='Predizer')

# Processar a predição ao clicar no botão
if submit_button:
    input_features = np.array([feature1, feature2, feature3]).reshape(1, -1)
    
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)

    # Exibir o resultado da predição
    st.subheader("Resultado da Predição:")
    if prediction[0] == 1:
        st.success('Crédito aprovado.')
    else:
        st.warning('Crédito em análise. Forneça mais dados.')
    st.info(f'Probabilidades de aprovação: {prediction_proba[0][1]:.2f}')

# Rodapé personalizado
st.markdown("""
    ---
    Desenvolvido por [Nayara Valevskii](https://github.com/NayaraWakewski)  
    Projeto de aprendizado de máquina para previsão de crédito.
""")


