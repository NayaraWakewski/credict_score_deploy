# Modelo de Crédito - Diamond - Nayara Valevskii

Este é um aplicativo de modelo de crédito desenvolvido em Streamlit. O aplicativo permite aos usuários inserir determinadas características financeiras e obter uma previsão sobre a aprovação de crédito com base em um modelo de machine learning.

## Descrição

O aplicativo foi desenvolvido como parte de uma aula prática de Machine Learning, onde implementamos um modelo de classificação para prever a aprovação de crédito. O modelo foi treinado utilizando técnicas de aprendizado supervisionado e serializado utilizando a biblioteca `joblib`.

## Funcionalidades

- **Entrada de Dados**: O aplicativo permite a entrada de três características financeiras principais:
  - Taxa de débito em relação à renda.
  - Taxa de mantimentos em relação à renda.
  - Taxa de renda.

- **Predição de Crédito**: Com base nos valores inseridos, o modelo faz uma predição para determinar se o crédito é aprovado ou se está em análise.

- **Probabilidades de Aprovação**: Além da predição binária, o aplicativo também exibe as probabilidades de aprovação do crédito.

## Como Usar

1. **Configurar o Ambiente**:
   - Certifique-se de ter o Python instalado em seu sistema.
   - Instale as dependências necessárias:
     ```sh
     pip install streamlit joblib numpy
     ```

2. **Executar o Aplicativo**:
   - Certifique-se de que o arquivo `model.jbl` (modelo treinado) está no mesmo diretório que o script `app_stream.py`.
   - Execute o aplicativo Streamlit:
     ```sh
     streamlit run app_stream.py
     ```

3. **Inserir os Dados**:
   - Abra o navegador no endereço indicado pelo Streamlit.
   - Insira os valores nos campos de entrada:
     - **Taxa de débito em relação à renda**: Número decimal.
     - **Taxa de mantimentos em relação à renda**: Número decimal.
     - **Taxa de renda**: Número decimal.
   - Clique no botão `Predizer` para obter a previsão.

## Código Fonte

```python
import streamlit as st
from joblib import load
import numpy as np

# Carregar o modelo
model = load('model.jbl')

# Título do aplicativo
st.title('Modelo de crédito')

# Campos de entrada
feature1 = st.number_input('Digite a taxa de débito em relação a renda:', format="%.2f")
feature2 = st.number_input('Digite a taxa de mantimentos em relação a renda:', format="%.2f")
feature3 = st.number_input('Digite a taxa de renda:', format="%.2f")

# Botão de predição
if st.button('Predizer'):
    input_features = np.array([feature1, feature2, feature3]).reshape(1, -1)
    prediction = model.predict(input_features)
    prediction_proba = model.predict_proba(input_features)

    if prediction[0] == 1:
        st.write('Crédito aprovado.')
    else:
        st.write('Crédito em análise. Forneça mais dados.')
    st.write(f'Probabilidades de aprovação: {prediction_proba[:, 1]}')


  ## 🎁 Expressões de gratidão

* Compartilhe com outras pessoas esse projeto 📢;
* Quer saber mais sobre o projeto? Entre em contato para tomarmos um :coffee:;

---
⌨️ com ❤️ por [Nayara Vakevskii](https://www.linkedin.com/in/nayaraba/) 😊    
