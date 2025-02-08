import streamlit as st
import pandas as pd
import joblib
import pickle


st.title('Wine Quality Prediction App')

st.sidebar.header('User Input Parameters')
												
fixed_acidity =  st.sidebar.text_input('fixed acidity', 8.5)
volatile_acidity =  st.sidebar.text_input('volatile acidity', 0.46)
citric_acid = st.sidebar.text_input('citric acid', 0.31)
residual_sugar = st.sidebar.text_input('residual sugar', 2.25)
chlorides = st.sidebar.text_input('chlorides', 0.078)
free_sulfur_dioxide = st.sidebar.text_input('free sulfur dioxide',32.0 )
total_sulfur_dioxide= st.sidebar.text_input('total sulfur dioxide', 58.0)
density= st.sidebar.text_input('density',0.998)
pH = st.sidebar.text_input('pH', 3.33)
sulphates= st.sidebar.text_input('sulphates', 0.54)
alcohol= st.sidebar.text_input('alcohol',9.8 )
quality = st.sidebar.text_input('quality', 5)


input_data = {
'fixed acidity' : fixed_acidity,
 'volatile acidity':volatile_acidity,
 'citric acid':citric_acid,
 'residual sugar':residual_sugar,
 'chlorides':chlorides,
 'free sulfur dioxide':free_sulfur_dioxide,
 'total sulfur dioxide':total_sulfur_dioxide,
 'density':density,
 'pH':pH,
 'sulphates':sulphates,
 'alcohol':alcohol,
 'quality':quality,
}

input_data_df = pd.DataFrame([input_data])


model = joblib.load("app\model_with_pipeline.joblib")

input_data_df = input_data_df.drop(columns=["quality"])
result = model.predict(input_data_df)

st.table(input_data_df)

st.metric('Predicted Wine Quality', f'{result[0]:,.2f}')