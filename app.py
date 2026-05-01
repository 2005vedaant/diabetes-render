# -*- coding: utf-8 -*-
"""
Created on Sat Apr 18 17:16:25 2026

@author: Admin
"""

import numpy as np
import pickle
import streamlit as st
loaded_model=pickle.load(open('trained_model (1).sav','rb'))
sc=pickle.load(open('Scaler (1).sav','rb'))

st.title('Diabetes Prediction System') #streamlit title
 
#user input
Pregnancies=st.text_input('Number of Pregnancies')
Glucose=st.text_input('Glucose Level')
BloodPresure=st.text_input('Blood Pressure Level')
SkinThickness=st.text_input('Skin Thickness')
Insulin=st.text_input('Insulin Level')
BMI=st.text_input('BMI Level')
DiabetesPedigreeFunction=st.text_input('Diabetes Pedegree Function')
Age=st.text_input('Age of patient')
 
#store prediction result
diab_pred=''
 
#creating button
if st.button('Diabetes Result'):
    try:
        input_data = [
            float(Pregnancies),
            float(Glucose),
            float(BloodPresure),
            float(SkinThickness),
            float(Insulin),
            float(BMI),
            float(DiabetesPedigreeFunction),
            float(Age)
        ]

        # Convert to 2D array
        input_data = np.asarray(input_data).reshape(1, -1)

        # Apply scaler
        input_data = sc.transform(input_data)

        # Prediction
        diab = loaded_model.predict(input_data)

        if diab[0] == 1:
            diab_pred = 'The person is diabetic'
        else:
            diab_pred = 'The person is not diabetic'

    except:
        diab_pred = "Please enter valid numeric values"
st.success(diab_pred)
   