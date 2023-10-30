import streamlit as st
import pandas as pd
import numpy as np
import pickle

file1 = open('smoke_cat_1.pkl', 'rb')
model = pickle.load(file1)
file1.close()
print("Model type:", type(model))
# 
data = pd.read_csv("df_1_for_ml.csv")

data['Road Name'].unique()

st.title("Predicte Car Accident Severity: ")

road = st.selectbox('Road', data['Road Name'].unique())



# weather condition

weather = st.selectbox('Weather', data['Weather'].unique())

# Light condition

light = st.selectbox('Light', data['Light'].unique())

# Surface Condition condition

Surface_Condition = st.selectbox('Surface Condition', data['Surface Condition'].unique())

# Speed limit

Speed = st.selectbox('Speed Limit, M/H', [0, 5, 10, 15, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70])




if st.button('Predict Severity'):

    query = np.array([road, weather, light, Surface_Condition, Speed ])

    query = query.reshape(1, 5)

    #prediction = int(np.exp(rf.predict(query)[0]))
    prediction = model.predict(query)[0]


    st.title("Predicted of car Accident Severity would be "+
             str(prediction))
