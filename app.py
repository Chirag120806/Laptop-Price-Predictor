import streamlit as st

import pickle 

import numpy as np

import pandas as pd

# Import the model

from pathlib import Path

MODELS_DIR = Path(__file__).parent / 'models'
pipe = pickle.load(open(MODELS_DIR / 'pipe.pkl','rb'))
df = pickle.load(open(MODELS_DIR / 'df.pkl','rb'))

st.title("Laptop Price Predictor")

# Brand
company = st.selectbox('Brand',df['Company'].unique())

# Type of laptop
type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('Ram(in GB)', [2,4,8,12, 16, 24,32,64])

# Weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS', ['No','Yes'])

# Screen size
screen_size = st.number_input('Screen Size', min_value=10.0, step=0.1)

# Resolution

resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

# CPU
cpu = st.selectbox('CPU', df['Cpu Brand'].unique())

# HDD
hdd = st.selectbox('HDD(in GB)',[0,128,256, 512, 1024, 2048])

# SSD
ssd = st.selectbox('SSD(in GB)',[0,8,128,256, 512, 1024, 2048])

#GPU
gpu = st.selectbox('GPU',df['Gpu brand'].unique())

# OS
os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):

    if screen_size == 0:
        st.error("Screen size cannot be 0")
        st.stop()

    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    ppi = np.sqrt((X_res**2) + (Y_res**2)) / screen_size

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'Ips': [ips],
        'ppi': [ppi],
        'Cpu Brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })

    prediction = pipe.predict(query)[0]
    st.title("Predicted Price: ₹" + str(round(int(np.exp(prediction)))))