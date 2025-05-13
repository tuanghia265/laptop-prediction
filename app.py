import streamlit as st
import pickle
import numpy as np
import pandas as pd
# import the model
pipe = pickle.load(open('pipe.pkl','rb'))
df = pickle.load(open('data.pkl','rb'))

st.set_page_config(page_title="Laptop Prediction", layout="wide")
st.title("Laptop Predictor")
columns = ['Company', 'TypeName', 'Ram', 'Weight', 'TouchScreen', 'IPS', 'ppi', 'CPU Brand', 'HDD', 'SSD', 'GPU Brand', 'os']
# brand
company = st.selectbox('Brand',df['Company'].unique())

# type of laptop
type = st.selectbox('Type',df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)',[2,4,6,8,12,16,24,32,64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen',['No','Yes'])

# IPS
ips = st.selectbox('IPS',['No','Yes'])

# screen size
screen_size = st.slider('Scrensize in inches', 10.0, 18.0, 13.0)

# resolution
resolution = st.selectbox('Screen Resolution',['1920x1080','1366x768','1600x900','3840x2160','3200x1800','2880x1800','2560x1600','2560x1440','2304x1440'])

#cpu
cpu = st.selectbox('CPU',df['CPU Brand'].unique())

hdd = st.selectbox('HDD(in GB)',[0,128,256,512,1024,2048])

ssd = st.selectbox('SSD(in GB)',[0,8,128,256,512,1024])

gpu = st.selectbox('GPU',df['GPU Brand'].unique())

os = st.selectbox('OS',df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5/screen_size
    query = pd.DataFrame([[company,type,ram,weight,touchscreen,ips,ppi,cpu,hdd,ssd,gpu,os]],columns=columns)

    # query = query.reshape(1,12)
    prediction = int(np.exp(pipe.predict(query)[0]))
    st.title("The predicted price of this configuration is " + str(int(np.exp(pipe.predict(query)[0]))))
