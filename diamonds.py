# IMPORT LIBRARY
import pandas as pd 
import streamlit as st
import pickle
import numpy as np

# LOAD SAVED FILE 
## load encoder
with open('encoder.pkl','rb')as f:
  encoder=pickle.load(f)

## load scaler
with open('scaler.pkl','rb')as f:
  scaler=pickle.load(f)

## save clustering model
with open('diamond_cluster_model.pkl','rb')as f:
  cluster=pickle.load(f)
    
## save regression model
with open('diamond_regression_model.pkl','rb')as f:
  regression=pickle.load(f)

st.title("💎DIAMONDS PRICE PREDICTION")

carat=st.number_input('carat',0.2,2.0)	
cut=st.selectbox('cut',['Fair','Good','Very Good','Premium','Ideal'])
color=st.selectbox('color',['J','I','H','G','F','E','D'])	
clarity=st.selectbox('clarity',['I1','SI2','SI1','VS2','VS1','VVS2','VVS1','IF'])
depth=st.number_input('depth',45.0,65.0,step=0.5)	
table=st.number_input('table',45.0,65.0,step=0.5)	
x=st.number_input('x',3.5,10.0,step=0.5)	
y=st.number_input('y',3.5,10.0,step=0.5)	
z=st.number_input('z',1.0,6.0,step=0.5)	
volume=st.number_input('volume',30,350,step=10)	
price_per_carat=st.number_input('price_per_carat',500,9000,step=100)
dimension_ratio=st.number_input('dimension_ratio',1.50,1.70)
carat_category=st.selectbox('carat_category',['light','medium','heavy'])

input_df=pd.DataFrame({
    'carat':[carat],	
    'cut':[cut],	
    'color':[color],
    'clarity':[clarity],
    'depth':[depth],
    'table':[table],
    'x':[x],
    'y':[y],
    'z':[z],
    'volume':[volume],	
    'price_per_carat':[price_per_carat],
    'dimension_ratio':[dimension_ratio],
    'carat_category':[carat_category]
})
# APPLY SAME TRANSFORMATIONS AS TRAINING
# -------------------------------------------------------
# ⿡ Log transform numeric columns (same as training)
for col in ['carat', 'volume', 'price_per_carat']:
    input_df[col] = np.log1p(input_df[col])

# ⿢ Encode categorical features
input_df[['cut','color','clarity','carat_category']] = encoder.transform(
    input_df[['cut','color','clarity','carat_category']]
)

# ⿣ Scale numeric features
numeric_cols = ['carat','depth','table','x','y','z','volume','price_per_carat','dimension_ratio']
input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])

# -------------------------------------------------------
# PREDICTION BUTTONS
# -------------------------------------------------------
if st.button("Predict Price"):
    regression_output = regression.predict(input_df)
    price_output = np.round(np.expm1(regression_output[0]), 2)
    st.success(f"Estimated Diamond Price: ₹{price_output:,.2f}")

if st.button("Predict Cluster"):
    cluster_output = cluster.predict(input_df)
    cluster_value = int(cluster_output[0])
    if cluster_value == 0:
        cluster_name = "Premium Heavy Diamonds"
    elif cluster_value == 1:
        cluster_name = "Affordable Small Diamonds"
    else:
        cluster_name = f"Cluster {cluster_value}"

    st.info(f"Cluster: *{cluster_name}* (Cluster {cluster_value})")

