import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def load_model():
    with open("Binary_pred_rainfall_dataset.pkl","rb") as file:
        model,preprocessor=pickle.load(file)
    return model,preprocessor

def preprocessing_input_data(data,preprocessor):
    data=preprocessor.transform(data)
    df=pd.DataFrame(data)
    return df

def predict_data(data):
    model, preprocessor=load_model()
    df=preprocessing_input_data([list(data.values())],preprocessor)
    prediction=model.predict(df)
    return prediction

def main():
    st.title("Binary Prediction with a Rainfall Dataset")
    st.write("Enter the metrics to predict the probability of rainfall")
    
    day=st.number_input("Day of the Year",min_value=1,max_value=365,value=1)
    pressure=st.number_input("Pressure",min_value=990,max_value=1050,value=990)
    maxtemp=st.number_input("Max Temperature",min_value=5,max_value=40,value=10)
    temparature=st.number_input("Temparature",min_value=5,max_value=38,value=10)
    mintemp=st.number_input("Minimum Temperature",min_value=2,max_value=35,value=10)
    dewpoint=st.number_input("Dew Point",min_value=0,max_value=30,value=20)
    humidity=st.number_input("Humidity",min_value=35,max_value=100,value=40)
    cloud=st.number_input("Cloud",min_value=0,max_value=100,value=1)
    sunshine=st.number_input("Sunshine",min_value=0,max_value=15,value=1)
    winddirection=st.number_input("Wind Direction",min_value=10,max_value=300,value=20)
    windspeed=st.number_input("Wind Speed",min_value=2,max_value=64,value=10)

    if st.button("Predict the probability of the Rainfall:"):
        user_data={
            "day":day,
            "pressure":pressure,
            "maxtemp":maxtemp,
            "temparature":temparature,
            "mintemp":mintemp,
            "dewpoint":dewpoint,
            "humidity":humidity,
            "cloud":cloud,
            "sunshine":sunshine,
            "winddirection":winddirection,
            "windspeed":windspeed,
            "extra_feature": 0
        }
        prediction=predict_data(user_data)
        st.success(f"Probability of the Rainfall:{prediction[0]}")


if __name__ == "__main__":
    main()