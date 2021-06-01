import pandas as pd
import streamlit as st
import pickle
import os
import numpy as np

import joblib





pipeline = pickle.load(open('models/pipeline', 'rb'))
model_path = os.path.join('models', "linear_regression.joblib")
linear_reg_model =  joblib.load(model_path)


st.title("House Value Prediction")


def predict_value(long, lat, total_rooms, total_bds, pop, hsh, inc, medi_age,prox):
    df = pd.DataFrame(
        data=[[long, lat,medi_age,total_rooms,total_bds,pop, hsh, inc,prox]],
        columns=['longitude', 'latitude', 'housing_median_age', 'total_rooms',
       'total_bedrooms', 'population', 'households', 'median_income',
        'ocean_proximity'])
    X = pipeline.transform(df)

    y = linear_reg_model.predict(X)
    return y



@st.cache
def load_data(dataset):
    return pd.read_csv(dataset)


def main():

    with st.form(key='housevalue-predict'):
        long = st.number_input("Longitude")
        lat = st.number_input("Latitude")
        total_rooms = st.number_input("Total Rooms", min_value=0,help="Total number of rooms within a block")
        total_bds = st.number_input("Total Bedrooms", min_value=0,help="Total number of bedrooms within a block")
        pop = st.number_input("Population", min_value=0,help="Total number of people residing within a block")
        hsh = st.number_input("Households", min_value=0,help="Total number of households, a group of people residing within a home unit, for a block")
        inc = st.number_input("Median Income (in tens of thousands of dollars)",min_value=0.0,help="Median income for households within a block of houses (measured in tens of thousands of US Dollars)")
        medi_age = st.number_input("Median House Age",min_value=0,help="Median age of a house within a block")
        prox = st.selectbox("Ocean Proximity", ['NEAR OCEAN', 'INLAND', '<1H OCEAN', 'NEAR BAY', 'ISLAND'])

        submit_button = st.form_submit_button(label='Predict')
        if submit_button:
            pred = predict_value(long, lat, total_rooms, total_bds, pop, hsh, inc, medi_age,prox)

            st.success(f"Predicted house value {np.asscalar(pred)}")




    # result = ""
    # if st.button("Predict"):
    #     result = predict_churn()
    # st.success('The output is {}'.format(result))


if __name__ == '__main__':
    main()
