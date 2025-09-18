import streamlit as st
import pandas as pd
import numpy as np
import joblib
from PIL import Image

# Set page configuration
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load the trained model and scaler
@st.cache_resource
def load_model():
    model = joblib.load('random_forest_house_price.pkl')
    return model

@st.cache_resource
def load_scaler():
    scaler = joblib.load('scaler.pkl')
    return scaler

model = load_model()
scaler = load_scaler()

# App title and description
st.title('🏠 House Price Prediction')
st.write("""
This app predicts house prices in California based on various features using a machine learning model.
Adjust the input values in the sidebar and see the predicted price update in real-time.
""")

# Sidebar for user input
st.sidebar.header('Input Features')

def user_input_features():
    med_inc = st.sidebar.slider('Median Income (in tens of thousands)', 0.0, 15.0, 3.0)
    house_age = st.sidebar.slider('House Age (years)', 0.0, 100.0, 20.0)
    ave_rooms = st.sidebar.slider('Average Rooms', 0.0, 20.0, 5.0)
    ave_bedrms = st.sidebar.slider('Average Bedrooms', 0.0, 10.0, 1.0)
    population = st.sidebar.slider('Population (in hundreds)', 0.0, 10000.0, 1000.0)
    ave_occup = st.sidebar.slider('Average Occupancy', 0.0, 10.0, 2.0)
    latitude = st.sidebar.slider('Latitude', 32.0, 42.0, 35.0)
    longitude = st.sidebar.slider('Longitude', -125.0, -114.0, -118.0)
    
    data = {
        'MedInc': med_inc,
        'HouseAge': house_age,
        'AveRooms': ave_rooms,
        'AveBedrms': ave_bedrms,
        'Population': population,
        'AveOccup': ave_occup,
        'Latitude': latitude,
        'Longitude': longitude
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Display user input
st.header('Input Features')
st.write(input_df)

# Make prediction
def predict_price(input_data):
    # Scale the data
    input_data_scaled = scaler.transform(input_data)
    
    # Make prediction
    prediction = model.predict(input_data_scaled)
    
    return prediction[0] * 100000  # Convert to actual dollar amount

# Predict button
if st.button('Predict Price'):
    price = predict_price(input_df)
    st.header('Prediction Result')
    st.success(f'Predicted House Price: ${price:,.2f}')

# Add some information about the model
st.header('About the Model')
st.write("""
This prediction model uses a Random Forest algorithm trained on the California Housing Dataset.
The model takes into account various factors including median income, house age, room statistics,
and geographical location to estimate house prices.

**Note:** This is a simplified model for demonstration purposes. Actual house prices depend on
many additional factors not included in this model.
""")

# Add a footer
st.markdown("---")
st.markdown(
    """
    <style>
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #f1f1f1;
        color: black;
        text-align: center;
        padding: 10px;
        font-size: 14px;
    }
    </style>
    <div class="footer">
        <p>House Price Prediction App • Built with Streamlit</p>
    </div>
    """,
    unsafe_allow_html=True
)