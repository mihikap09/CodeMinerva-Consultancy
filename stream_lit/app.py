# streamlit_app/app.py

import streamlit as st
import pandas as pd

from model import load_model, make_recommendations


# Load the trained model and label encoder
model, label_encoder = load_model()

# Load user1 data
user1_data = {
    'body_type': 'Pear',
    'style_preference': 'Casual',
    'color_preference': 'Blue',
    'occasion': 'Rakhi',
    'weather': 'Hot'
}

# Get recommendations for user1
recommendation = make_recommendations(user1_data, model, label_encoder)

# Streamlit app title and description
st.title("Personalized AI Fashion Consultant")
st.write("Welcome to the AI Fashion Consultant. Here is your personalized style report:")

# Display user1 data
st.write("### User1 Data")
st.write(pd.DataFrame([user1_data]))

# Display recommendations
st.write("### Personalized Recommendation")
st.write(f"Recommended Style: {recommendation}")

