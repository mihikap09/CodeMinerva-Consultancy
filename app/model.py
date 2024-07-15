# app/model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

def preprocess_data(df):
    # Define features and target
    X = df.drop(columns=['user_id', 'username', 'recommended'])
    y = df['recommended']
    
    # Preprocess categorical data
    categorical_features = ['body_type', 'style_preference', 'color_preference', 'occasion', 'weather']
    numerical_features = []  # Assuming there are no numerical features for now
    
    # Encoding categorical features
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(), categorical_features)
        ],
        remainder='passthrough'
    )
    
    # Encode the target variable
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    return X, y_encoded, preprocessor, label_encoder

def train_model():
    # Load the dataset
    df = pd.read_csv('C:\AI-Consultant-Streamlit\Data\dummy_data.csv')
    
    # Preprocess the data
    X, y, preprocessor, label_encoder = preprocess_data(df)
    
    # Create and train the model
    model = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier())
    ])
    
    model.fit(X, y)
    
    # Save the model and label encoder
    joblib.dump(model, 'model.joblib')
    joblib.dump(label_encoder, 'label_encoder.joblib')

def load_model():
    # Load the saved model and label encoder
    model = joblib.load('model.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    return model, label_encoder

def make_recommendations(user_data, model, label_encoder):
    # Convert user_data to DataFrame and make predictions
    user_df = pd.DataFrame([user_data])
    prediction = model.predict(user_df)
    recommendation = label_encoder.inverse_transform(prediction)
    return recommendation[0]

if __name__ == '__main__':
    train_model()


