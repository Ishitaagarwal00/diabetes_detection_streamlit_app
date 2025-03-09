import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set page config
st.set_page_config(
    page_title="Diabetes Detection App",
    page_icon="🏥",
    layout="wide"
)

# Title
st.title("Diabetes Detection App")
st.write("This app predicts diabetes using various health metrics.")

# Load data
@st.cache_data
def load_data():
    # Load the Pima Indians Diabetes Database
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    data = pd.read_csv('https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', 
                       names=columns)
    return data

# Load and prepare data
data = load_data()

# Sidebar for user input
st.sidebar.header('User Input Features')

def user_input_features():
    pregnancies = st.sidebar.slider('Pregnancies', 0, 17, 3)
    glucose = st.sidebar.slider('Glucose', 0, 200, 120)
    blood_pressure = st.sidebar.slider('Blood Pressure', 0, 122, 70)
    skin_thickness = st.sidebar.slider('Skin Thickness', 0, 100, 20)
    insulin = st.sidebar.slider('Insulin', 0, 846, 79)
    bmi = st.sidebar.slider('BMI', 0.0, 67.1, 31.4)
    diabetes_pedigree = st.sidebar.slider('Diabetes Pedigree Function', 0.078, 2.42, 0.3725)
    age = st.sidebar.slider('Age', 21, 81, 29)
    
    data = {
        'Pregnancies': pregnancies,
        'Glucose': glucose,
        'BloodPressure': blood_pressure,
        'SkinThickness': skin_thickness,
        'Insulin': insulin,
        'BMI': bmi,
        'DiabetesPedigreeFunction': diabetes_pedigree,
        'Age': age
    }
    return pd.DataFrame(data, index=[0])

# Get user input
user_data = user_input_features()

# Display user input
st.subheader('User Input Features')
st.write(user_data)

# Prepare the model
X = data.drop('Outcome', axis=1)
y = data['Outcome']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Make prediction on user input
user_data_scaled = scaler.transform(user_data)
prediction = model.predict(user_data_scaled)
prediction_proba = model.predict_proba(user_data_scaled)

# Show prediction
st.subheader('Prediction')
if prediction[0] == 0:
    st.write('The model predicts: No Diabetes')
else:
    st.write('The model predicts: Diabetes')

st.subheader('Prediction Probability')
st.write(f'Probability of No Diabetes: {prediction_proba[0][0]:.2%}')
st.write(f'Probability of Diabetes: {prediction_proba[0][1]:.2%}')

# Model performance
st.subheader('Model Performance')
y_pred = model.predict(X_test_scaled)
st.write(f'Model Accuracy: {accuracy_score(y_test, y_pred):.2%}')

# Feature importance
st.subheader('Feature Importance')
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

st.bar_chart(feature_importance.set_index('Feature'))
