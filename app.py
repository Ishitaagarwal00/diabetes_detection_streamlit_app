import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Set page configuration at the top
st.set_page_config(
    page_title="Diabetes Detection App",
    page_icon="🏥",
    layout="wide"
)

# Hide Streamlit UI elements
st.markdown(
    """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)

# Title and description
st.title("Diabetes Detection App")
st.write("This app predicts diabetes using various health metrics.")

# Load dataset function
@st.cache_data
def load_data():
    columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    data = pd.read_csv(
        'https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv', 
        names=columns
    )
    return data

# Load the dataset
data = load_data()

# Sidebar input
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
    
    user_data = pd.DataFrame({
        'Pregnancies': [pregnancies],
        'Glucose': [glucose],
        'BloodPressure': [blood_pressure],
        'SkinThickness': [skin_thickness],
        'Insulin': [insulin],
        'BMI': [bmi],
        'DiabetesPedigreeFunction': [diabetes_pedigree],
        'Age': [age]
    })
    
    return user_data

# Get user input
user_data = user_input_features()

# Show user input
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

# Show prediction result
st.subheader('Prediction Result')
if prediction[0] == 0:
    st.success('The model predicts: No Diabetes')
else:
    st.error('The model predicts: Diabetes')

# Feature importance visualization
st.subheader('Feature Importance')
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)

# Plot feature importance
fig, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x=feature_importance['Importance'], y=feature_importance['Feature'], palette='coolwarm', ax=ax)
ax.set_title('Feature Importance in Diabetes Prediction')

# Show plot in Streamlit
st.pyplot(fig)

