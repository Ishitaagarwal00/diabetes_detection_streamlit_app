# Diabetes Detection App

A Streamlit-based web application for diabetes prediction using machine learning.

## Features

- Interactive web interface
- Real-time predictions
- Feature importance visualization
- Model performance metrics
- Built with Streamlit, scikit-learn, and pandas

## Installation

1. Clone this repository
2. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows use: .venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the app:
```bash
streamlit run app.py
```

The app will open in your default web browser at http://localhost:8501

## Model Details

- Uses Random Forest Classifier
- Trained on the Pima Indians Diabetes Dataset
- Features include: Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, and Age