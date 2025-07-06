import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the saved model
@st.cache_resource
def load_model():
    return joblib.load('penguin_model.joblib')

model_data = load_model()
model = model_data['model']
features = model_data['features']
classes = model_data['classes']

# App UI
st.title("🐧 Penguin Predictor (Optimized)")


# User input
st.sidebar.header("Input Features")
inputs = {}
col1, col2 = st.columns(2)

with col1:
    inputs['bill_length_mm'] = st.slider("Bill length (mm)", 30.0, 60.0, 40.0)
    inputs['bill_depth_mm'] = st.slider("Bill depth (mm)", 10.0, 25.0, 18.0)
    inputs['flipper_length_mm'] = st.slider("Flipper length (mm)", 170.0, 240.0, 200.0)

with col2:
    inputs['body_mass_g'] = st.slider("Body mass (g)", 2500, 6500, 4000)
    inputs['island_Dream'] = 1 if st.radio("Island", ["Biscoe", "Dream", "Torgersen"]) == "Dream" else 0
    inputs['island_Torgersen'] = 1 if inputs['island_Dream'] == 0 and st.radio("", ["", "", ""]) == "Torgersen" else 0
    inputs['sex_male'] = 1 if st.radio("Sex", ["Female", "Male"]) == "Male" else 0

# Prediction
if st.button("Predict"):
    input_df = pd.DataFrame([inputs])
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    st.success(f"Predicted Species: **{prediction}**")
    
    # Show confidence
    fig, ax = plt.subplots()
    sns.barplot(x=classes, y=proba, palette="Blues_d")
    plt.title("Prediction Confidence")
    st.pyplot(fig)



# Add visualizations/explanation as needed
# ===================================================|Viz - 1|================+====================================
st.subheader("Top Influential Features")

# Get feature importance
importance = pd.Series(model.feature_importances_, index=features)
importance = importance.sort_values(ascending=True).tail(5)  # Top 5 only

# Create figure explicitly
fig, ax = plt.subplots(figsize=(8,4))
sns.barplot(x=importance.values, y=importance.index, palette="Blues_d", ax=ax)
ax.set_title("Which features matter most?")
ax.set_xlabel("Importance Score")

# Explicitly pass figure to st.pyplot()
st.pyplot(fig)

# ==================================================|Viz - 2|=======================================================
st.subheader("Prediction Confidence")

if 'prediction' in locals():
    max_prob = round(proba.max()*100)
    
    # Create figure explicitly
    fig, ax = plt.subplots(figsize=(6,2))
    ax.barh(0, max_prob, color=plt.cm.Blues(0.6))
    ax.set_xlim(0, 100)
    ax.text(max_prob/2, 0, f"{max_prob}%", 
            ha='center', va='center', color='white', fontsize=12)
    ax.axis('off')
    ax.set_title(f"Model Confidence in '{prediction}' Prediction")
    
    # Explicitly pass figure
    st.pyplot(fig)