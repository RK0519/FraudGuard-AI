import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler

# --- Page Config ---
st.set_page_config(page_title="FraudGuard AI", page_icon="💳", layout="wide")

# --- Custom Styling ---
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    </style>
    """, unsafe_allow_html=True)

# --- Data & Model Logic ---
@st.cache_resource
def load_and_train():
    # 1. Load Data
    try:
        data = pd.read_csv('credit.csv')
    except FileNotFoundError:
        return None, None, None, None, None

    # 2. Balance Dataset (Undersampling)
    legit = data[data.Class == 0]
    fraud = data[data.Class == 1]
    legit_sample = legit.sample(n=len(fraud), random_state=2)
    balanced_data = pd.concat([legit_sample, fraud], axis=0)

    # 3. Features & Target
    X = balanced_data.drop(columns="Class")
    Y = balanced_data["Class"]
    
    # 4. Split & Scale
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. Train Model
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, Y_train)
    
    # 6. Accuracy for efficiency check
    train_acc = accuracy_score(model.predict(X_train_scaled), Y_train)
    test_acc = accuracy_score(model.predict(X_test_scaled), Y_test)
    
    return model, scaler, X_test_scaled, Y_test, (train_acc, test_acc)

# Initialize
model, scaler, X_test_scaled, Y_test, accs = load_and_train()

# --- Sidebar ---
st.sidebar.title("🛡️ Model Control")
if model:
    st.sidebar.success("Model Trained Successfully")
    st.sidebar.write(f"**Training Accuracy:** {accs[0]:.2%}")
    st.sidebar.write(f"**Testing Accuracy:** {accs[1]:.2%}")
    
    if st.sidebar.checkbox("Show Efficiency (Confusion Matrix)"):
        st.subheader("Model Efficiency Analysis")
        y_pred = model.predict(X_test_scaled)
        cm = confusion_matrix(Y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Reds', ax=ax)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        st.pyplot(fig)

# --- Main UI ---
st.title("Credit Card Fraud Detection System")
st.write("Enter transaction features below to predict if a transaction is Legitimate or Fraudulent.")

if not model:
    st.error("Error: 'credit.csv' not found in the project folder. Please add the file and refresh.")
else:
    # Input Area
    input_text = st.text_area(
        "Paste all features (30 numerical values separated by commas):",
        placeholder="Example: 0.0, -1.35, 2.1, 0.5, ...",
        help="The model expects the Time, V1-V28, and Amount columns."
    )

    if st.button("Run Diagnostic"):
        try:
            # Clean input
            values = [float(x.strip()) for x in input_text.replace('\t', ',').split(',') if x.strip()]
            
            if len(values) != 30:
                st.warning(f"Feature mismatch: Expected 30 features, but got {len(values)}.")
            else:
                # Prediction
                features = np.array(values).reshape(1, -1)
                features_scaled = scaler.transform(features)
                prediction = model.predict(features_scaled)
                prob = model.predict_proba(features_scaled)[0][1]

                st.markdown("---")
                if prediction[0] == 0:
                    st.balloons()
                    st.success(f"✅ **LEGITIMATE TRANSACTION** (Confidence: {1-prob:.2%})")
                else:
                    st.error(f"🚨 **FRAUDULENT TRANSACTION DETECTED** (Fraud Probability: {prob:.2%})")
                    
        except ValueError:
            st.error("Invalid Input: Please ensure all values are numbers.")

st.markdown("---")
st.caption("Built with Python, Scikit-Learn, and Streamlit.")