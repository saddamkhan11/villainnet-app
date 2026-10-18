import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="VillainNet | Hero vs Villain Predictor",
    page_icon="🦹",
    layout="centered"
)

# -----------------------------------------------------------------------------
# 2. Define the Neural Network Architecture
#    (Must match the architecture in class-20.ipynb exactly)
# -----------------------------------------------------------------------------
class VillainNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(26, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        return torch.sigmoid(self.output(x))

# -----------------------------------------------------------------------------
# 3. Cache and Load Artifacts (Model, Scaler, Feature Column Order)
# -----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    # Load Scaler and Feature Column Schema
    scaler = joblib.load("scaler.joblib")
    feature_columns = joblib.load("feature_columns.joblib")

    # Load Model Weights
    model = VillainNet()
    model.load_state_dict(torch.load("villainnet.pth", map_location=torch.device("cpu")))
    model.eval()

    return model, scaler, feature_columns

try:
    model, scaler, feature_columns = load_artifacts()
except FileNotFoundError as e:
    st.error(
        f"Missing artifact: {e.filename}. "
        "Make sure 'villainnet.pth', 'scaler.joblib', and 'feature_columns.joblib' "
        "are placed in the same folder as app.py."
    )
    st.stop()

# -----------------------------------------------------------------------------
# 4. User Interface Form
# -----------------------------------------------------------------------------
st.title("🦹 VillainNet Character Classifier")
st.write(
    "Enter a comic character's characteristics below to predict whether "
    "they are aligned with **Good** (Hero) or **Bad** (Villain)."
)

with st.form("character_form"):
    st.subheader("Character Details")
    
    col1, col2 = st.columns(2)
    
    with col1:
        publisher = st.selectbox("Publisher", ["DC", "MARVEL"])
        sex = st.selectbox("Sex", ["Male", "Female", "Other"])
        identity = st.selectbox("Identity", ["Secret", "Public", "Unknown", "Other"])
        alive = st.selectbox("Status", ["Living", "Deceased", "Other"])

    with col2:
        eye = st.selectbox("Eye Colour", ["Blue", "Brown", "Black", "Green", "Red", "Other"])
        hair = st.selectbox("Hair Colour", ["Black", "Brown", "Blond", "Red", "White", "Other"])
        appearances = st.number_input(
            "Number of Appearances", 
            min_value=0, 
            value=50, 
            step=1,
            help="Raw total appearances in comic history"
        )
        year = st.number_input(
            "Year of First Appearance", 
            min_value=1930, 
            max_value=2026, 
            value=1980, 
            step=1
        )

    submit_button = st.form_submit_button("Predict Alignment")

# -----------------------------------------------------------------------------
# 5. Preprocessing & Inference
# -----------------------------------------------------------------------------
if submit_button:
    # Build categorical raw data
    input_cat = pd.DataFrame([{
        'publisher': publisher,
        'sex': sex,
        'identity': identity,
        'alive': alive,
        'eye': eye,
        'hair': hair
    }])

    # One-hot encode using get_dummies
    encoded_cat = pd.get_dummies(input_cat, dtype=float)

    # Initialize a row of zeros matching the exact 26 training columns
    full_row = pd.DataFrame(0.0, index=[0], columns=feature_columns)

    # Fill in matching dummy columns
    for col in encoded_cat.columns:
        if col in full_row.columns:
            full_row.loc[0, col] = encoded_cat.loc[0, col]

    # Preprocess numerical columns exactly as during training:
    # - appearances: np.log1p (log(1 + x))
    # - year: numeric year value
    full_row['appearances'] = np.log1p(appearances if appearances > 0 else 1)
    full_row['year'] = float(year)

    # Scale values using the saved StandardScaler
    scaled_input = scaler.transform(full_row)

    # Forward pass through PyTorch model
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
    with torch.no_grad():
        bad_prob = model(tensor_input).item()

    good_prob = 1.0 - bad_prob

    # Display prediction
    st.divider()
    if bad_prob >= 0.5:
        st.error(f"### 🦹 Prediction: Villain: {bad_prob * 100:.1f}%")
        st.write(f"Confidence: {bad_prob * 100:.1f}% Villain vs {good_prob * 100:.1f}% Hero")
    else:
        st.success(f"### 🦸 Prediction: Hero: {good_prob * 100:.1f}%")
        st.write(f"Confidence: {good_prob * 100:.1f}% Hero vs {bad_prob * 100:.1f}% Villain")