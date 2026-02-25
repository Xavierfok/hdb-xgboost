import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load("hdb_xgb_model.pkl")

model = load_model()

# Page setup
st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 HDB Resale Price Predictor")
st.write("Enter flat details below to get an estimated resale price.")

# ============================================================
# TOWN LIST (from model's OneHotEncoder categories)
# ============================================================
TOWNS = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
    "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
    "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
    "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
    "TOA PAYOH", "WOODLANDS", "YISHUN",
]

# ============================================================
# INPUT SECTION
# ============================================================
st.header("Input Data")

col1, col2 = st.columns(2)

with col1:
    floor_area_sqft = st.number_input(
        "Floor Area (sqft)", min_value=200.0, max_value=3000.0, value=1000.0, step=10.0
    )
    remaining_lease_years = st.number_input(
        "Remaining Lease (years)", min_value=1, max_value=99, value=75, step=1
    )

with col2:
    mrt_nearest_distance = st.number_input(
        "Nearest MRT Distance (m)", min_value=0.0, max_value=5000.0, value=500.0, step=10.0
    )
    mid = st.number_input(
        "Mid Storey", min_value=1, max_value=50, value=10, step=1
    )

town = st.selectbox("Town", TOWNS)

st.divider()

# ============================================================
# PREDICTION
# ============================================================
def predict_price(floor_area_sqft, remaining_lease_years, mrt_nearest_distance,
                  mid, town):
    input_data = pd.DataFrame([{
        "floor_area_sqft": floor_area_sqft,
        "remaining_lease_years": remaining_lease_years,
        "mrt_nearest_distance": mrt_nearest_distance,
        "mid": mid,
        "bto_launched": 19600,
        "mop_flats": 13480,
        "town": town,
    }])
    prediction = model.predict(input_data)[0]
    return round(prediction, -2)

# ============================================================
# PREDICTION OUTPUT
# ============================================================
price = predict_price(
    floor_area_sqft, remaining_lease_years, mrt_nearest_distance,
    mid, town
)
price_per_sqft = price / floor_area_sqft

st.header("Predicted Price")

col1, col2 = st.columns(2)
col1.metric("Resale Price", f"${price:,.0f}")
col2.metric("Price per sqft", f"${price_per_sqft:,.0f}")

st.divider()

# Summary of inputs
st.subheader("Your Inputs")
summary = pd.DataFrame({
    "Feature": [
        "Floor Area (sqft)", "Remaining Lease (years)", "Nearest MRT (m)",
        "Mid Storey", "Town"
    ],
    "Value": [
        floor_area_sqft, remaining_lease_years, mrt_nearest_distance,
        mid, town
    ],
})
st.table(summary)
