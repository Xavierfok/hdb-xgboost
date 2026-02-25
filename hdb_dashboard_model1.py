import streamlit as st
import pandas as pd
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load("hdb_valuation_model.pkl")

model = load_model()

# Page setup
st.set_page_config(page_title="HDB Resale Price Predictor", page_icon="🏠", layout="centered")

st.title("🏠 HDB Resale Price Predictor")
st.write("Enter flat details below to get an estimated resale price.")

# ============================================================
# CATEGORY LISTS (exact values from training data)
# ============================================================
TOWNS = [
    "ANG MO KIO", "BEDOK", "BISHAN", "BUKIT BATOK", "BUKIT MERAH",
    "BUKIT PANJANG", "BUKIT TIMAH", "CENTRAL AREA", "CHOA CHU KANG",
    "CLEMENTI", "GEYLANG", "HOUGANG", "JURONG EAST", "JURONG WEST",
    "KALLANG/WHAMPOA", "MARINE PARADE", "PASIR RIS", "PUNGGOL",
    "QUEENSTOWN", "SEMBAWANG", "SENGKANG", "SERANGOON", "TAMPINES",
    "TOA PAYOH", "WOODLANDS", "YISHUN",
]

FLAT_MODEL_TIERS = ["core", "executive", "high_end", "legacy", "premium"]

# CategoricalDtype with FULL category lists (must match training data exactly)
TOWN_DTYPE = pd.CategoricalDtype(categories=TOWNS)
TIER_DTYPE = pd.CategoricalDtype(categories=FLAT_MODEL_TIERS)

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
    mrt_nearest_distance = st.number_input(
        "Nearest MRT Distance (m)", min_value=0.0, max_value=5000.0, value=500.0, step=10.0
    )

with col2:
    mid = st.number_input(
        "Storey", min_value=1, max_value=50, value=10, step=1
    )
    flat_model_tier = st.selectbox("Flat Model Tier", FLAT_MODEL_TIERS)

town = st.selectbox("Town", TOWNS)

st.divider()

# ============================================================
# PREDICTION
# ============================================================
def predict_price(floor_area_sqft, remaining_lease_years, mrt_nearest_distance,
                  mid, town, flat_model_tier):
    # Build DataFrame with columns in the SAME ORDER as training data
    input_data = pd.DataFrame({
        "floor_area_sqft": [float(floor_area_sqft)],
        "remaining_lease_years": [int(remaining_lease_years)],
        "mrt_nearest_distance": [float(mrt_nearest_distance)],
        "mid": [int(mid)],
        "town": pd.Categorical([town], dtype=TOWN_DTYPE),
        "flat_model_tier": pd.Categorical([flat_model_tier], dtype=TIER_DTYPE),
        "bto_launched": [19600],
        "mop_flats": [13480],
    })

    log_prediction = model.predict(input_data)[0]
    prediction = np.exp(log_prediction)
    return round(prediction, -2)

# ============================================================
# PREDICTION OUTPUT
# ============================================================
price = predict_price(
    floor_area_sqft, remaining_lease_years, mrt_nearest_distance,
    mid, town, flat_model_tier
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
        "Storey", "Flat Model Tier", "Town"
    ],
    "Value": [
        floor_area_sqft, remaining_lease_years, mrt_nearest_distance,
        mid, flat_model_tier, town
    ],
})
st.table(summary)
