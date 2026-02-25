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

# Storey range midpoints (exact values from training data)
STOREY_OPTIONS = {
    "01 - 03": 2,
    "04 - 05": 3,
    "04 - 06": 5,
    "07 - 09": 8,
    "10 - 12": 11,
    "13 - 13": 13,
    "13 - 15": 14,
    "16 - 18": 17,
    "16 - 20": 18,
    "19 - 21": 20,
    "22 - 24": 23,
    "25 - 27": 26,
    "26 - 30": 28,
    "28 - 30": 29,
    "31 - 33": 32,
    "31 - 35": 33,
    "34 - 36": 35,
    "37 - 39": 38,
    "40 - 42": 41,
    "43 - 45": 44,
    "46 - 48": 47,
    "49 - 51": 50,
}

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
    storey_label = st.selectbox("Storey Range", list(STOREY_OPTIONS.keys()), index=3)
    mid = STOREY_OPTIONS[storey_label]
    flat_model_tier = st.selectbox("Flat Model Tier", FLAT_MODEL_TIERS)

town = st.selectbox("Town", TOWNS)

st.divider()

# ============================================================
# PREDICTION
# ============================================================
def predict_price(floor_area_sqft, remaining_lease_years, mrt_nearest_distance,
                  mid, town, flat_model_tier):
    # Build DataFrame with columns in the SAME ORDER as training data
    input_data = pd.DataFrame([{
        "floor_area_sqft": floor_area_sqft,
        "remaining_lease_years": remaining_lease_years,
        "mrt_nearest_distance": mrt_nearest_distance,
        "mid": mid,
        "town": town,
        "flat_model_tier": flat_model_tier,
        "bto_launched": 19600,
        "mop_flats": 13480,
    }])

    # Convert to category dtype to match training data
    input_data["town"] = input_data["town"].astype("category")
    input_data["mid"] = input_data["mid"].astype("category")
    input_data["flat_model_tier"] = input_data["flat_model_tier"].astype("category")

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
        "Storey Range", "Flat Model Tier", "Town"
    ],
    "Value": [
        floor_area_sqft, remaining_lease_years, mrt_nearest_distance,
        storey_label, flat_model_tier, town
    ],
})
st.table(summary)
