import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Title ---
st.title("🛍️ Feature-Based Sales Prediction App")

# --- Instructions ---
st.markdown("""
### 📌 Instructions
Please upload a CSV file with the following columns:
- **Gender** (Male/Female)
- **Age** (numeric)
- **Product Category** (e.g., Clothing, Electronics, Beauty, etc.)
- **Quantity** (numeric)
- **Price per Unit** (numeric)
- **Total Amount** (numeric → this is the target column)

Example:
| Gender | Age | Product Category | Quantity | Price per Unit | Total Amount |
|--------|-----|------------------|----------|----------------|--------------|
| Male   | 25  | Clothing         | 2        | 500            | 1000         |
| Female | 30  | Electronics      | 1        | 1200           | 1200         |
""")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())

    # --- Preprocessing ---
    features = ["Gender", "Age", "Product Category", "Quantity", "Price per Unit"]
    target = "Total Amount"

    df = df.dropna()
    le_gender = LabelEncoder()
    le_category = LabelEncoder()
    df["Gender"] = le_gender.fit_transform(df["Gender"])
    df["Product Category"] = le_category.fit_transform(df["Product Category"])

    X = df[features]
    y = df[target]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # --- Train Model ---
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- Predictions ---
    y_pred = model.predict(X_test)

    # --- Metrics ---
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    st.subheader("Model Performance")
    st.write(f"**MAE:** {mae:.2f}")
    st.write(f"**R² Score:** {r2:.2f}")

    # --- Single Prediction ---
    st.subheader("🔮 Make a Prediction")
    gender_input = st.selectbox("Gender", le_gender.classes_)
    age_input = st.slider("Age", int(df["Age"].min()), int(df["Age"].max()), 30)
    category_input = st.selectbox("Product Category", le_category.classes_)
    quantity_input = st.number_input("Quantity", min_value=1, max_value=20, value=1)
    price_input = st.number_input("Price per Unit", min_value=1, max_value=10000, value=100)

    gender_val = le_gender.transform([gender_input])[0]
    category_val = le_category.transform([category_input])[0]

    input_features = pd.DataFrame([[gender_val, age_input, category_val, quantity_input, price_input]],
                                  columns=features)

    input_scaled = scaler.transform(input_features)
    prediction = model.predict(input_scaled)[0]

    st.success(f"💰 Predicted Sales Amount: **{prediction:.2f}**")