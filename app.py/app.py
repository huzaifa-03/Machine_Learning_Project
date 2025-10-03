import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

# --- Title ---
st.title("📊 Sales Forecasting App")

# --- Instructions ---
st.markdown("""
### 📌 Instructions
Please upload a CSV file with the following columns:
- **Date** (format: YYYY-MM-DD)
- **Total Amount** (numeric → this is the target column)
- **Quantity** (numeric)
- **Price per Unit** (numeric)

Example:
| Date       | Quantity | Price per Unit | Total Amount |
|------------|----------|----------------|--------------|
| 2023-01-01 |    2     |      500       |    1000      |
| 2023-01-02 |    1     |      700       |     700      |
""")

# --- Upload CSV ---
uploaded_file = st.file_uploader("Upload your dataset (CSV)", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.write("### Raw Data", df.head())

    # --- Preprocessing ---
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")
    df["lag_1"] = df["Total Amount"].shift(1)
    df["lag_7"] = df["Total Amount"].shift(7)
    df = df.dropna()

    # --- Train/Test Split ---
    train_size = int(len(df) * 0.8)
    train = df.iloc[:train_size]
    test = df.iloc[train_size:]

    X_train = train[["lag_1", "lag_7", "Quantity", "Price per Unit"]]
    y_train = train["Total Amount"]
    X_test = test[["lag_1", "lag_7", "Quantity", "Price per Unit"]]
    y_test = test["Total Amount"]

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

    # --- Actual vs Predicted Graph ---
    fig, ax = plt.subplots(figsize=(12,6))
    ax.plot(test["Date"], y_test, label="Actual Sales", marker="o")
    ax.plot(test["Date"], y_pred, label="Predicted Sales", marker="x", linestyle="--")
    ax.set_xlabel("Date")
    ax.set_ylabel("Total Amount")
    ax.set_title("Actual vs Predicted Sales")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    # --- Future Forecast ---
    st.subheader("Future Forecast (Next 7 Days)")
    import numpy as np
    last_row = df.iloc[-1].copy()
    future_preds = []
    future_dates = pd.date_range(start=df["Date"].iloc[-1] + pd.Timedelta(days=1), periods=7)

    for date in future_dates:
        X_future = np.array([[last_row["lag_1"], 
                              last_row.get("lag_7", last_row["lag_1"]), 
                              last_row["Quantity"], 
                              last_row["Price per Unit"]]])
        future_sales = model.predict(X_future)[0]
        future_preds.append((date, future_sales))
        last_row["lag_7"] = last_row["lag_1"]
        last_row["lag_1"] = future_sales

    future_df = pd.DataFrame(future_preds, columns=["Date", "Predicted_Sales"])
    st.write(future_df)

    # --- Future Forecast Graph ---
    fig2, ax2 = plt.subplots(figsize=(12,6))
    ax2.plot(df["Date"], df["Total Amount"], label="History", marker="o")
    ax2.plot(future_df["Date"], future_df["Predicted_Sales"], label="Forecast", marker="x", linestyle="--")
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Total Amount")
    ax2.set_title("Future Sales Forecast")
    ax2.legend()
    ax2.grid(True)
    st.pyplot(fig2)