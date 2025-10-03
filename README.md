# 📊 Retail Sales Forecasting App

A machine learning project that predicts daily retail sales and provides a **7-day sales forecast** using a **Random Forest Regressor**.  
The app is deployed with **Streamlit** and allows interactive exploration of forecasts.

---

## 📝 Problem Statement
Retail businesses often struggle to forecast future sales accurately due to fluctuating demand and changing buying patterns.  
The objective of this project is to **build a sales forecasting system** that:
- Analyzes historical sales data  
- Learns patterns using machine learning  
- Provides accurate short-term (7-day) forecasts  
- Visualizes actual vs predicted sales interactively  

---

## 🏗️ Project Workflow

```mermaid
graph TD
A[Upload CSV File] --> B[Data Preprocessing]
B --> C[Feature Engineering_lags]
C --> D[Train-Test Split]
D --> E[Train Random Forest Model]
E --> F[Evaluate MAE & R²]
F --> G[Future Forecast 7 Days]
G --> H[Interactive Dashboard in Streamlit]


```markdown
📂 Folder Structure
Retail-Sales-Forecasting/
│
├── data/
│   └── sample_sales_100.csv       # Sample dataset
│
├── scripts/
│   ├── train_model.py             # Model training
│   ├── forecast.py                # Forecasting logic
│   └── utils.py                   # Helper functions
│
├── app/
│   └── app.py                     # Streamlit app
│
├── requirements.txt               # Dependencies
└── README.md                      # Project documentation


📌 Features

1. Upload sales data in CSV format.

2. Perform automatic preprocessing (date conversion, lag features).

3. Train & test ML model with metrics.

4. Predict future 7 days sales.

5. Interactive visualization of Actual vs Predicted & Future Forecast.

🧪 Example Output

Model Performance:

MAE: 592.94

R² Score: 0.81

Forecast Example:
| Date       | Predicted_Sales |
| ---------- | --------------- |
| 2023-04-11 | 555.18          |
| 2023-04-12 | 606.28          |
| 2023-04-13 | 629.02          |

▶️ How to Run

Install dependencies:

pip install -r requirements.txt


Run the Streamlit app:

streamlit run app/app.py


Upload your CSV dataset and view results 🚀

📈 Future Improvements

Add more ML/DL models (XGBoost, LSTM).

Include seasonality & holiday effects.

Deploy on Streamlit Cloud / HuggingFace Spaces / Heroku.

🤝 Contribution

Pull requests are welcome! Feel free to fork and improve this project.

