📊 Retail Sales Forecasting App
🚀 Problem Statement

Retail businesses struggle to predict future sales accurately due to fluctuating customer demand, seasonality, and multiple influencing factors such as price and quantity.
Without accurate forecasts, businesses may:

Overestimate sales → leading to overstocking and losses.

Underestimate sales → causing stockouts and missed opportunities.

📌 Goal: Build a Machine Learning powered Sales Forecasting App that predicts future sales using historical sales data.

✅ Solution

This project provides a Streamlit-based forecasting application where users can:

Upload their sales dataset (CSV).

Automatically preprocess and clean the data.

Train a Random Forest Regressor model.

Evaluate performance using MAE & R² score.

Generate 7-day sales forecasts with interactive charts.

🏗️ Project Workflow


graph TD
A[Upload CSV File] --> B[Data Preprocessing]
B --> C[Feature Engineering_lags]
C --> D[Train-Test Split]
D --> E[Train Random Forest Model]
E --> F[Evaluate MAE R2]
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

