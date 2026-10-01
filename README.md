# Stock Price Prediction, Sales Forecasting and Waiter Tips Regression

**Author:** Rabea Akter Shoily (Student ID 905253001) · United International University, School of Business and Economics

This project builds and compares predictive models for three problems on a synthetic dataset: a two-layer **LSTM** neural network for stock prices (one ticker, RETAILCO), **Holt-Winters and SARIMA** time-series models for store sales (East store), and **regression models** for waiter tips (a bill-only baseline, then all features, Ridge, Random Forest and Gradient Boosting). Every model is tested on data it did not train on, and the stock and sales models use a time-based split rather than a random one. The best tips pipeline (Ridge) is saved to a single file and reloaded to predict new bills. The written report is in the `report/` folder.

## Repository structure

```
.
├── data/        predictive_analytics_dataset.xlsx     (3 sheets, see below)
├── notebooks/   Stock_price_prediction_project.ipynb  (full workflow, run top to bottom)
├── models/      tips_pipeline.joblib                  (saved preprocessing + Ridge model)
│                lstm_retailco.keras, lstm_scaler.joblib   (saved LSTM and its price scaler)
├── outputs/     charts (01_ to 07_*.png) and comparison tables (*_model_comparison.csv)
├── report/      project report (PDF)
├── requirements.txt
└── README.md
```

## How to install and run

1. Install Python 3.10 to 3.12 (developed with 3.12).
2. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Start Jupyter, open the notebook and choose **Kernel > Restart & Run All**:
   ```bash
   jupyter notebook notebooks/Stock_price_prediction_project.ipynb
   ```

Notes:
- Keep the folder layout above: the notebook reads `../data` and writes charts, tables and models to `../outputs` and `../models`.
- A fixed random seed (42) is used. The full run takes a few minutes (mostly the LSTM and the grid searches).
- Tested with Python 3.12, TensorFlow 2.21, scikit-learn 1.8, statsmodels 0.15 and pandas 3.0. Other versions should work, but LSTM numbers can differ slightly.

## Datasets (sheets in `data/predictive_analytics_dataset.xlsx`)

| Sheet | Content | Used for |
| --- | --- | --- |
| `Stock_Prices` | 3 tickers, 750 trading days each (2023-01-02 to 2025-11-14): date, ticker, open, high, low, close, volume | LSTM on **RETAILCO** closing prices |
| `Sales_Data` | 4 stores, 730 daily sales each: date, store, sales (trend + weekly seasonality) | Holt-Winters and SARIMA on the **East** store |
| `Waiter_Tips` | 300 restaurant bills: total_bill, tip, sex, smoker, day, time, size | Regression of `tip` |

The data is synthetic and supplied by the instructor.

## Key results

**Stock (RETAILCO, last 150 days):** the LSTM did not beat the naive "tomorrow = today" rule, because test prices rose above the highest training price (125.4 vs 117.4).

| Model | RMSE |
| --- | --- |
| LSTM (30-day window) | 2.744 |
| Naive (previous close) | 1.624 |

**Sales (East store, last 60 days):** Holt-Winters is close to the noise floor (residual std about 23).

| Model | RMSE (units/day) |
| --- | --- |
| Seasonal naive | 60.31 |
| **Holt-Winters** | **24.07** |
| SARIMA(1,1,1)(1,0,1,7) | 44.27 |

**Waiter tips:** refinements improved test R² by only about 0.04, because the total bill dominates tip size.

| Model | Train R² | Test R² | Test RMSE | CV R² |
| --- | --- | --- | --- | --- |
| 1. Baseline (bill only) | 0.463 | 0.503 | 1.024 | 0.464 |
| 2. Linear (all features) | 0.534 | 0.544 | 0.981 | 0.506 |
| 3. **Ridge (tuned), deployed** | 0.523 | 0.539 | 0.986 | 0.506 |
| 4. Random Forest (tuned) | 0.594 | 0.507 | 1.020 | 0.447 |
| 5. Gradient Boosting (tuned) | 0.597 | 0.524 | 1.003 | 0.465 |

Ridge was deployed: it ties the best cross-validated R², shows no overfitting and is easy to explain. All charts and tables are in `outputs/`.

## Using the saved pipelines

**Tips pipeline.** It calls a small helper function, so define `add_per_person` first, then load (run as a script or in a notebook, from the repository root):

```python
import joblib, pandas as pd

def add_per_person(df):                      # needed by the pipeline's feature-engineering step
    df = df.copy()
    df["bill_per_person"] = df["total_bill"] / df["size"]
    return df

pipe = joblib.load("models/tips_pipeline.joblib")
new = pd.DataFrame([{"total_bill": 45.0, "sex": "Male", "smoker": "No", "day": "Sat", "time": "Dinner", "size": 4}])
print(pipe.predict(new).round(2))            # [6.81]
```

**LSTM.** `models/lstm_retailco.keras` loads with `keras.models.load_model(...)`. Scale the last 30 closing prices with `models/lstm_scaler.joblib` first, then inverse-transform the prediction.
