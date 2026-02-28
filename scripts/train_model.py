import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load featured data
df = pd.read_csv("data/featured_sales.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Temporal Split (e.g., last 3 months for testing)
split_date = df['Date'].max() - pd.Timedelta(days=90)
train = df[df['Date'] <= split_date]
test = df[df['Date'] > split_date]

features = ['Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'IsWeekend', 
            'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_30', 
            'Rolling_Mean_7', 'Rolling_Mean_30']
target = 'Sales'

X_train, y_train = train[features], train[target]
X_test, y_test = test[features], test[target]

print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

# 1. Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_preds = rf_model.predict(X_test)

# 2. XGBoost
xgb_model = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42)
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)

# Evaluation
def evaluate(y_true, y_pred, name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"--- {name} Results ---")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}\n")
    return mae, rmse

rf_mae, rf_rmse = evaluate(y_test, rf_preds, "Random Forest")
xgb_mae, xgb_rmse = evaluate(y_test, xgb_preds, "XGBoost")

# Save the best model (using RMSE as criteria)
if rf_rmse < xgb_rmse:
    best_model = rf_model
    print("Saving Random Forest as best model.")
else:
    best_model = xgb_model
    print("Saving XGBoost as best model.")

joblib.dump(best_model, "scripts/best_forecasting_model.pkl")

# Save predictions for visualization later
results = test[['Date', 'Sales']].copy()
results['RF_Preds'] = rf_preds
results['XGB_Preds'] = xgb_preds
results.to_csv("data/forecast_results.csv", index=False)
print("Forecast results saved to data/forecast_results.csv")
