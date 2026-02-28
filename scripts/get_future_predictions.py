import pandas as pd
import numpy as np
import joblib
import os

def generate_future_forecast(days_to_predict=30):
    # 1. Load best model and latest data
    model_path = "scripts/best_forecasting_model.pkl"
    featured_data_path = "data/featured_sales.csv"
    
    if not os.path.exists(model_path) or not os.path.exists(featured_data_path):
        print("Error: Model or featured data not found.")
        return

    model = joblib.load(model_path)
    df = pd.read_csv(featured_data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # We need the most recent window of data to start recursive predictions
    # Lag 30 and Rolling 30 are our largest dependencies
    current_data = df.sort_values('Date').copy()
    
    future_dates = pd.date_range(
        start=current_data['Date'].max() + pd.Timedelta(days=1), 
        periods=days_to_predict, 
        freq='D'
    )
    
    predictions = []
    
    # Recursive prediction loop
    for date in future_dates:
        # Prepare features for the next date
        # 1. Temporal features
        row = {
            'Date': date,
            'Month': date.month,
            'Day': date.day,
            'DayOfWeek': date.dayofweek,
            'WeekOfYear': int(date.isocalendar().week),
            'IsWeekend': 1 if date.dayofweek >= 5 else 0
        }
        
        # 2. Lag features (extract from current_data)
        for lag in [1, 7, 14, 30]:
            target_date = date - pd.Timedelta(days=lag)
            lag_val = current_data[current_data['Date'] == target_date]['Sales']
            row[f'Sales_Lag_{lag}'] = lag_val.values[0] if not lag_val.empty else 0
            
        # 3. Rolling Mean features (extract from current_data)
        for window in [7, 30]:
            # Rolling mean of the LAST 'window' days relative to 'date'
            roll_data = current_data[current_data['Date'] < date].tail(window)
            row[f'Rolling_Mean_{window}'] = roll_data['Sales'].mean() if not roll_data.empty else 0
            
        # Convert to DataFrame for model
        input_features = ['Month', 'Day', 'DayOfWeek', 'WeekOfYear', 'IsWeekend', 
                         'Sales_Lag_1', 'Sales_Lag_7', 'Sales_Lag_14', 'Sales_Lag_30', 
                         'Rolling_Mean_7', 'Rolling_Mean_30']
        
        x_input = pd.DataFrame([row])[input_features]
        pred_sales = model.predict(x_input)[0]
        
        # Save prediction
        row['Sales'] = max(0, pred_sales) # Sales cannot be negative
        predictions.append(row)
        
        # Append to current_data to be used for next iteration's lags/rolling
        current_data = pd.concat([current_data, pd.DataFrame([row])], ignore_index=True)

    # Save results
    forecast_df = pd.DataFrame(predictions)[['Date', 'Sales']]
    forecast_df.to_csv("data/future_forecast.csv", index=False)
    print(f"Generated {days_to_predict} days of future forecast at data/future_forecast.csv")

if __name__ == "__main__":
    generate_future_forecast()
