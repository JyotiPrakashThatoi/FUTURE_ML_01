import pandas as pd
import numpy as np

# Load cleaned data
df = pd.read_csv("data/cleaned_sales.csv")
df['Order Date'] = pd.to_datetime(df['Order Date'])

# Aggregate sales by date
# For forecasting, we usually want daily or weekly totals
daily_sales = df.groupby('Order Date')['Sales'].sum().reset_index()
daily_sales = daily_sales.set_index('Order Date').sort_index()

# Ensure we have all dates in the range (fill missing with 0)
all_dates = pd.date_range(start=daily_sales.index.min(), end=daily_sales.index.max(), freq='D')
daily_sales = daily_sales.reindex(all_dates, fill_value=0)
daily_sales.index.name = 'Date'
daily_sales = daily_sales.reset_index()

# Create time-based features
daily_sales['Month'] = daily_sales['Date'].dt.month
daily_sales['Day'] = daily_sales['Date'].dt.day
daily_sales['DayOfWeek'] = daily_sales['Date'].dt.dayofweek
daily_sales['WeekOfYear'] = daily_sales['Date'].dt.isocalendar().week.astype(int)
daily_sales['IsWeekend'] = daily_sales['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

# Create Lag Features (Sales from previous days)
for i in [1, 7, 14, 30]:
    daily_sales[f'Sales_Lag_{i}'] = daily_sales['Sales'].shift(i)

# Create Rolling Windows (Moving Averages)
for i in [7, 30]:
    daily_sales[f'Rolling_Mean_{i}'] = daily_sales['Sales'].shift(1).rolling(window=i).mean()

# Drop rows with NaN (due to lags/rolling)
daily_sales.dropna(inplace=True)

# Save featured data
featured_path = "data/featured_sales.csv"
daily_sales.to_csv(featured_path, index=False)
print(f"Featured data saved to {featured_path}")
print("\n--- Features Created ---")
print(daily_sales.columns.tolist())
print(f"Total records: {len(daily_sales)}")
