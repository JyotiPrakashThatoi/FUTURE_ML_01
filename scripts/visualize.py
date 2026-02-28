import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
df = pd.read_csv("data/forecast_results.csv")
df['Date'] = pd.to_datetime(df['Date'])

# Plotting Actual vs Predicted
plt.figure(figsize=(14, 7))
sns.lineplot(data=df, x='Date', y='Sales', label='Actual Sales', color='blue', linewidth=2)
sns.lineplot(data=df, x='Date', y='RF_Preds', label='Predicted (Random Forest)', color='red', linestyle='--')

plt.title('Sales Forecast vs Actual (Last 90 Days)', fontsize=16)
plt.xlabel('Date', fontsize=12)
plt.ylabel('Sales ($)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)

# Save the plot
plt.savefig('reports/forecast_vs_actual.png')
print("Visualization saved to reports/forecast_vs_actual.png")

# Error Distribution
df['Error'] = df['Sales'] - df['RF_Preds']
plt.figure(figsize=(10, 5))
sns.histplot(df['Error'], kde=True, color='purple')
plt.title('Forecast Error Distribution (Residuals)', fontsize=14)
plt.xlabel('Error Weight', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.savefig('reports/error_distribution.png')
print("Error distribution saved to reports/error_distribution.png")
