# Sales & Demand Forecasting | Task 1 (Internship)

## Project Overview
This project targets **Sales & Demand Forecasting** for businesses using historical sales data. It leverages Machine Learning models (Random Forest and XGBoost) to predict future demand, enabling informed decision-making for inventory and resource management.

### Key Features
- **Data Cleaning & EDA**: Handling missing values, date formatting, and basic statistical analysis.
- **Feature Engineering**: Creating time-based features (day, month, week) and lag/rolling features for time-series modeling.
- **Multi-Model Approach**: Training and comparing Random Forest and XGBoost regressors.
- **Recursive Forecasting**: Generating 30-day future projections based on historical patterns.
- **Interactive Dashboard**: A high-end Streamlit UI for real-time data exploration and prediction visualization.

## Project Structure
```text
FUTURE_ML_01/
├── data/               # Raw and processed datasets
├── notebooks/          # Exploratory Analysis (Empty)
├── reports/            # Exported charts and insights
├── scripts/            # Core Python logic
│   ├── app.py          # Streamlit Dashboard
│   ├── train_model.py  # Model training pipeline
│   └── ...             # Data and Feature scripts
├── requirements.txt    # Project dependencies
└── README.md           # Documentation
```

## How to Run
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the Dashboard**:
   ```bash
   streamlit run scripts/app.py
   ```

3. **Train / Retrain Model**:
   ```bash
   python scripts/train_model.py
   ```

## Technologies Used
- **Python**: Core language
- **Pandas/Numpy**: Data manipulation
- **Scikit-Learn/XGBoost**: Machine Learning models
- **Streamlit**: Web dashboard
- **Plotly**: Dynamic visualizations
- **Matplotlib/Seaborn**: Static report plots


