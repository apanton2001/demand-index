# Data Processing Documentation

## Overview

The data processing component of the Content Demand Index transforms raw Parrot Analytics demand data and revenue information into actionable insights and valuations. This document explains the processing pipeline and calculations.

## Data Requirements

### Input Data Format

The system expects data in an Excel file (`show_analysis.xlsx`) with these sheets:

1. **demand_data**
   ```
   | show_name | date       | demand_score | territory | source |
   |-----------|------------|--------------|-----------|--------|
   | FRIENDS   | 2023-01-01 | 22.5         | US        | Parrot |
   ```

2. **revenue_data**
   ```
   | show_name | date       | gross_receipts | territory | source      |
   |-----------|------------|----------------|-----------|-------------|
   | FRIENDS   | 2023-01-01 | 8500000        | US        | Distributor |
   ```

3. **show_metadata** (optional)
   ```
   | show_name | genre  | type     | status  |
   |-----------|--------|----------|---------|
   | FRIENDS   | Comedy | Scripted | Library |
   ```

## Processing Pipeline

The data processing follows this sequence:

1. **Data Loading**
   - Reads Excel sheets
   - Validates required columns
   - Checks data integrity

2. **Data Preprocessing**
   - Converts dates to datetime format
   - Handles missing values
   - Normalizes column names

3. **Territory Weighting**
   - Applies importance weights to territories
   - Default weights: US (40%), UK (20%), Canada (15%), etc.
   - Creates weighted demand scores

4. **Time Lag Analysis**
   - For each show, calculates correlation between demand and revenue
   - Tests different lag periods (0-12 months)
   - Identifies optimal lag with highest correlation
   - Produces lag results per show

5. **Momentum Calculation**
   - Compares recent demand (last period) to previous period
   - Calculates momentum factor (>1 means growing demand)
   - Adjusts for seasonal effects

6. **Valuation Generation**
   - Applies the valuation formula
   - Incorporates genre multipliers
   - Adjusts for volatility
   - Produces final valuations

7. **Results Serialization**
   - Formats results as JSON
   - Creates separate files for different metrics
   - Saves to the `dataprocessed` directory

## Key Calculations

### Weighted Demand

```python
def calculate_weighted_demand(row):
    territory = row['territory']
    weight = territory_weights.get(territory, 0.05)
    return row['demand_score'] * weight
```

### Time Lag Correlation

```python
for lag in range(max_lag_months + 1):
    lagged_demand['lag_date'] = lagged_demand['date'] + pd.DateOffset(months=lag)
    merged = pd.merge(lagged_demand, show_revenue, left_on='lag_date', right_on='date')
    corr = merged['weighted_demand'].corr(merged['gross_receipts'])
    if abs(corr) > abs(best_corr):
        best_corr = corr
        best_lag = lag
```

### Momentum Factor

```python
recent = show_data.tail(window)['weighted_demand'].mean()
previous = show_data.iloc[-2*window:-window]['weighted_demand'].mean()
momentum = recent / previous if previous > 0 else 1.0
```

### Valuation Formula

```python
valuation = (current_demand * 
            revenue_per_demand * 
            momentum_factor * 
            genre_mult * 
            volatility_discount)
```

## Output Data

### valuations.json

```json
[
  {
    "show_name": "FRIENDS",
    "current_demand": 21.3,
    "weighted_demand": 21.3,
    "momentum_factor": 1.05,
    "revenue_per_demand": 9760000,
    "genre": "Comedy",
    "genre_multiplier": 1.1,
    "optimal_lag": 2,
    "lag_correlation": 0.89,
    "volatility_discount": 0.97,
    "predicted_valuation": 245678900
  }
]
```

### lag_results.json

```json
{
  "FRIENDS": {
    "optimal_lag": 2,
    "correlation": 0.89
  }
}
```

### momentum_results.json

```json
{
  "FRIENDS": 1.05
}
```

## Performance Considerations

1. **Memory Usage**
   - For large datasets (100+ shows), memory consumption can increase
   - Consider processing in batches for very large datasets

2. **Processing Time**
   - Time lag correlation is the most computationally intensive step
   - Overall processing typically completes in seconds for normal datasets

3. **Error Handling**
   - The script provides basic error handling
   - Missing data for individual shows will not halt overall processing

## Extending the Processing

To add new metrics or calculations:

1. Create a new function in `process_data.py`
2. Add the calculation to the main pipeline
3. Include the new metrics in the output JSON structure

Example for adding a "stability score":

```python
def calculate_stability(demand_data):
    # Calculate demand stability over time
    stability_scores = {}
    for show in demand_data['show_name'].unique():
        show_data = demand_data[demand_data['show_name'] == show]
        stability = 1 - (show_data['demand_score'].std() / show_data['demand_score'].mean())
        stability_scores[show] = max(0, min(1, stability))
    return stability_scores
``` 