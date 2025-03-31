# User Guide: Content Demand Index

## Introduction

The Content Demand Index is a tool for analyzing the relationship between audience demand metrics and revenue data for television shows. This guide will help you use the system effectively.

## Getting Started

### System Requirements

- Modern web browser (Chrome, Firefox, Edge, or Safari)
- Python 3.7+ (for data processing)
- Required Python packages: pandas, numpy, matplotlib, openpyxl

### First-time Setup

1. Ensure Python and required packages are installed:
   ```bash
   pip install pandas numpy matplotlib openpyxl
   ```

2. Prepare your data file:
   - Create or use an existing Excel file with three sheets:
     - `demand_data`: Containing show demand metrics
     - `revenue_data`: Containing revenue information
     - `show_metadata`: Containing additional show information

3. Place your data file in the `dataraw` folder as `show_analysis.xlsx`

## Processing Your Data

### Running the Data Processor

1. Open a terminal/command prompt
2. Navigate to the scripts directory:
   ```bash
   cd scripts
   ```
3. Run the processing script:
   ```bash
   python process_data.py
   ```
4. The script will:
   - Load your data from `dataraw/show_analysis.xlsx`
   - Process and calculate all metrics
   - Save results to the `dataprocessed` folder

### Understanding the Output

The processor generates three JSON files:
- `valuations.json`: Main show valuations with all metrics
- `lag_results.json`: Time lag correlation data
- `momentum_results.json`: Momentum factors for each show

## Using the Web Interface

### Accessing the Interface

1. Open `web/index.html` in your browser
2. Enter the password when prompted: `DemandAnalytics2025`
3. Choose your preferred interface:
   - Directory View: Minimalist tree-style navigation
   - Dashboard View: Visual cards and tables

### Directory View Features

1. **Expandable Sections**:
   - Click on section names (shows, territories, analytics, valuations) to expand
   - Shows are sorted by valuation amount

2. **Show Details**:
   - Click on any show name to view detailed metrics
   - Back button returns to directory

3. **Correlation Analysis**:
   - Available under the "valuations" section
   - Explains methodology and findings

### Dashboard View Features

1. **Show Cards**:
   - Displays all shows as visual cards
   - Each card shows key metrics and valuation

2. **Table View**:
   - Comprehensive table of all metrics
   - Sortable columns

3. **Methodology Section**:
   - Explains the valuation formula and approach

## Understanding Key Metrics

### Demand Score
The weighted demand metric that accounts for territory importance.

### Momentum Factor
Measures how demand is trending (above 1.0 means increasing demand).

### Optimal Lag
The time delay between demand changes and revenue impact, in months.

### Revenue per Demand
How much revenue each unit of demand typically generates.

### Predicted Valuation
The calculated value using the formula:
```
Value = Current Demand × Revenue/Demand Point × Momentum Factor × 
        Genre Multiplier × Volatility Discount
```

## Troubleshooting

### Processing Script Issues

| Problem | Solution |
|---------|----------|
| Missing data file | Ensure `show_analysis.xlsx` is in the `dataraw` folder |
| Script errors | Check Excel format matches expected structure |
| No output files | Check permissions on `dataprocessed` folder |

### Web Interface Issues

| Problem | Solution |
|---------|----------|
| Password not accepted | Try clearing browser cache or use incognito mode |
| Data not loading | Ensure JSON files exist in `dataprocessed` folder |
| Missing shows | Check that show names match between Excel and JSON |

## Getting Help

For additional assistance, please refer to the project documentation or contact the development team. 