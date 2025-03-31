# Technical Architecture

## Overview

The Content Demand Index system is based on a simple but effective architecture that focuses on:

1. Data processing in Python
2. Static web interface for viewing results
3. File-based storage for simplicity

## Architecture Diagram

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│  Excel Source   │────▶│  Python Script  │────▶│  JSON Output    │
│  (Raw Data)     │     │  (Processing)   │     │  (Processed)    │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └────────┬────────┘
                                                         │
                                                         ▼
                                                ┌─────────────────┐
                                                │                 │
                                                │  Web Interface  │
                                                │  (HTML/JS)      │
                                                │                 │
                                                └─────────────────┘
```

## Component Details

### Data Storage

1. **Input Data**
   - Format: Excel spreadsheet (.xlsx)
   - Location: `dataraw/show_analysis.xlsx`
   - Structure:
     - `demand_data` sheet: show_name, date, demand_score, territory, source
     - `revenue_data` sheet: show_name, date, gross_receipts, territory, source
     - `show_metadata` sheet: Additional metadata about shows

2. **Output Data**
   - Format: JSON files
   - Location: `dataprocessed/` directory
   - Files:
     - `valuations.json`: Main calculated show valuations
     - `lag_results.json`: Time lag correlation findings
     - `momentum_results.json`: Show momentum factors

### Data Processing

1. **Python Scripts**
   - Main script: `scripts/process_data.py`
   - Dependencies: pandas, numpy, matplotlib, openpyxl
   - Functions:
     - Data loading
     - Demand weighting
     - Time lag calculation
     - Momentum calculation
     - Valuation formula application

2. **Processing Pipeline**
   ```
   Load Excel Data → Preprocess → Calculate Weights → Apply Weights →
   Calculate Time Lag → Calculate Momentum → Generate Valuations → Save Results
   ```

### Web Interface

1. **Components**
   - Static HTML pages with JavaScript
   - Bootstrap CSS for styling
   - Client-side password protection

2. **Interface Types**
   - Directory-style UI (monospace, expandable tree)
   - Card-based dashboard
   - Detailed show pages
   - Correlation analysis documentation

3. **Data Flow**
   ```
   Load Page → Password Check → Fetch JSON Data → Render UI → Handle User Interaction
   ```

## Security Considerations

1. **Access Control**
   - Basic password protection
   - Password: "DemandAnalytics2025"
   - Storage: Browser localStorage
   - Limitations: Client-side only, not secure for sensitive data

2. **Data Privacy**
   - No PII or sensitive business information
   - All data accessible to anyone with the password

## Scalability Considerations

1. **Current Limitations**
   - File-based storage limits data volume
   - Manual processing step required
   - No real-time updates

2. **Scaling Options**
   - Database integration for larger datasets
   - Automated data pipeline for regular updates
   - API integration for real-time data access
   - Server-side processing for more complex calculations

## Future Architecture Possibilities

1. **API-Driven Architecture**
   ```
   Parrot API → Processing Service → Database → REST API → Web App
   ```

2. **Real-time Dashboard**
   ```
   Data Sources → ETL Pipeline → Time-series DB → WebSocket API → React Dashboard
   ```

3. **Enterprise Integration**
   ```
   Multiple Data Sources → Data Lake → Processing Pipeline → BI Tools → Executive Dashboards
   ``` 