# Content Demand Index Project - Complete Knowledge Transfer

## 1. COMPLETE PROJECT SUMMARY

### Current Phase and Status
- **Phase**: Completion (MVP fully functional)
- **Progress**: 25/50 tasks completed
- **Current Status**: Core functionality working with web interface implemented

### Implemented Features
- Data processing pipeline for Parrot Analytics demand data
- Time lag correlation analysis between demand and revenue
- Territory-weighted demand calculation
- Momentum factor computation
- Genre-specific multipliers
- Revenue-per-demand-point calculation
- Valuation formula application
- Interactive web interfaces:
  - Directory-style UI (matching requested format)
  - Dashboard view with cards and tables
  - Show detail pages
  - Correlation analysis page

### Pending Tasks
- GitHub Pages deployment
- API integration with Parrot Analytics (currently using exported data)
- Additional visualizations (charts, graphs)
- User management beyond basic password protection
- Expanding territory analysis capabilities

### Known Issues/Bugs
- Back navigation requires manual implementation (fixed for show pages)
- Some dummy links in directory UI don't lead to actual content
- Time lag correlation calculation defaults to 0 with limited data points

### Technical Debt
- Hardcoded genre assignments and multipliers
- Limited error handling in data processing scripts
- No logging system implemented
- Password stored in localStorage (not secure for production)
- No unit tests or automated testing

### APIs/Dependencies Used
- **Python Libraries**: pandas, numpy, matplotlib, openpyxl
- **Frontend Libraries**: Bootstrap CSS
- **Data Sources**: Parrot Analytics data (via Excel export)

## 2. CODEBASE OVERVIEW

### File Structure
```
demand_index/
├── dataraw/
│   └── show_analysis.xlsx    # Raw data with demand_data, revenue_data sheets
├── dataprocessed/
│   ├── valuations.json       # Calculated show valuations
│   ├── lag_results.json      # Time lag correlation data
│   └── momentum_results.json # Momentum calculations
├── scripts/
│   └── process_data.py       # Main data processing script
└── web/
    ├── index.html            # Landing page
    ├── directory.html        # Directory-style interface
    ├── dashboard.html        # Card/table dashboard
    ├── show.html             # Individual show details
    └── correlation.html      # Correlation analysis explanation
```

### Key Components

#### Data Processing (process_data.py)
- Loads raw Excel data from `dataraw/show_analysis.xlsx`
- Calculates weighted demand based on territory importance
- Computes time lag correlations between demand and revenue
- Determines momentum factors for each show
- Applies valuation formula and genre multipliers
- Outputs JSON files to `dataprocessed/` folder

#### Web Interfaces
1. **Directory View**: Monospace font, expandable tree-style navigation
2. **Dashboard View**: Card-based UI with Bootstrap styling
3. **Show Details**: Individual metrics for each show
4. **Correlation Analysis**: Explanation of methodology and findings

### Important Functions

#### Data Processing
- `load_data()`: Loads Excel sheets from raw data
- `calculate_weighted_demand()`: Applies territory weights to demand scores
- `calculate_time_lag()`: Determines optimal lag between demand and revenue
- `calculate_momentum()`: Computes momentum factors from historical data
- `calculate_valuations()`: Applies the valuation formula to all shows
- `save_results()`: Outputs JSON files for web consumption

#### Web Interface
- `checkPassword()`: Validates access code
- `loadData()`, `loadShowData()`, `loadCorrelationData()`: Fetch and display data
- Event listeners for directory tree expansion

### Database Schema
No database is used. All data is stored in:
- Input: Excel files (`show_analysis.xlsx`)
- Output: JSON files (`valuations.json`, `lag_results.json`, `momentum_results.json`)

### Environment Variables
None required. The project uses file paths relative to script locations.

## 3. DEVELOPMENT CONTEXT

### Architectural Decisions

1. **File-Based Storage vs. Database**
   - **Decision**: Use file-based storage (Excel input, JSON output)
   - **Rationale**: Simplicity, ease of deployment, no server requirements

2. **Static Web Pages vs. Web App Framework**
   - **Decision**: Use static HTML/JS pages with client-side rendering
   - **Rationale**: No build process needed, instant deployment

3. **Monospace Directory UI vs. Rich Dashboard**
   - **Decision**: Implement both, with directory as primary
   - **Rationale**: Directory UI matches requested style; dashboard provides richer visualization

4. **Password Protection Method**
   - **Decision**: Use basic localStorage for password retention
   - **Rationale**: Simple implementation for MVP, adequate for internal tool

### Trade-offs Chosen

1. **Simplicity vs. Robustness**
   - **Trade-off**: Favored simplicity over error handling and edge cases
   - **Impact**: Better for demonstration but requires hardening for production

2. **Manual Processing vs. Automated Pipeline**
   - **Trade-off**: Manual script execution vs. automated data pipeline
   - **Impact**: Requires manual intervention but simpler implementation

3. **Static Data vs. Real-time API**
   - **Trade-off**: Static data files vs. real-time API integration
   - **Impact**: Cannot automatically update with new Parrot data

### Rejected Alternatives

1. **React/Vue Frontend**
   - **Why Rejected**: Added complexity for simple UI requirements
   - **Potential Benefit**: Component reuse, state management

2. **Server-side Processing**
   - **Why Rejected**: Deployment complexity, hosting requirements
   - **Potential Benefit**: Better security, centralized processing

3. **Database Backend**
   - **Why Rejected**: Overkill for current data volume and requirements
   - **Potential Benefit**: Easier querying, better data relationships

### Performance Considerations

1. **Data Processing**
   - Pre-computation of all metrics rather than on-demand calculation
   - All heavy processing done once, results cached in JSON

2. **Web Interface**
   - Minimal dependencies (only Bootstrap CSS)
   - Asynchronous data loading to prevent UI blocking

### Security Measures

1. **Access Control**
   - Basic password protection (DemandAnalytics2025)
   - Password stored in localStorage (not secure for sensitive data)

2. **Data Protection**
   - No PII or sensitive business data in the system
   - All data publicly accessible to anyone with the password

## 4. MIGRATION CHECKLIST

### Critical Files to Transfer

1. **Data Processing**
   - `scripts/process_data.py`: Main data processing script

2. **Data Files**
   - `dataraw/show_analysis.xlsx`: Raw input data
   - `dataprocessed/*.json`: Generated output files

3. **Web Interface**
   - `web/index.html`: Landing page
   - `web/directory.html`: Directory interface
   - `web/dashboard.html`: Dashboard interface
   - `web/show.html`: Show detail page
   - `web/correlation.html`: Correlation analysis page

### Environment Setup Steps

1. **Python Environment**
   ```bash
   # Create and activate virtual environment (optional but recommended)
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   
   # Install required packages
   pip install pandas numpy matplotlib openpyxl
   ```

2. **Project Structure**
   ```bash
   # Create required directories
   mkdir -p demand_index/{dataraw,dataprocessed,scripts,web}
   ```

### Configuration Changes Needed

1. **Data File Paths**
   - If folder structure changes, update paths in `process_data.py`:
   ```python
   excel_path = Path('../dataraw/show_analysis.xlsx')  # Adjust as needed
   ```

2. **Web File Paths**
   - If hosting on a subdirectory, update fetch paths in HTML files:
   ```javascript
   const response = await fetch('../dataprocessed/valuations.json');  # Adjust as needed
   ```

### Testing Requirements

1. **Data Processing Validation**
   - Run `process_data.py` and verify JSON output files are created
   - Check valuations match expected calculations

2. **Web Interface Testing**
   - Test all navigation links
   - Verify show data displays correctly
   - Ensure password protection works
   - Test directory tree expand/collapse

3. **Cross-browser Testing**
   - Test in Chrome, Firefox, Safari, and Edge
   - Verify responsive design on mobile devices

### Deployment Considerations

1. **GitHub Pages Deployment**
   ```bash
   # Push to GitHub
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/YOUR-USERNAME/demand-index.git
   git push -u origin main
   
   # Enable GitHub Pages in repository settings
   # Set source to main branch, /web folder
   ```

2. **Alternative Static Hosting**
   - Copy the `web/` directory to any static hosting service
   - Ensure `dataprocessed/` folder is also copied to maintain relative paths

3. **Local Usage**
   - Run data processing: `python scripts/process_data.py`
   - Open `web/index.html` in a browser

4. **Password Updates**
   - Change password in all HTML files if needed:
   ```javascript
   if (password !== 'DemandAnalytics2025') {  # Update this value
   ``` 