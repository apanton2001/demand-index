import pandas as pd
import numpy as np
import json
from pathlib import Path
import os
import traceback

def load_data():
    """Load data from the raw data folder"""
    try:
        # Assume the show_analysis.xlsx file has these sheets:
        # - demand_data
        # - revenue_data
        # - show_metadata
        
        # Path relative to the script's parent directory (project root)
        excel_path = Path('dataraw/show_analysis.xlsx') 
        print(f"Attempting to load data from: {excel_path.resolve()}") # Use resolve for absolute path
        
        # Check if the file exists using the resolved path
        if not excel_path.resolve().exists():
            print(f"ERROR: File not found: {excel_path.resolve()}")
            return None, None, None
        
        # List available sheets
        print("Reading Excel file...")
        xls = pd.ExcelFile(excel_path)
        print(f"Available sheets: {xls.sheet_names}")
        
        # Load each sheet
        demand_data = None
        revenue_data = None
        metadata = None
        
        if 'demand_data' in xls.sheet_names:
            demand_data = pd.read_excel(excel_path, sheet_name='demand_data')
            print(f"Loaded demand_data sheet with {len(demand_data)} rows")
            print(f"Columns: {demand_data.columns.tolist()}")
        else:
            print("WARNING: 'demand_data' sheet not found")
        
        if 'revenue_data' in xls.sheet_names:
            revenue_data = pd.read_excel(excel_path, sheet_name='revenue_data')
            print(f"Loaded revenue_data sheet with {len(revenue_data)} rows")
            print(f"Columns: {revenue_data.columns.tolist()}")
        else:
            print("WARNING: 'revenue_data' sheet not found")
        
        # --- DEBUGGING START ---
        if revenue_data is not None and 'show_name' in revenue_data.columns:
            unique_shows_in_revenue = revenue_data['show_name'].unique()
            print(f"DEBUG: Unique shows found in loaded revenue_data: {unique_shows_in_revenue}")
            print(f"DEBUG: Total rows loaded in revenue_data: {len(revenue_data)}")
        elif revenue_data is not None:
            print("DEBUG: revenue_data loaded but 'show_name' column is missing.")
        else:
            print("DEBUG: revenue_data is None after loading attempt.")
        # --- DEBUGGING END ---
        
        if 'show_metadata' in xls.sheet_names:
            metadata = pd.read_excel(excel_path, sheet_name='show_metadata')
            print(f"Loaded show_metadata sheet with {len(metadata)} rows")
            print(f"Columns: {metadata.columns.tolist()}")
        else:
            print("WARNING: 'show_metadata' sheet not found")
        
        return demand_data, revenue_data, metadata
    
    except Exception as e:
        print(f"ERROR loading data: {str(e)}")
        print(traceback.format_exc())
        return None, None, None

def preprocess_data(demand_data, revenue_data):
    """Clean and prepare the data for analysis"""
    print("Preprocessing data...")
    
    if demand_data is None or revenue_data is None:
        print("ERROR: Missing required data for preprocessing")
        return None, None
    
    # Ensure date columns are datetime
    if 'date' in demand_data.columns:
        demand_data['date'] = pd.to_datetime(demand_data['date'])
        print("Converted demand_data dates to datetime")
    else:
        print("WARNING: No 'date' column in demand_data")
    
    if 'date' in revenue_data.columns:
        revenue_data['date'] = pd.to_datetime(revenue_data['date'])
        print("Converted revenue_data dates to datetime")
    else:
        print("WARNING: No 'date' column in revenue_data")
    
    # Check for required columns
    req_demand_cols = ['show_name', 'date', 'demand_score']
    req_revenue_cols = ['show_name', 'date', 'gross_receipts']
    
    missing_demand = [col for col in req_demand_cols if col not in demand_data.columns]
    missing_revenue = [col for col in req_revenue_cols if col not in revenue_data.columns]
    
    if missing_demand:
        print(f"WARNING: Missing required columns in demand_data: {missing_demand}")
    
    if missing_revenue:
        print(f"WARNING: Missing required columns in revenue_data: {missing_revenue}")
    
    return demand_data, revenue_data

def calculate_territory_weights(demand_data):
    """Calculate territory weights based on importance"""
    print("Calculating territory weights...")
    
    if 'territory' not in demand_data.columns:
        print("No territory column in demand data, using default weights")
        return {}
    
    # Default territory weights
    territory_weights = {
        'US': 0.4,
        'UK': 0.2,
        'Canada': 0.15,
        'Australia': 0.15,
        'France': 0.1,
    }
    
    # Print available territories
    available_territories = demand_data['territory'].unique()
    print(f"Available territories: {available_territories}")
    
    return territory_weights

def calculate_weighted_demand(demand_data, territory_weights):
    """Calculate territory-weighted demand for each show"""
    print("Calculating weighted demand...")
    
    if demand_data is None:
        print("ERROR: No demand data available")
        return None
    
    # Copy the dataframe
    weighted_data = demand_data.copy()
    
    # Apply territory weighting if possible
    if 'territory' in weighted_data.columns and 'demand_score' in weighted_data.columns:
        def apply_weight(row):
            territory = row.get('territory')
            weight = territory_weights.get(territory, 0.05)  # Default weight for unlisted territories
            return row['demand_score'] * weight
        
        weighted_data['weighted_demand'] = weighted_data.apply(apply_weight, axis=1)
        print("Applied territory weights to demand scores")
    else:
        # If no territory data, just use raw demand
        weighted_data['weighted_demand'] = weighted_data['demand_score']
        print("No territory column, using raw demand scores")
    
    return weighted_data

def calculate_time_lag(demand_data, revenue_data, max_lag_months=12):
    """
    Calculate the optimal time lag between quarterly demand and quarterly revenue for each show
    Returns a dictionary with show_name: {optimal_lag_quarters, correlation} pairs
    """
    print("Calculating time lag correlation (Quarterly)...")
    
    if demand_data is None or revenue_data is None:
        print("ERROR: Missing required data for time lag calculation")
        return {}
    
    # Ensure date columns are present and datetime type
    if 'date' not in demand_data.columns or 'date' not in revenue_data.columns:
        print("ERROR: Missing 'date' column in demand or revenue data")
        return {}
    if not pd.api.types.is_datetime64_any_dtype(demand_data['date']):
         demand_data['date'] = pd.to_datetime(demand_data['date'])
         print("Converted demand_data dates to datetime in time_lag")
    if not pd.api.types.is_datetime64_any_dtype(revenue_data['date']):
        revenue_data['date'] = pd.to_datetime(revenue_data['date'])
        print("Converted revenue_data dates to datetime in time_lag")
        
    # Get unique shows
    if 'show_name' in demand_data.columns:
        shows = demand_data['show_name'].unique()
        print(f"Analyzing time lag for {len(shows)} shows")
    else:
        print("No show_name column in demand data")
        return {}
    
    lag_results = {}
    max_lag_quarters = max_lag_months // 3 # Convert max lag to quarters
    
    for show in shows:
        show_demand = demand_data[demand_data['show_name'] == show].set_index('date')
        show_revenue = revenue_data[revenue_data['show_name'] == show].set_index('date')
        
        # --- DEBUGGING START ---
        print(f"DEBUG: Processing show '{show}' in calculate_time_lag.")
        print(f"DEBUG: Shape of filtered show_revenue: {show_revenue.shape}")
        # --- DEBUGGING END ---
        
        print(f"- {show}: {len(show_demand)} monthly demand points, {len(show_revenue)} revenue points")
        
        # Resample demand to quarterly frequency (using mean)
        # Use 'QE' for calendar quarter end frequency (replaces deprecated 'Q')
        quarterly_demand = show_demand[['weighted_demand']].resample('QE').mean()
        
        # Resample revenue to quarterly frequency (using sum)
        # Assuming revenue is reported quarterly total
        quarterly_revenue = show_revenue[['gross_receipts']].resample('QE').sum()

        print(f"  Resampled to {len(quarterly_demand)} quarterly demand points, {len(quarterly_revenue)} quarterly revenue points")
        
        best_lag_q = 0
        best_corr = 0.0 # Initialize correlation as float
        
        # Try different lag periods in quarters
        for lag_q in range(max_lag_quarters + 1):
            # Skip if we don't have enough quarterly data
            if len(quarterly_demand) < 2 or len(quarterly_revenue) < 2:
                print(f"  Insufficient quarterly data for {show}, using default lag of 0")
                break # No point checking further lags if data is insufficient

            # Create lagged quarterly demand data
            lagged_q_demand = quarterly_demand.shift(lag_q)
            
            # Merge lagged demand with revenue on the quarterly index
            merged_q = pd.merge(
                lagged_q_demand,
                quarterly_revenue,
                left_index=True,
                right_index=True,
                how='inner' # Only keep quarters with both demand and revenue
            )
            
            # Calculate correlation if we have enough merged data points (e.g., >= 3 quarters)
            if len(merged_q) >= 3 and 'weighted_demand' in merged_q.columns and 'gross_receipts' in merged_q.columns:
                # Drop rows with NaN values that might result from resampling or merging
                merged_q.dropna(inplace=True)
                if len(merged_q) >= 3:
                    try:
                        corr = merged_q['weighted_demand'].corr(merged_q['gross_receipts'])
                        # Handle potential NaN correlation if variance is zero
                        if pd.isna(corr):
                            corr = 0.0
                            
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_lag_q = lag_q
                    except Exception as e:
                         print(f"  Could not calculate correlation for lag {lag_q}: {e}")
                         corr = 0.0
                else:
                     print(f"  Not enough non-NaN matching quarterly data points after dropna for lag {lag_q}")

            else:
                 print(f"  Not enough matching quarterly data points for lag {lag_q}")

        lag_results[show] = {
            'optimal_lag_quarters': best_lag_q, # Store lag in quarters
            'correlation': float(best_corr) # Ensure correlation is a standard float
        }
        print(f"  Optimal quarterly lag for {show}: {best_lag_q} quarters (r = {best_corr:.2f})")
    
    return lag_results

def calculate_momentum(demand_data, window=90):
    """
    Calculate momentum factor for each show
    Compares recent demand to previous period
    """
    print("Calculating momentum factors...")
    
    if demand_data is None:
        print("ERROR: No demand data available")
        return {}
    
    # Get unique shows
    if 'show_name' not in demand_data.columns or 'date' not in demand_data.columns:
        print("Missing required columns for momentum calculation")
        return {}
    
    shows = demand_data['show_name'].unique()
    momentum_results = {}
    
    for show in shows:
        show_data = demand_data[demand_data['show_name'] == show].sort_values('date')
        
        if len(show_data) < window * 2:
            # Not enough data for momentum calculation
            momentum_results[show] = 1.0
            print(f"- {show}: Insufficient data for momentum, using default of 1.0")
            continue
        
        # Calculate recent vs previous demand
        recent = show_data.tail(window)['weighted_demand'].mean()
        previous = show_data.iloc[-2*window:-window]['weighted_demand'].mean()
        
        # Calculate momentum factor
        if previous > 0:
            momentum = recent / previous
        else:
            momentum = 1.0
            
        momentum_results[show] = momentum
        print(f"- {show}: Momentum factor = {momentum:.2f}")
    
    return momentum_results

def calculate_valuations(demand_data, lag_results, momentum_results, revenue_data):
    """
    Calculate final valuations for each show using the formula:
    Value = Current Monthly Demand × Quarterly Revenue/Quarterly Demand Point × Momentum Factor × 
            Genre Multiplier × Volatility Discount
    """
    print("Calculating final valuations (using quarterly rev/demand ratio)...")
    
    if demand_data is None or revenue_data is None: # Check revenue_data as well
        print("ERROR: No demand or revenue data available")
        return []
        
    # Ensure date columns are present and datetime type
    if 'date' not in demand_data.columns or 'date' not in revenue_data.columns:
        print("ERROR: Missing 'date' column in demand or revenue data")
        return []
    if not pd.api.types.is_datetime64_any_dtype(demand_data['date']):
         demand_data['date'] = pd.to_datetime(demand_data['date'])
         print("Converted demand_data dates to datetime in valuations")
    if not pd.api.types.is_datetime64_any_dtype(revenue_data['date']):
        revenue_data['date'] = pd.to_datetime(revenue_data['date'])
        print("Converted revenue_data dates to datetime in valuations")
    
    # Default genre multipliers
    genre_multipliers = {
        'Drama': 1.2,
        'Comedy': 1.1,
        'Crime': 1.3,
        'Medical': 1.25,
        'Default': 1.0
    }
    
    # Default genre assignments
    default_genres = {
        'CRIMINAL MINDS': 'Crime',
        'GREY\'S ANATOMY': 'Medical',
        'FRIENDS': 'Comedy',
        'BIG BANG THEORY': 'Comedy',
        'MODERN FAMILY': 'Comedy',
        'NCIS': 'Crime',
        'HOUSE': 'Medical',
        'BONES': 'Crime',
        'TWO AND A HALF MEN': 'Comedy',
        'THE KING OF QUEENS': 'Comedy',
        'FRASIER': 'Comedy',
        'WILL AND GRACE': 'Comedy',
        'HOW I MET YOUR MOTHER': 'Comedy',
        'ACCORDING TO JIM': 'Comedy',
        'SCRUBS': 'Medical'
    }
    
    # Get unique shows
    shows = demand_data['show_name'].unique()
    
    # Calculate revenue per demand point using QUARTERLY averages
    revenue_per_demand = {}
    for show in shows:
        show_demand = demand_data[demand_data['show_name'] == show].set_index('date')
        show_revenue = revenue_data[revenue_data['show_name'] == show].set_index('date')
        
        # --- DEBUGGING START ---
        print(f"DEBUG: Processing show '{show}' in calculate_valuations (for rev/demand).")
        print(f"DEBUG: Shape of filtered show_revenue: {show_revenue.shape}")
        # --- DEBUGGING END ---
        
        if not show_demand.empty and not show_revenue.empty:
             # Resample demand to quarterly frequency (using mean)
            quarterly_demand = show_demand[['weighted_demand']].resample('QE').mean()
            
            # Resample revenue to quarterly frequency (using sum)
            quarterly_revenue = show_revenue[['gross_receipts']].resample('QE').sum()

            # Merge quarterly data
            merged_q = pd.merge(
                quarterly_demand,
                quarterly_revenue,
                left_index=True,
                right_index=True,
                how='inner'
            ).dropna() # Drop quarters missing either value
            
            if not merged_q.empty:
                avg_q_demand = merged_q['weighted_demand'].mean()
                avg_q_revenue = merged_q['gross_receipts'].mean()
                
                # Calculate Revenue per Quarterly Demand point
                if avg_q_demand > 0:
                    revenue_per_demand[show] = avg_q_revenue / avg_q_demand
                else:
                    revenue_per_demand[show] = 0
            else:
                 revenue_per_demand[show] = 0
                 print(f"- {show}: No overlapping quarterly data found for rev/demand calculation")

        else:
            revenue_per_demand[show] = 0
            print(f"- {show}: Missing demand or revenue data for rev/demand calculation")
    
    # Calculate valuations
    valuations = []
    for show in shows:
        # Get latest MONTHLY demand
        latest_demand_rows = demand_data[demand_data['show_name'] == show].sort_values('date')
        
        if not latest_demand_rows.empty:
            current_demand = latest_demand_rows['weighted_demand'].iloc[-1]
            
            # Get momentum (calculated based on monthly demand, which is fine)
            momentum = momentum_results.get(show, 1.0)
            
            # Get QUARTERLY revenue per demand point
            rev_per_demand = revenue_per_demand.get(show, 0)
            
            # Get genre multiplier
            genre = default_genres.get(show.upper(), 'Default') # Use upper for lookup consistency
            genre_mult = genre_multipliers.get(genre, genre_multipliers['Default'])
            
            # Get lag information (now contains quarterly lag and correlation)
            lag_info = lag_results.get(show, {'optimal_lag_quarters': 0, 'correlation': 0.0})
            
            # Volatility discount (using correlation from quarterly analysis)
            # Ensure correlation is treated as float
            lag_correlation = float(lag_info.get('correlation', 0.0))
            volatility_discount = 0.9 + (abs(lag_correlation) * 0.1)
            
            # Calculate valuation
            valuation = current_demand * rev_per_demand * momentum * genre_mult * volatility_discount
            
            print(f"- {show}: ${valuation/1000000:.2f}M (Quarterly Rev/Demand Ratio)")
            print(f"  * Current Monthly Demand: {current_demand:.2f}")
            print(f"  * Quarterly Revenue/Demand: ${rev_per_demand/1000000:.2f}M")
            print(f"  * Momentum: {momentum:.2f}x")
            print(f"  * Genre ({genre}): {genre_mult:.2f}x")
            print(f"  * Volatility (Q Lag Corr: {lag_correlation:.2f}): {volatility_discount:.2f}")
            
            valuations.append({
                'show_name': show,
                'current_demand': current_demand, # Latest monthly demand
                'weighted_demand': current_demand, # Using monthly for consistency here, though ratio uses quarterly
                'momentum_factor': momentum,
                'revenue_per_demand': rev_per_demand, # Quarterly based ratio
                'genre': genre,
                'genre_multiplier': genre_mult,
                'optimal_lag_quarters': lag_info['optimal_lag_quarters'], # Lag in quarters
                'lag_correlation': lag_correlation, # Quarterly correlation
                'volatility_discount': volatility_discount,
                'predicted_valuation': valuation
            })
        else:
             print(f"- {show}: No demand data found for final valuation.")
    
    return valuations

def save_results(valuations, lag_results, momentum_results):
    """Save processed results to the processed data folder within the project"""
    print("Saving results...")
    
    try:
        # Define output directory relative to the script's location (scripts/../dataprocessed -> dataprocessed/)
        output_dir = Path('dataprocessed')
        output_dir.mkdir(parents=True, exist_ok=True) # Use Path object's mkdir
        print(f"Output directory: {output_dir.resolve()}")
        
        # Convert lag_results to a serializable format
        serializable_lag_results = {}
        for show, data in lag_results.items():
            serializable_lag_results[show] = {
                'optimal_lag_quarters': data['optimal_lag_quarters'], # Key updated
                'correlation': float(data['correlation']) 
            }

        # Convert momentum results (assuming it's {show: value})
        serializable_momentum_results = {k: float(v) for k, v in momentum_results.items()}

        # Save valuations
        valuations_path = output_dir / 'valuations.json'
        with open(valuations_path, 'w') as f:
            json.dump(valuations, f, indent=2)
            print(f"Saved {valuations_path.name} with {len(valuations)} shows")
        
        # Save lag results
        lag_path = output_dir / 'lag_results.json'
        with open(lag_path, 'w') as f:
            json.dump(serializable_lag_results, f, indent=2)
        print(f"Saved {lag_path.name} with {len(lag_results)} shows")
        
        # Save momentum results
        momentum_path = output_dir / 'momentum_results.json'
        with open(momentum_path, 'w') as f:
            json.dump(serializable_momentum_results, f, indent=2)
        print(f"Saved {momentum_path.name} with {len(momentum_results)} shows")
        
    except Exception as e:
        print(f"ERROR saving results: {str(e)}")
        print(traceback.format_exc())

def main():
    print("=" * 60)
    print("CONTENT DEMAND INDEX - DATA PROCESSOR")
    print("=" * 60)
    print()
    
    try:
        # Load data
        demand_data, revenue_data, metadata = load_data()
        
        if demand_data is None or revenue_data is None:
            print("ERROR: Critical data missing, cannot proceed")
            return
        
        # Preprocess data
        demand_data, revenue_data = preprocess_data(demand_data, revenue_data)
        
        # Calculate territory weights
        territory_weights = calculate_territory_weights(demand_data)
        
        # Calculate weighted demand
        weighted_demand_data = calculate_weighted_demand(demand_data, territory_weights)
        
        # Calculate time lag
        lag_results = calculate_time_lag(weighted_demand_data, revenue_data)
        
        # Calculate momentum
        momentum_results = calculate_momentum(weighted_demand_data)
        
        # Calculate valuations
        valuations = calculate_valuations(weighted_demand_data, lag_results, momentum_results, revenue_data)
        
        # Save results
        save_results(valuations, lag_results, momentum_results)
        
        print("\nData processing complete!")
        print("=" * 60)
        print(f"Generated valuations for {len(valuations)} shows")
        print(f"Results saved to: {os.path.abspath('../dataprocessed')}")
        print("=" * 60)
        
    except Exception as e:
        print(f"ERROR during processing: {str(e)}")
        print(traceback.format_exc())

if __name__ == "__main__":
    main() 