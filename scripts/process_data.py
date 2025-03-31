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
        
        excel_path = Path('../dataraw/show_analysis.xlsx')
        print(f"Attempting to load data from: {excel_path.absolute()}")
        
        if not excel_path.exists():
            print(f"ERROR: File not found: {excel_path.absolute()}")
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
    Calculate the optimal time lag between demand and revenue for each show
    Returns a dictionary with show_name: optimal_lag pairs
    """
    print("Calculating time lag correlation...")
    
    if demand_data is None or revenue_data is None:
        print("ERROR: Missing required data for time lag calculation")
        return {}
    
    # Get unique shows
    if 'show_name' in demand_data.columns:
        shows = demand_data['show_name'].unique()
        print(f"Analyzing time lag for {len(shows)} shows")
    else:
        print("No show_name column in demand data")
        return {}
    
    lag_results = {}
    
    for show in shows:
        show_demand = demand_data[demand_data['show_name'] == show]
        show_revenue = revenue_data[revenue_data['show_name'] == show]
        
        print(f"- {show}: {len(show_demand)} demand points, {len(show_revenue)} revenue points")
        
        best_lag = 0
        best_corr = 0
        
        # Try different lag periods
        for lag in range(max_lag_months + 1):
            # Skip if we don't have enough data
            if len(show_demand) < 5 or len(show_revenue) < 5:
                print(f"  Insufficient data for {show}, using default lag of 0")
                continue
            
            # Create lagged demand data
            lagged_demand = show_demand.copy()
            if 'date' in lagged_demand.columns:
                lagged_demand['lag_date'] = lagged_demand['date'] + pd.DateOffset(months=lag)
                
                # Merge with revenue on date
                if 'date' in show_revenue.columns:
                    merged = pd.merge(
                        lagged_demand,
                        show_revenue,
                        left_on='lag_date',
                        right_on='date',
                        suffixes=('_demand', '_revenue')
                    )
                    
                    # Calculate correlation if we have enough data points
                    if len(merged) >= 5 and 'weighted_demand' in merged.columns and 'gross_receipts' in merged.columns:
                        corr = merged['weighted_demand'].corr(merged['gross_receipts'])
                        if abs(corr) > abs(best_corr):
                            best_corr = corr
                            best_lag = lag
                    else:
                        print(f"  Not enough matching data points for lag {lag}")
        
        lag_results[show] = {
            'optimal_lag': best_lag,
            'correlation': best_corr
        }
        print(f"  Optimal lag for {show}: {best_lag} months (r = {best_corr:.2f})")
    
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
    Value = Current Demand × Revenue/Demand Point × Momentum Factor × 
            Genre Multiplier × Volatility Discount × Options Multiplier
    """
    print("Calculating final valuations...")
    
    if demand_data is None:
        print("ERROR: No demand data available")
        return []
    
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
    
    # Calculate revenue per demand point
    revenue_per_demand = {}
    for show in shows:
        show_demand = demand_data[demand_data['show_name'] == show]
        show_revenue = revenue_data[revenue_data['show_name'] == show]
        
        if len(show_demand) > 0 and len(show_revenue) > 0:
            avg_demand = show_demand['weighted_demand'].mean()
            avg_revenue = show_revenue['gross_receipts'].mean()
            
            # Revenue per demand point (in millions)
            if avg_demand > 0:
                revenue_per_demand[show] = avg_revenue / avg_demand
            else:
                revenue_per_demand[show] = 0
        else:
            revenue_per_demand[show] = 0
    
    # Calculate valuations
    valuations = []
    for show in shows:
        # Get latest demand
        latest_demand = demand_data[demand_data['show_name'] == show].sort_values('date')
        
        if len(latest_demand) > 0:
            current_demand = latest_demand['weighted_demand'].iloc[-1]
            
            # Get momentum
            momentum = momentum_results.get(show, 1.0)
            
            # Get revenue per demand point
            rev_per_demand = revenue_per_demand.get(show, 0)
            
            # Get genre multiplier
            genre = default_genres.get(show, 'Default')
            genre_mult = genre_multipliers.get(genre, genre_multipliers['Default'])
            
            # Get lag information
            lag_info = lag_results.get(show, {'optimal_lag': 0, 'correlation': 0})
            
            # Volatility discount (using correlation as a proxy for volatility)
            volatility_discount = 0.9 + (abs(lag_info['correlation']) * 0.1)
            
            # Calculate valuation
            valuation = current_demand * rev_per_demand * momentum * genre_mult * volatility_discount
            
            print(f"- {show}: ${valuation/1000000:.2f}M")
            print(f"  * Demand: {current_demand:.2f}")
            print(f"  * Revenue/Demand: ${rev_per_demand/1000000:.2f}M")
            print(f"  * Momentum: {momentum:.2f}x")
            print(f"  * Genre ({genre}): {genre_mult:.2f}x")
            print(f"  * Volatility: {volatility_discount:.2f}")
            
            valuations.append({
                'show_name': show,
                'current_demand': current_demand,
                'weighted_demand': current_demand,  # Same as current for simplicity
                'momentum_factor': momentum,
                'revenue_per_demand': rev_per_demand,
                'genre': genre,
                'genre_multiplier': genre_mult,
                'optimal_lag': lag_info['optimal_lag'],
                'lag_correlation': lag_info['correlation'],
                'volatility_discount': volatility_discount,
                'predicted_valuation': valuation
            })
    
    return valuations

def save_results(valuations, lag_results, momentum_results):
    """Save processed results to the processed data folder"""
    print("Saving results...")
    
    try:
        os.makedirs('../dataprocessed', exist_ok=True)
        print(f"Output directory: {os.path.abspath('../dataprocessed')}")
        
        # Convert lag_results to a serializable format
        serializable_lag_results = {}
        for show, data in lag_results.items():
            serializable_lag_results[show] = {
                'optimal_lag': data['optimal_lag'],
                'correlation': float(data['correlation'])  # Convert numpy types to Python types
            }
        
        # Save valuations
        with open('../dataprocessed/valuations.json', 'w') as f:
            json.dump(valuations, f, indent=2)
            print(f"Saved valuations.json with {len(valuations)} shows")
        
        # Save lag results
        with open('../dataprocessed/lag_results.json', 'w') as f:
            json.dump(serializable_lag_results, f, indent=2)
            print(f"Saved lag_results.json with {len(serializable_lag_results)} shows")
        
        # Save momentum results
        serializable_momentum = {show: float(value) for show, value in momentum_results.items()}
        with open('../dataprocessed/momentum_results.json', 'w') as f:
            json.dump(serializable_momentum, f, indent=2)
            print(f"Saved momentum_results.json with {len(serializable_momentum)} shows")
        
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