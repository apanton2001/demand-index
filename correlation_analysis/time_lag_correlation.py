import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import os

def load_data():
    """Load raw data from Excel file"""
    try:
        excel_path = Path('../dataraw/show_analysis.xlsx')
        print(f"Loading data from: {excel_path.absolute()}")
        
        if not excel_path.exists():
            print(f"ERROR: File not found: {excel_path.absolute()}")
            return None, None
        
        # Load demand and revenue data
        demand_data = pd.read_excel(excel_path, sheet_name='demand_data')
        revenue_data = pd.read_excel(excel_path, sheet_name='revenue_data')
        
        print(f"Loaded {len(demand_data)} demand records and {len(revenue_data)} revenue records")
        return demand_data, revenue_data
    
    except Exception as e:
        print(f"ERROR loading data: {str(e)}")
        return None, None

def calculate_time_lag_correlation(demand_data, revenue_data, max_lag_months=12):
    """
    Calculate time lag correlation between demand and revenue
    This is a more detailed version of the function in process_data.py
    with additional visualization and analysis
    """
    print("\nCalculating time lag correlation...")
    
    if demand_data is None or revenue_data is None:
        print("ERROR: Missing data for analysis")
        return {}, {}
    
    # Ensure date columns are datetime
    demand_data['date'] = pd.to_datetime(demand_data['date'])
    revenue_data['date'] = pd.to_datetime(revenue_data['date'])
    
    # Get unique shows
    shows = demand_data['show_name'].unique()
    print(f"Analyzing {len(shows)} shows for time lag patterns")
    
    # Results dictionaries
    lag_results = {}
    correlation_by_lag = {}  # For tracking correlation at each lag period
    
    for show in shows:
        show_demand = demand_data[demand_data['show_name'] == show]
        show_revenue = revenue_data[revenue_data['show_name'] == show]
        
        if len(show_demand) < 5 or len(show_revenue) < 5:
            print(f"- {show}: Insufficient data points, skipping lag analysis")
            lag_results[show] = {'optimal_lag': 0, 'correlation': 0}
            continue
        
        # Track correlation at each lag period for this show
        show_lag_corr = []
        
        # Calculate weighted demand if available
        if 'weighted_demand' not in show_demand.columns and 'demand_score' in show_demand.columns:
            show_demand['weighted_demand'] = show_demand['demand_score']
        
        # Try different lag periods
        best_lag = 0
        best_corr = 0
        
        for lag in range(max_lag_months + 1):
            # Create lagged demand data
            lagged_demand = show_demand.copy()
            lagged_demand['lag_date'] = lagged_demand['date'] + pd.DateOffset(months=lag)
            
            # Merge with revenue on date
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
                show_lag_corr.append((lag, corr))
                
                if abs(corr) > abs(best_corr):
                    best_corr = corr
                    best_lag = lag
            else:
                show_lag_corr.append((lag, 0))
        
        lag_results[show] = {
            'optimal_lag': best_lag,
            'correlation': best_corr
        }
        
        correlation_by_lag[show] = show_lag_corr
        print(f"- {show}: Optimal lag = {best_lag} months, Correlation = {best_corr:.2f}")
    
    return lag_results, correlation_by_lag

def generate_visualizations(correlation_by_lag, output_dir='../web/images'):
    """Generate visualizations for time lag correlation analysis"""
    print("\nGenerating time lag visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Overall correlation by lag period
    plt.figure(figsize=(12, 8))
    
    # Select top shows with non-zero correlations
    relevant_shows = []
    for show, corr_data in correlation_by_lag.items():
        max_corr = max([abs(c) for _, c in corr_data])
        if max_corr > 0.1:  # Only include shows with meaningful correlation
            relevant_shows.append((show, max_corr))
    
    # Sort and get top shows
    top_shows = [s[0] for s in sorted(relevant_shows, key=lambda x: x[1], reverse=True)[:8]]
    
    for show in top_shows:
        corr_data = correlation_by_lag[show]
        lags = [l for l, _ in corr_data]
        corrs = [c for _, c in corr_data]
        plt.plot(lags, corrs, marker='o', linewidth=2, label=show)
    
    plt.title('Time Lag Correlation Analysis: Demand vs. Revenue', fontsize=16)
    plt.xlabel('Lag Period (months)', fontsize=14)
    plt.ylabel('Correlation Coefficient (r)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/time_lag_correlation_chart.png"
    plt.savefig(chart_path)
    print(f"Saved time lag correlation chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # Create a summary chart of optimal lag by show
    plt.figure(figsize=(12, 8))
    
    # Extract optimal lag and correlation data from lag_results
    shows = []
    optimal_lags = []
    correlations = []
    
    for show, data in lag_results.items():
        # Only include shows with meaningful correlation
        if abs(data['correlation']) > 0.1:
            shows.append(show)
            optimal_lags.append(data['optimal_lag'])
            correlations.append(abs(data['correlation']))
    
    # Sort by correlation strength
    sorted_indices = np.argsort(correlations)[::-1]
    shows = [shows[i] for i in sorted_indices[:12]]  # Top 12 shows
    optimal_lags = [optimal_lags[i] for i in sorted_indices[:12]]
    correlations = [correlations[i] for i in sorted_indices[:12]]
    
    # Create the bar chart
    bars = plt.bar(range(len(shows)), optimal_lags, color='skyblue')
    
    # Color bars by correlation strength
    for i, bar in enumerate(bars):
        bar.set_color(plt.cm.viridis(correlations[i]))
    
    plt.xticks(range(len(shows)), shows, rotation=45, ha='right')
    plt.title('Optimal Lag Period by Show', fontsize=16)
    plt.xlabel('Show', fontsize=14)
    plt.ylabel('Optimal Lag (months)', fontsize=14)
    plt.tight_layout()
    
    # Add correlation strength as annotations
    for i, corr in enumerate(correlations):
        plt.annotate(f'r = {corr:.2f}', 
                     xy=(i, optimal_lags[i] + 0.1), 
                     ha='center')
    
    # Save figure
    chart_path = f"{output_dir}/optimal_lag_by_show.png"
    plt.savefig(chart_path)
    print(f"Saved optimal lag chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()

def save_results(lag_results, output_path='../dataprocessed/lag_analysis_results.json'):
    """Save the full lag correlation results to a JSON file"""
    print(f"\nSaving detailed lag analysis results to {output_path}")
    
    # Ensure all values are JSON serializable
    serializable_results = {}
    for show, data in lag_results.items():
        serializable_results[show] = {
            'optimal_lag': int(data['optimal_lag']),
            'correlation': float(data['correlation'])
        }
    
    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    print(f"Results saved for {len(serializable_results)} shows")

def generate_html_report(lag_results, correlation_by_lag, output_path='../correlation_analysis/time_lag_report.html'):
    """Generate an HTML report of the time lag analysis results"""
    print(f"\nGenerating HTML report to {output_path}")
    
    # Create an HTML report with findings
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Time Lag Correlation Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #e6f7ff; }}
            .chart {{ margin: 20px 0; max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Time Lag Correlation Analysis</h1>
        
        <h2>Overview</h2>
        <p>
            This analysis explores the time lag relationship between demand metrics and revenue data for various shows.
            The time lag represents the period (in months) between changes in demand and corresponding changes in revenue.
        </p>
        
        <h2>Key Findings</h2>
        <ul>
    """
    
    # Get shows with non-zero correlations
    valid_results = {show: data for show, data in lag_results.items() if abs(data['correlation']) > 0.1}
    if valid_results:
        # Find the show with the highest correlation
        best_show = max(valid_results.items(), key=lambda x: abs(x[1]['correlation']))
        
        # Find the average optimal lag
        avg_lag = sum(data['optimal_lag'] for data in valid_results.values()) / len(valid_results)
        
        html_content += f"""
            <li>The average optimal lag period across shows with meaningful correlation is {avg_lag:.1f} months</li>
            <li>The strongest correlation was found for <strong>{best_show[0]}</strong> with r = {best_show[1]['correlation']:.2f} at a lag of {best_show[1]['optimal_lag']} months</li>
            <li>{len(valid_results)} out of {len(lag_results)} shows showed meaningful lag correlation (|r| > 0.1)</li>
        </ul>
        """
    else:
        html_content += """
            <li>No shows exhibited meaningful correlation between demand and revenue</li>
        </ul>
        """
    
    # Add visualization section
    html_content += """
        <h2>Visualizations</h2>
        
        <div class="chart">
            <h3>Time Lag Correlation by Show</h3>
            <img src="../web/images/time_lag_correlation_chart.png" alt="Time Lag Correlation Chart" style="max-width: 100%;">
        </div>
        
        <div class="chart">
            <h3>Optimal Lag Period by Show</h3>
            <img src="../web/images/optimal_lag_by_show.png" alt="Optimal Lag by Show" style="max-width: 100%;">
        </div>
        
        <h2>Detailed Results</h2>
        <table>
            <tr>
                <th>Show</th>
                <th>Optimal Lag (months)</th>
                <th>Correlation (r)</th>
            </tr>
    """
    
    # Add table rows for each show, sorted by correlation strength
    sorted_results = sorted(lag_results.items(), key=lambda x: abs(x[1]['correlation']), reverse=True)
    for show, data in sorted_results:
        highlight = " class='highlight'" if abs(data['correlation']) > 0.3 else ""
        html_content += f"""
            <tr{highlight}>
                <td>{show}</td>
                <td>{data['optimal_lag']}</td>
                <td>{data['correlation']:.3f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Methodology</h2>
        <p>
            The analysis calculated the correlation between demand metrics and revenue data with varying time offsets (0 to 12 months).
            For each show, the optimal lag period was determined as the offset that maximized the absolute correlation.
            This approach helps identify the typical delay between changes in audience demand and corresponding revenue impacts.
        </p>
        
        <h2>Limitations</h2>
        <ul>
            <li>The analysis is limited by the temporal granularity of the available data</li>
            <li>Shows with insufficient data points may not yield reliable correlation results</li>
            <li>External factors affecting revenue beyond demand are not accounted for in this analysis</li>
        </ul>
    </body>
    </html>
    """
    
    # Write the HTML content to a file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated at {output_path}")

# Execute the analysis
if __name__ == "__main__":
    print("=" * 60)
    print("TIME LAG CORRELATION ANALYSIS")
    print("=" * 60)
    
    # Create output directory for visualizations
    os.makedirs('../web/images', exist_ok=True)
    
    # Load data
    demand_data, revenue_data = load_data()
    
    # Calculate time lag correlation
    lag_results, correlation_by_lag = calculate_time_lag_correlation(demand_data, revenue_data)
    
    # Generate visualizations
    generate_visualizations(correlation_by_lag)
    
    # Save results
    save_results(lag_results)
    
    # Generate HTML report
    generate_html_report(lag_results, correlation_by_lag)
    
    print("\nTime lag correlation analysis complete!")
    print("=" * 60) 