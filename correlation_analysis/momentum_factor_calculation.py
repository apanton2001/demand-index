import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
import os
from datetime import datetime, timedelta

def load_data():
    """Load raw data from Excel file"""
    try:
        excel_path = Path('../dataraw/show_analysis.xlsx')
        print(f"Loading data from: {excel_path.absolute()}")
        
        if not excel_path.exists():
            print(f"ERROR: File not found: {excel_path.absolute()}")
            return None
        
        # Load demand data
        demand_data = pd.read_excel(excel_path, sheet_name='demand_data')
        
        print(f"Loaded {len(demand_data)} demand records")
        return demand_data
    
    except Exception as e:
        print(f"ERROR loading data: {str(e)}")
        return None

def preprocess_data(demand_data):
    """Prepare data for momentum analysis"""
    print("\nPreprocessing data for momentum analysis...")
    
    if demand_data is None:
        print("ERROR: No demand data available")
        return None
    
    # Ensure date column is datetime
    demand_data['date'] = pd.to_datetime(demand_data['date'])
    
    # If weighted_demand doesn't exist, use demand_score
    if 'weighted_demand' not in demand_data.columns and 'demand_score' in demand_data.columns:
        demand_data['weighted_demand'] = demand_data['demand_score']
        print("Using demand_score as weighted_demand")
    
    # Sort data by show and date
    demand_data = demand_data.sort_values(['show_name', 'date'])
    
    # Get unique shows and territories
    shows = demand_data['show_name'].unique()
    territories = demand_data['territory'].unique() if 'territory' in demand_data.columns else []
    
    print(f"Preprocessed data for {len(shows)} shows across {len(territories)} territories")
    return demand_data

def calculate_momentum_factors(demand_data, window_days=90):
    """
    Calculate momentum factors by comparing recent demand to previous period
    
    Momentum factor = Recent demand (last window_days) / Previous demand (preceding window_days)
    Values > 1 indicate positive momentum, values < 1 indicate negative momentum
    """
    print(f"\nCalculating momentum factors with {window_days}-day window...")
    
    if demand_data is None:
        print("ERROR: No data available for momentum calculation")
        return None
    
    # Get unique shows
    shows = demand_data['show_name'].unique()
    print(f"Analyzing momentum for {len(shows)} shows")
    
    # Results dictionary
    momentum_results = {}
    momentum_details = {}
    
    for show in shows:
        # Filter data for this show
        show_data = demand_data[demand_data['show_name'] == show].copy()
        
        # Calculate average demand per date (across territories)
        daily_demand = show_data.groupby('date')['weighted_demand'].mean().reset_index()
        daily_demand = daily_demand.sort_values('date')
        
        # Handle case with insufficient data
        if len(daily_demand) < window_days * 2:
            print(f"- {show}: Insufficient data points ({len(daily_demand)} < {window_days*2}), using default momentum of 1.0")
            momentum_results[show] = 1.0
            momentum_details[show] = {
                'data_points': len(daily_demand),
                'recent_window': None,
                'previous_window': None,
                'recent_demand': None,
                'previous_demand': None,
                'momentum_factor': 1.0,
                'trend': 'neutral (insufficient data)'
            }
            continue
        
        # Find the most recent date
        latest_date = daily_demand['date'].max()
        
        # Calculate window boundaries
        recent_end = latest_date
        recent_start = latest_date - timedelta(days=window_days)
        previous_end = recent_start
        previous_start = recent_start - timedelta(days=window_days)
        
        # Filter data for recent and previous periods
        recent_data = daily_demand[(daily_demand['date'] >= recent_start) & 
                                   (daily_demand['date'] <= recent_end)]
        
        previous_data = daily_demand[(daily_demand['date'] >= previous_start) & 
                                     (daily_demand['date'] <= previous_end)]
        
        # Calculate average demand for each period
        recent_demand = recent_data['weighted_demand'].mean()
        previous_demand = previous_data['weighted_demand'].mean()
        
        # Calculate momentum factor
        if previous_demand > 0:
            momentum = recent_demand / previous_demand
        else:
            momentum = 1.0  # Default to neutral momentum if previous demand is zero
        
        # Determine trend description
        if momentum > 1.15:
            trend = "strong upward"
        elif momentum > 1.05:
            trend = "upward"
        elif momentum > 0.95:
            trend = "neutral"
        elif momentum > 0.85:
            trend = "downward"
        else:
            trend = "strong downward"
        
        # Store results
        momentum_results[show] = momentum
        momentum_details[show] = {
            'data_points': len(daily_demand),
            'recent_window': {
                'start': recent_start.strftime('%Y-%m-%d'),
                'end': recent_end.strftime('%Y-%m-%d')
            },
            'previous_window': {
                'start': previous_start.strftime('%Y-%m-%d'),
                'end': previous_end.strftime('%Y-%m-%d')
            },
            'recent_demand': float(recent_demand),
            'previous_demand': float(previous_demand),
            'momentum_factor': float(momentum),
            'trend': trend
        }
        
        print(f"- {show}: Momentum = {momentum:.2f} ({trend} trend)")
    
    return momentum_results, momentum_details

def analyze_momentum_distribution(momentum_results):
    """Analyze the distribution of momentum factors"""
    print("\nAnalyzing momentum factor distribution...")
    
    if not momentum_results:
        print("ERROR: No momentum results available for analysis")
        return None
    
    # Convert to pandas Series for analysis
    momentum_series = pd.Series(momentum_results)
    
    # Calculate distribution statistics
    stats = {
        'count': len(momentum_series),
        'mean': momentum_series.mean(),
        'median': momentum_series.median(),
        'std': momentum_series.std(),
        'min': momentum_series.min(),
        'max': momentum_series.max(),
        'quartiles': momentum_series.quantile([0.25, 0.5, 0.75]).to_dict()
    }
    
    # Calculate trend distribution
    trend_counts = {
        'strong_upward': sum(momentum_series > 1.15),
        'upward': sum((momentum_series > 1.05) & (momentum_series <= 1.15)),
        'neutral': sum((momentum_series >= 0.95) & (momentum_series <= 1.05)),
        'downward': sum((momentum_series >= 0.85) & (momentum_series < 0.95)),
        'strong_downward': sum(momentum_series < 0.85)
    }
    
    # Calculate percentages
    total = len(momentum_series)
    trend_percentages = {k: (v / total * 100) for k, v in trend_counts.items()}
    
    print(f"Momentum factor statistics:")
    print(f"- Mean: {stats['mean']:.2f}")
    print(f"- Median: {stats['median']:.2f}")
    print(f"- Range: {stats['min']:.2f} to {stats['max']:.2f}")
    print(f"- Standard deviation: {stats['std']:.2f}")
    
    print(f"\nMomentum trend distribution:")
    print(f"- Strong upward trend (>1.15): {trend_percentages['strong_upward']:.1f}%")
    print(f"- Upward trend (1.05-1.15): {trend_percentages['upward']:.1f}%")
    print(f"- Neutral trend (0.95-1.05): {trend_percentages['neutral']:.1f}%")
    print(f"- Downward trend (0.85-0.95): {trend_percentages['downward']:.1f}%")
    print(f"- Strong downward trend (<0.85): {trend_percentages['strong_downward']:.1f}%")
    
    return {
        'stats': stats,
        'trend_counts': trend_counts,
        'trend_percentages': trend_percentages
    }

def plot_time_series_examples(demand_data, momentum_details, output_dir='../web/images'):
    """Plot demand time series for shows with different momentum patterns"""
    print("\nCreating time series visualizations for momentum examples...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find examples of different momentum patterns
    momentum_values = [(show, details['momentum_factor']) for show, details in momentum_details.items() 
                      if details['momentum_factor'] != 1.0]  # Exclude default values
    
    if not momentum_values:
        print("No suitable momentum examples found for visualization")
        return
    
    # Sort by momentum factor
    momentum_values.sort(key=lambda x: x[1])
    
    # Select representative examples
    examples = []
    
    # Highest momentum
    if momentum_values[-1][1] > 1.15:
        examples.append(momentum_values[-1][0])  # Strong upward
    
    # Lowest momentum
    if momentum_values[0][1] < 0.85:
        examples.append(momentum_values[0][1])  # Strong downward
    
    # Find neutral (momentum close to 1.0)
    neutral_idx = min(range(len(momentum_values)), key=lambda i: abs(momentum_values[i][1] - 1.0))
    examples.append(momentum_values[neutral_idx][0])
    
    # Ensure we have at least 3 examples
    while len(examples) < 3 and len(momentum_values) > len(examples):
        # Add examples from different parts of the distribution
        quartile_idx = int(len(momentum_values) * 0.75)
        if momentum_values[quartile_idx][0] not in examples:
            examples.append(momentum_values[quartile_idx][0])
        if len(examples) < 3:
            quartile_idx = int(len(momentum_values) * 0.25)
            if momentum_values[quartile_idx][0] not in examples:
                examples.append(momentum_values[quartile_idx][0])
    
    # Limit to maximum 5 examples
    examples = examples[:5]
    
    # Plot each example
    for show in examples:
        # Filter data for this show
        show_data = demand_data[demand_data['show_name'] == show].copy()
        
        # Calculate average demand per date (across territories)
        daily_demand = show_data.groupby('date')['weighted_demand'].mean().reset_index()
        daily_demand = daily_demand.sort_values('date')
        
        # Plot time series
        plt.figure(figsize=(12, 6))
        plt.plot(daily_demand['date'], daily_demand['weighted_demand'], 'b-', marker='o', alpha=0.7)
        
        # Get window details
        details = momentum_details[show]
        if details['recent_window'] is not None:
            recent_start = pd.to_datetime(details['recent_window']['start'])
            recent_end = pd.to_datetime(details['recent_window']['end'])
            previous_start = pd.to_datetime(details['previous_window']['start'])
            previous_end = pd.to_datetime(details['previous_window']['end'])
            
            # Highlight the two windows
            recent_data = daily_demand[(daily_demand['date'] >= recent_start) & 
                                       (daily_demand['date'] <= recent_end)]
            previous_data = daily_demand[(daily_demand['date'] >= previous_start) & 
                                         (daily_demand['date'] <= previous_end)]
            
            # Plot highlighted windows
            plt.plot(recent_data['date'], recent_data['weighted_demand'], 'g-', linewidth=3, 
                     label=f"Recent window (avg: {details['recent_demand']:.3f})")
            plt.plot(previous_data['date'], previous_data['weighted_demand'], 'r-', linewidth=3,
                     label=f"Previous window (avg: {details['previous_demand']:.3f})")
            
            # Add horizontal lines for averages
            plt.axhline(y=details['recent_demand'], color='g', linestyle='--', alpha=0.7)
            plt.axhline(y=details['previous_demand'], color='r', linestyle='--', alpha=0.7)
        
        plt.title(f"{show} - Demand Trend (Momentum: {details['momentum_factor']:.2f}, {details['trend']})", 
                  fontsize=16)
        plt.xlabel('Date', fontsize=14)
        plt.ylabel('Weighted Demand', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # Format date axis
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=2))
        plt.gcf().autofmt_xdate()
        
        # Save the figure
        chart_path = f"{output_dir}/momentum_{show.replace(' ', '_')}.png"
        plt.savefig(chart_path)
        print(f"Saved time series chart for {show} to {chart_path}")
        
        # Close figure to free memory
        plt.close()

def generate_visualizations(momentum_results, momentum_analysis, output_dir='../web/images'):
    """Generate visualizations for momentum factor analysis"""
    print("\nGenerating momentum factor visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Momentum Factor Distribution
    plt.figure(figsize=(12, 8))
    
    # Create histogram with KDE
    momentum_values = list(momentum_results.values())
    plt.hist(momentum_values, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
    
    # Add vertical lines for key statistics
    plt.axvline(x=momentum_analysis['stats']['mean'], color='red', linestyle='--', 
                label=f"Mean: {momentum_analysis['stats']['mean']:.2f}")
    plt.axvline(x=momentum_analysis['stats']['median'], color='green', linestyle='--',
                label=f"Median: {momentum_analysis['stats']['median']:.2f}")
    plt.axvline(x=1.0, color='black', linestyle='-', label="Neutral momentum (1.0)")
    
    # Add reference areas for trend categories
    plt.axvspan(0.0, 0.85, alpha=0.2, color='red', label="Strong downward trend")
    plt.axvspan(0.85, 0.95, alpha=0.1, color='orange', label="Downward trend")
    plt.axvspan(0.95, 1.05, alpha=0.1, color='gray', label="Neutral trend")
    plt.axvspan(1.05, 1.15, alpha=0.1, color='lightgreen', label="Upward trend")
    plt.axvspan(1.15, 2.0, alpha=0.2, color='green', label="Strong upward trend")
    
    plt.title('Distribution of Momentum Factors Across Shows', fontsize=16)
    plt.xlabel('Momentum Factor', fontsize=14)
    plt.ylabel('Number of Shows', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper right')
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/momentum_distribution.png"
    plt.savefig(chart_path)
    print(f"Saved momentum distribution chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # 2. Bar chart of top and bottom momentum shows
    plt.figure(figsize=(14, 8))
    
    # Sort shows by momentum
    sorted_momentum = sorted(momentum_results.items(), key=lambda x: x[1])
    
    # Extract top 7 and bottom 7 shows
    bottom_shows = sorted_momentum[:7]
    top_shows = sorted_momentum[-7:]
    
    # Combine into one set for visualization
    combined_shows = bottom_shows + top_shows
    show_names = [item[0] for item in combined_shows]
    momentum_values = [item[1] for item in combined_shows]
    
    # Create bar chart with color coding
    colors = ['red' if value < 1 else 'green' for value in momentum_values]
    bars = plt.bar(range(len(show_names)), momentum_values, color=colors)
    
    # Customize colors based on momentum categories
    for i, bar in enumerate(bars):
        value = momentum_values[i]
        if value < 0.85:
            bar.set_color('darkred')
        elif value < 0.95:
            bar.set_color('red')
        elif value < 1.05:
            bar.set_color('gray')
        elif value < 1.15:
            bar.set_color('lightgreen')
        else:
            bar.set_color('darkgreen')
    
    # Add reference line for neutral momentum
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label="Neutral momentum (1.0)")
    
    plt.title('Shows with Highest and Lowest Momentum Factors', fontsize=16)
    plt.xlabel('Show', fontsize=14)
    plt.ylabel('Momentum Factor', fontsize=14)
    plt.xticks(range(len(show_names)), show_names, rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/momentum_extremes.png"
    plt.savefig(chart_path)
    print(f"Saved momentum extremes chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # 3. Pie chart of momentum trend distribution
    plt.figure(figsize=(10, 8))
    
    # Extract data for pie chart
    trend_labels = [
        f"Strong upward (>1.15): {momentum_analysis['trend_percentages']['strong_upward']:.1f}%",
        f"Upward (1.05-1.15): {momentum_analysis['trend_percentages']['upward']:.1f}%",
        f"Neutral (0.95-1.05): {momentum_analysis['trend_percentages']['neutral']:.1f}%",
        f"Downward (0.85-0.95): {momentum_analysis['trend_percentages']['downward']:.1f}%",
        f"Strong downward (<0.85): {momentum_analysis['trend_percentages']['strong_downward']:.1f}%"
    ]
    
    trend_values = [
        momentum_analysis['trend_counts']['strong_upward'],
        momentum_analysis['trend_counts']['upward'],
        momentum_analysis['trend_counts']['neutral'],
        momentum_analysis['trend_counts']['downward'],
        momentum_analysis['trend_counts']['strong_downward']
    ]
    
    trend_colors = ['darkgreen', 'lightgreen', 'gray', 'orange', 'red']
    
    plt.pie(trend_values, labels=trend_labels, colors=trend_colors, autopct='%1.1f%%',
            startangle=90, shadow=True, explode=[0.05, 0, 0, 0, 0.05])
    plt.title('Distribution of Momentum Trend Categories', fontsize=16)
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle
    
    # Save figure
    chart_path = f"{output_dir}/momentum_trend_distribution.png"
    plt.savefig(chart_path)
    print(f"Saved momentum trend distribution chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()

def save_results(momentum_results, momentum_details, momentum_analysis, output_dir='../dataprocessed'):
    """Save the momentum analysis results"""
    print(f"\nSaving momentum analysis results to {output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save momentum factors
    serializable_results = {k: float(v) for k, v in momentum_results.items()}
    with open(f"{output_dir}/momentum_factors.json", 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    # 2. Save detailed momentum analysis
    with open(f"{output_dir}/momentum_details.json", 'w') as f:
        json.dump(momentum_details, f, indent=2)
    
    # 3. Save momentum distribution analysis
    with open(f"{output_dir}/momentum_analysis.json", 'w') as f:
        # Convert numpy types to Python native types for serialization
        serializable_analysis = {
            'stats': {
                'count': int(momentum_analysis['stats']['count']),
                'mean': float(momentum_analysis['stats']['mean']),
                'median': float(momentum_analysis['stats']['median']),
                'std': float(momentum_analysis['stats']['std']),
                'min': float(momentum_analysis['stats']['min']),
                'max': float(momentum_analysis['stats']['max']),
                'quartiles': {k: float(v) for k, v in momentum_analysis['stats']['quartiles'].items()}
            },
            'trend_counts': {k: int(v) for k, v in momentum_analysis['trend_counts'].items()},
            'trend_percentages': {k: float(v) for k, v in momentum_analysis['trend_percentages'].items()}
        }
        json.dump(serializable_analysis, f, indent=2)
    
    print("Momentum analysis results saved successfully")

def generate_html_report(momentum_results, momentum_details, momentum_analysis, output_path='../correlation_analysis/momentum_factor_report.html'):
    """Generate an HTML report of the momentum factor analysis"""
    print(f"\nGenerating HTML report to {output_path}")
    
    # Create an HTML report with findings
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Momentum Factor Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .trend-positive {{ color: green; }}
            .trend-negative {{ color: red; }}
            .trend-neutral {{ color: gray; }}
            .highlight {{ background-color: #e6f7ff; }}
            .chart {{ margin: 20px 0; max-width: 100%; }}
        </style>
    </head>
    <body>
        <h1>Momentum Factor Analysis</h1>
        
        <h2>Overview</h2>
        <p>
            This analysis calculates momentum factors by comparing recent demand to previous demand periods.
            Momentum factors above 1.0 indicate increasing demand, while factors below 1.0 indicate decreasing demand.
        </p>
        
        <div class="highlight">
            <p><strong>Momentum Factor = Recent Demand / Previous Demand</strong></p>
            <ul>
                <li>Values > 1.15: Strong upward trend</li>
                <li>Values 1.05-1.15: Upward trend</li>
                <li>Values 0.95-1.05: Neutral trend</li>
                <li>Values 0.85-0.95: Downward trend</li>
                <li>Values < 0.85: Strong downward trend</li>
            </ul>
        </div>
        
        <h2>Summary Statistics</h2>
        <p>
            The analysis covers {momentum_analysis['stats']['count']} shows with the following distribution:
        </p>
        <ul>
            <li>Mean momentum factor: <strong>{momentum_analysis['stats']['mean']:.2f}</strong></li>
            <li>Median momentum factor: <strong>{momentum_analysis['stats']['median']:.2f}</strong></li>
            <li>Range: {momentum_analysis['stats']['min']:.2f} to {momentum_analysis['stats']['max']:.2f}</li>
            <li>Standard deviation: {momentum_analysis['stats']['std']:.2f}</li>
        </ul>
        
        <h2>Trend Distribution</h2>
        <div class="chart">
            <img src="../web/images/momentum_trend_distribution.png" alt="Momentum Trend Distribution" style="max-width: 100%;">
        </div>
        
        <h2>Momentum Factor Distribution</h2>
        <div class="chart">
            <img src="../web/images/momentum_distribution.png" alt="Momentum Distribution" style="max-width: 100%;">
        </div>
        
        <h2>Shows with Extreme Momentum</h2>
        <div class="chart">
            <img src="../web/images/momentum_extremes.png" alt="Shows with Extreme Momentum" style="max-width: 100%;">
        </div>
        
        <h2>Example Time Series</h2>
    """
    
    # Add example time series images if available
    example_files = [f for f in os.listdir(f'../web/images') if f.startswith('momentum_') and f.endswith('.png') and not f.endswith('_extremes.png') and not f.endswith('_distribution.png') and not f.endswith('_trend_distribution.png')]
    
    for file in example_files:
        show_name = file.replace('momentum_', '').replace('.png', '').replace('_', ' ')
        html_content += f"""
        <h3>{show_name}</h3>
        <div class="chart">
            <img src="../web/images/{file}" alt="{show_name} Momentum" style="max-width: 100%;">
        </div>
        """
    
    html_content += """
        <h2>Top Shows by Momentum</h2>
        <table>
            <tr>
                <th>Show</th>
                <th>Momentum Factor</th>
                <th>Trend</th>
                <th>Recent Demand</th>
                <th>Previous Demand</th>
                <th>Change (%)</th>
            </tr>
    """
    
    # Add top shows to the table
    top_shows = sorted([(show, details) for show, details in momentum_details.items() 
                        if details['momentum_factor'] != 1.0],
                       key=lambda x: x[1]['momentum_factor'], reverse=True)[:10]
    
    for show, details in top_shows:
        if details['recent_demand'] is not None and details['previous_demand'] is not None:
            change_pct = (details['recent_demand'] / details['previous_demand'] - 1) * 100 if details['previous_demand'] > 0 else 0
            trend_class = "trend-positive" if details['momentum_factor'] > 1.05 else "trend-negative" if details['momentum_factor'] < 0.95 else "trend-neutral"
            
            html_content += f"""
                <tr>
                    <td>{show}</td>
                    <td class="{trend_class}">{details['momentum_factor']:.2f}</td>
                    <td class="{trend_class}">{details['trend']}</td>
                    <td>{details['recent_demand']:.3f}</td>
                    <td>{details['previous_demand']:.3f}</td>
                    <td class="{trend_class}">{change_pct:+.1f}%</td>
                </tr>
            """
    
    html_content += """
        </table>
        
        <h2>Bottom Shows by Momentum</h2>
        <table>
            <tr>
                <th>Show</th>
                <th>Momentum Factor</th>
                <th>Trend</th>
                <th>Recent Demand</th>
                <th>Previous Demand</th>
                <th>Change (%)</th>
            </tr>
    """
    
    # Add bottom shows to the table
    bottom_shows = sorted([(show, details) for show, details in momentum_details.items() 
                           if details['momentum_factor'] != 1.0],
                          key=lambda x: x[1]['momentum_factor'])[:10]
    
    for show, details in bottom_shows:
        if details['recent_demand'] is not None and details['previous_demand'] is not None:
            change_pct = (details['recent_demand'] / details['previous_demand'] - 1) * 100 if details['previous_demand'] > 0 else 0
            trend_class = "trend-positive" if details['momentum_factor'] > 1.05 else "trend-negative" if details['momentum_factor'] < 0.95 else "trend-neutral"
            
            html_content += f"""
                <tr>
                    <td>{show}</td>
                    <td class="{trend_class}">{details['momentum_factor']:.2f}</td>
                    <td class="{trend_class}">{details['trend']}</td>
                    <td>{details['recent_demand']:.3f}</td>
                    <td>{details['previous_demand']:.3f}</td>
                    <td class="{trend_class}">{change_pct:+.1f}%</td>
                </tr>
            """
    
    html_content += """
        </table>
        
        <h2>Methodology</h2>
        <p>
            The momentum factor calculation follows these steps:
        </p>
        <ol>
            <li>Segment demand data into two adjacent time windows (90 days each by default)</li>
            <li>Calculate the average demand in each window</li>
            <li>Compute the momentum factor as the ratio of recent demand to previous demand</li>
            <li>Classify the trend based on the momentum factor value</li>
        </ol>
        
        <h2>Applications</h2>
        <p>
            Momentum factors provide several advantages in content valuation:
        </p>
        <ul>
            <li>Identify content with growing or declining audience interest</li>
            <li>Adjust valuations based on demand trajectory, not just current level</li>
            <li>Recognize early indicators of potential breakout content</li>
            <li>Identify content that may be near the end of its commercial lifecycle</li>
            <li>Provide a dynamic component to otherwise static valuation models</li>
        </ul>
    </body>
    </html>
    """
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Write the HTML content to a file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated at {output_path}")

# Execute the analysis
if __name__ == "__main__":
    print("=" * 60)
    print("MOMENTUM FACTOR CALCULATION")
    print("=" * 60)
    
    # Create output directory for visualizations
    os.makedirs('../web/images', exist_ok=True)
    
    # Load data
    demand_data = load_data()
    
    # Preprocess data
    processed_data = preprocess_data(demand_data)
    
    # Calculate momentum factors
    momentum_results, momentum_details = calculate_momentum_factors(processed_data)
    
    # Analyze momentum distribution
    momentum_analysis = analyze_momentum_distribution(momentum_results)
    
    # Generate time series examples
    plot_time_series_examples(processed_data, momentum_details)
    
    # Generate visualizations
    generate_visualizations(momentum_results, momentum_analysis)
    
    # Save results
    save_results(momentum_results, momentum_details, momentum_analysis)
    
    # Generate HTML report
    generate_html_report(momentum_results, momentum_details, momentum_analysis)
    
    print("\nMomentum factor calculation complete!")
    print("=" * 60) 