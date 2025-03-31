import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split

def load_data():
    """Load raw data from Excel file"""
    try:
        excel_path = Path('../dataraw/show_analysis.xlsx')
        print(f"Loading data from: {excel_path.absolute()}")
        
        if not excel_path.exists():
            print(f"ERROR: File not found: {excel_path.absolute()}")
            return None, None, None
        
        # Load data sheets
        demand_data = pd.read_excel(excel_path, sheet_name='demand_data')
        revenue_data = pd.read_excel(excel_path, sheet_name='revenue_data')
        
        # Try to load metadata if available
        try:
            metadata = pd.read_excel(excel_path, sheet_name='show_metadata')
            print(f"Loaded metadata for {len(metadata)} shows")
        except Exception as e:
            print(f"No metadata sheet found: {str(e)}")
            metadata = None
        
        print(f"Loaded {len(demand_data)} demand records and {len(revenue_data)} revenue records")
        return demand_data, revenue_data, metadata
    
    except Exception as e:
        print(f"ERROR loading data: {str(e)}")
        return None, None, None

def preprocess_data(demand_data, revenue_data, metadata=None):
    """Prepare data for genre-specific regression analysis"""
    print("\nPreprocessing data for genre-specific regression...")
    
    if demand_data is None or revenue_data is None:
        print("ERROR: Missing demand or revenue data")
        return None
    
    # Ensure date columns are datetime
    demand_data['date'] = pd.to_datetime(demand_data['date'])
    revenue_data['date'] = pd.to_datetime(revenue_data['date'])
    
    # If weighted_demand doesn't exist, use demand_score
    if 'weighted_demand' not in demand_data.columns and 'demand_score' in demand_data.columns:
        demand_data['weighted_demand'] = demand_data['demand_score']
        print("Using demand_score as weighted_demand")
    
    # Aggregate demand and revenue data by show
    show_demand = demand_data.groupby('show_name')['weighted_demand'].mean().reset_index()
    show_revenue = revenue_data.groupby('show_name')['gross_receipts'].mean().reset_index()
    
    # Merge demand and revenue data
    merged_data = pd.merge(show_demand, show_revenue, on='show_name', how='inner')
    print(f"Merged data contains {len(merged_data)} shows with both demand and revenue data")
    
    # If we have metadata with genres, use it
    if metadata is not None and 'genre' in metadata.columns:
        merged_data = pd.merge(merged_data, metadata[['show_name', 'genre']], on='show_name', how='left')
        print(f"Added genre information from metadata")
    else:
        # Assign default genres based on predefined mapping
        default_genres = {
            'CRIMINAL MINDS': 'Crime',
            'GREY\'S ANATOMY': 'Medical',
            'FRIENDS': 'Comedy',
            'THE BIG BANG THEORY': 'Comedy',
            'MODERN FAMILY': 'Comedy',
            'NCIS': 'Crime',
            'HOUSE': 'Medical',
            'BONES': 'Crime',
            'TWO AND A HALF MEN': 'Comedy',
            'THE KING OF QUEENS': 'Comedy',
            'FRASIER': 'Comedy',
            'WILL & GRACE': 'Comedy',
            'HOW I MET YOUR MOTHER': 'Comedy',
            'ACCORDING TO JIM': 'Comedy',
            'SCRUBS': 'Medical'
        }
        
        # Apply case-insensitive matching
        show_to_genre = {show.upper(): genre for show, genre in default_genres.items()}
        
        def assign_genre(show_name):
            return show_to_genre.get(show_name.upper(), 'Default')
        
        merged_data['genre'] = merged_data['show_name'].apply(assign_genre)
        print(f"Applied default genre assignments")
    
    # Fill missing genres with 'Default'
    merged_data['genre'].fillna('Default', inplace=True)
    
    # Print genre distribution
    genre_counts = merged_data['genre'].value_counts()
    print("\nGenre distribution:")
    for genre, count in genre_counts.items():
        print(f"- {genre}: {count} shows")
    
    return merged_data

def perform_regression_by_genre(data, min_shows_per_genre=3):
    """
    Perform linear regression analysis by genre
    Predicting revenue from demand for each genre
    """
    print("\nPerforming genre-specific regression analysis...")
    
    if data is None:
        print("ERROR: No data available for regression")
        return None
    
    # Get unique genres
    genres = data['genre'].unique()
    print(f"Analyzing {len(genres)} genres")
    
    # Results dictionary
    regression_results = {}
    
    # Overall regression (all shows)
    X_all = data[['weighted_demand']]
    y_all = data['gross_receipts']
    
    model_all = LinearRegression()
    model_all.fit(X_all, y_all)
    
    y_pred_all = model_all.predict(X_all)
    r2_all = r2_score(y_all, y_pred_all)
    rmse_all = np.sqrt(mean_squared_error(y_all, y_pred_all))
    
    # Calculate revenue per demand point (coefficient)
    revenue_per_demand_all = model_all.coef_[0]
    
    regression_results['All Shows'] = {
        'coefficient': float(revenue_per_demand_all),
        'intercept': float(model_all.intercept_),
        'r2': float(r2_all),
        'rmse': float(rmse_all),
        'count': len(data),
        'multiplier': 1.0  # Base multiplier
    }
    
    print(f"Overall regression results:")
    print(f"- Revenue per demand point: ${revenue_per_demand_all:.2f}")
    print(f"- R² score: {r2_all:.3f}")
    print(f"- RMSE: ${rmse_all:.2f}")
    
    # Perform regression by genre
    for genre in genres:
        # Filter data for this genre
        genre_data = data[data['genre'] == genre]
        
        # Skip genres with too few shows
        if len(genre_data) < min_shows_per_genre:
            print(f"- {genre}: Insufficient data ({len(genre_data)} < {min_shows_per_genre}), skipping regression")
            regression_results[genre] = {
                'coefficient': float(revenue_per_demand_all),  # Use overall coefficient
                'intercept': float(model_all.intercept_),
                'r2': 0.0,
                'rmse': 0.0,
                'count': len(genre_data),
                'multiplier': 1.0  # Default multiplier
            }
            continue
        
        # Regression for this genre
        X_genre = genre_data[['weighted_demand']]
        y_genre = genre_data['gross_receipts']
        
        model_genre = LinearRegression()
        model_genre.fit(X_genre, y_genre)
        
        y_pred_genre = model_genre.predict(X_genre)
        r2_genre = r2_score(y_genre, y_pred_genre)
        rmse_genre = np.sqrt(mean_squared_error(y_genre, y_pred_genre))
        
        # Calculate revenue per demand point (coefficient)
        revenue_per_demand_genre = model_genre.coef_[0]
        
        # Calculate genre multiplier (relative to overall coefficient)
        genre_multiplier = revenue_per_demand_genre / revenue_per_demand_all if revenue_per_demand_all != 0 else 1.0
        
        regression_results[genre] = {
            'coefficient': float(revenue_per_demand_genre),
            'intercept': float(model_genre.intercept_),
            'r2': float(r2_genre),
            'rmse': float(rmse_genre),
            'count': len(genre_data),
            'multiplier': float(genre_multiplier)
        }
        
        print(f"- {genre} regression results:")
        print(f"  * Revenue per demand point: ${revenue_per_demand_genre:.2f}")
        print(f"  * Genre multiplier: {genre_multiplier:.2f}x")
        print(f"  * R² score: {r2_genre:.3f}")
        print(f"  * RMSE: ${rmse_genre:.2f}")
        print(f"  * Shows analyzed: {len(genre_data)}")
    
    return regression_results

def calculate_adjusted_valuations(data, regression_results):
    """
    Calculate genre-adjusted valuations for each show
    """
    print("\nCalculating genre-adjusted valuations...")
    
    if data is None or regression_results is None:
        print("ERROR: Missing data or regression results")
        return None
    
    # Create a copy of the data
    adjusted_data = data.copy()
    
    # Add columns for regression metrics
    adjusted_data['revenue_per_demand'] = adjusted_data['genre'].map(
        lambda g: regression_results.get(g, regression_results['All Shows'])['coefficient']
    )
    
    adjusted_data['genre_multiplier'] = adjusted_data['genre'].map(
        lambda g: regression_results.get(g, regression_results['All Shows'])['multiplier']
    )
    
    # Calculate predicted revenue (based on genre-specific coefficient)
    adjusted_data['predicted_revenue'] = adjusted_data['weighted_demand'] * adjusted_data['revenue_per_demand']
    
    # Calculate error (actual vs predicted)
    adjusted_data['revenue_error'] = adjusted_data['gross_receipts'] - adjusted_data['predicted_revenue']
    adjusted_data['revenue_error_pct'] = adjusted_data['revenue_error'] / adjusted_data['gross_receipts'] * 100
    
    # Calculate genre-adjusted valuation
    # Simple version: Valuation = Demand × Revenue/Demand Point × Genre Multiplier
    adjusted_data['genre_adjusted_valuation'] = adjusted_data['weighted_demand'] * \
                                               adjusted_data['revenue_per_demand'] * \
                                               adjusted_data['genre_multiplier']
    
    print(f"Calculated adjusted valuations for {len(adjusted_data)} shows")
    
    # Summarize by genre
    genre_summary = adjusted_data.groupby('genre').agg({
        'weighted_demand': 'mean',
        'gross_receipts': 'mean',
        'predicted_revenue': 'mean',
        'revenue_error_pct': 'mean',
        'genre_adjusted_valuation': 'mean',
        'genre_multiplier': 'first'
    }).reset_index()
    
    print("\nGenre summary:")
    for _, row in genre_summary.iterrows():
        print(f"- {row['genre']} (multiplier: {row['genre_multiplier']:.2f}x):")
        print(f"  * Average actual revenue: ${row['gross_receipts']:.2f}")
        print(f"  * Average predicted revenue: ${row['predicted_revenue']:.2f}")
        print(f"  * Average error: {row['revenue_error_pct']:.1f}%")
    
    return adjusted_data

def generate_visualizations(data, regression_results, output_dir='../web/images'):
    """Generate visualizations for genre-specific regression analysis"""
    print("\nGenerating genre regression visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Scatter plot of demand vs revenue with regression lines by genre
    plt.figure(figsize=(12, 8))
    
    # Get colors for each genre
    genres = [g for g in regression_results.keys() if g != 'All Shows']
    colors = plt.cm.tab10(np.linspace(0, 1, len(genres)))
    
    # Plot scatter points for each genre
    for i, genre in enumerate(genres):
        genre_data = data[data['genre'] == genre]
        if len(genre_data) > 0:
            plt.scatter(
                genre_data['weighted_demand'], 
                genre_data['gross_receipts'],
                label=f"{genre} (n={len(genre_data)})",
                color=colors[i],
                alpha=0.7,
                s=80
            )
    
    # Add regression lines
    x_range = np.linspace(data['weighted_demand'].min(), data['weighted_demand'].max(), 100)
    
    # Overall regression line
    plt.plot(
        x_range, 
        regression_results['All Shows']['coefficient'] * x_range + regression_results['All Shows']['intercept'],
        'k--', 
        label=f"All Shows (n={regression_results['All Shows']['count']})",
        linewidth=2
    )
    
    # Genre-specific regression lines
    for i, genre in enumerate(genres):
        if regression_results[genre]['count'] >= 3:  # Only plot if we have enough data
            plt.plot(
                x_range, 
                regression_results[genre]['coefficient'] * x_range + regression_results[genre]['intercept'],
                '-', 
                color=colors[i],
                linewidth=2
            )
    
    plt.title('Demand vs. Revenue by Genre with Regression Lines', fontsize=16)
    plt.xlabel('Weighted Demand', fontsize=14)
    plt.ylabel('Revenue (USD)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='upper left')
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/genre_regression_scatter.png"
    plt.savefig(chart_path)
    print(f"Saved genre regression scatter plot to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # 2. Bar chart of genre multipliers
    plt.figure(figsize=(10, 6))
    
    # Sort genres by multiplier
    multipliers = {genre: results['multiplier'] for genre, results in regression_results.items() 
                  if genre != 'All Shows'}
    sorted_genres = sorted(multipliers.items(), key=lambda x: x[1], reverse=True)
    
    genres = [x[0] for x in sorted_genres]
    multiplier_values = [x[1] for x in sorted_genres]
    
    # Create bar chart
    bars = plt.bar(genres, multiplier_values)
    
    # Color bars based on multiplier value
    for i, bar in enumerate(bars):
        if multiplier_values[i] > 1.1:
            bar.set_color('green')
        elif multiplier_values[i] < 0.9:
            bar.set_color('red')
        else:
            bar.set_color('gray')
    
    # Add reference line for baseline multiplier
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, label="Baseline (All Shows)")
    
    plt.title('Genre Multipliers (Relative Revenue per Demand)', fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel('Multiplier', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/genre_multipliers.png"
    plt.savefig(chart_path)
    print(f"Saved genre multipliers chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # 3. Comparison of predicted vs actual revenue by genre
    plt.figure(figsize=(12, 8))
    
    # Group data by genre and calculate actual vs predicted revenue
    genre_comparison = data.groupby('genre').agg({
        'gross_receipts': 'mean',
        'predicted_revenue': 'mean'
    }).reset_index()
    
    # Sort by actual revenue
    genre_comparison = genre_comparison.sort_values('gross_receipts', ascending=False)
    
    # Create bar positions
    x = np.arange(len(genre_comparison))
    width = 0.35
    
    # Create grouped bar chart
    plt.bar(x - width/2, genre_comparison['gross_receipts'], width, label='Actual Revenue', color='royalblue')
    plt.bar(x + width/2, genre_comparison['predicted_revenue'], width, label='Predicted Revenue', color='lightcoral')
    
    plt.title('Actual vs. Predicted Revenue by Genre', fontsize=16)
    plt.xlabel('Genre', fontsize=14)
    plt.ylabel('Average Revenue (USD)', fontsize=14)
    plt.xticks(x, genre_comparison['genre'], rotation=45, ha='right')
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/genre_prediction_comparison.png"
    plt.savefig(chart_path)
    print(f"Saved genre prediction comparison chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # 4. R² and RMSE by genre
    plt.figure(figsize=(12, 6))
    
    # Sort genres by R²
    metrics = {genre: {'r2': results['r2'], 'rmse': results['rmse']} 
              for genre, results in regression_results.items() 
              if genre != 'All Shows' and results['count'] >= 3}
    
    sorted_genres = sorted(metrics.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    genres = [x[0] for x in sorted_genres]
    r2_values = [x[1]['r2'] for x in sorted_genres]
    
    # Create primary y-axis for R²
    fig, ax1 = plt.subplots(figsize=(12, 6))
    
    # Plot R² bars
    bars = ax1.bar(genres, r2_values, color='royalblue', alpha=0.7)
    ax1.set_xlabel('Genre', fontsize=14)
    ax1.set_ylabel('R² Score', fontsize=14, color='royalblue')
    ax1.tick_params(axis='y', labelcolor='royalblue')
    ax1.set_ylim([0, 1.0])  # R² range is 0-1
    
    # Add a line for overall R²
    ax1.axhline(y=regression_results['All Shows']['r2'], color='blue', linestyle='--', 
               label=f"Overall R² = {regression_results['All Shows']['r2']:.2f}")
    
    # Create secondary y-axis for RMSE
    ax2 = ax1.twinx()
    
    # Plot RMSE line
    rmse_values = [x[1]['rmse'] for x in sorted_genres]
    ax2.plot(genres, rmse_values, 'ro-', label='RMSE')
    ax2.set_ylabel('RMSE', fontsize=14, color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add a line for overall RMSE
    ax2.axhline(y=regression_results['All Shows']['rmse'], color='red', linestyle='--',
               label=f"Overall RMSE = {regression_results['All Shows']['rmse']:.2f}")
    
    plt.title('Model Performance Metrics by Genre', fontsize=16)
    plt.xticks(rotation=45, ha='right')
    
    # Create combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/genre_model_performance.png"
    plt.savefig(chart_path)
    print(f"Saved genre model performance chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()

def save_results(regression_results, adjusted_valuations, output_dir='../dataprocessed'):
    """Save the genre regression analysis results"""
    print(f"\nSaving genre regression results to {output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save regression results
    with open(f"{output_dir}/genre_regression.json", 'w') as f:
        json.dump(regression_results, f, indent=2)
    
    # 2. Save genre multipliers (for easier access)
    genre_multipliers = {genre: results['multiplier'] for genre, results in regression_results.items()}
    with open(f"{output_dir}/genre_multipliers.json", 'w') as f:
        json.dump(genre_multipliers, f, indent=2)
    
    # 3. Save adjusted valuations summary
    if adjusted_valuations is not None:
        valuations_summary = adjusted_valuations.to_dict(orient='records')
        with open(f"{output_dir}/genre_adjusted_valuations.json", 'w') as f:
            json.dump(valuations_summary, f, indent=2)
    
    print("Genre regression analysis results saved successfully")

def generate_html_report(regression_results, adjusted_valuations, output_path='../correlation_analysis/genre_regression_report.html'):
    """Generate an HTML report of the genre-specific regression analysis"""
    print(f"\nGenerating HTML report to {output_path}")
    
    # Create an HTML report with findings
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Genre-Specific Regression Analysis</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; line-height: 1.6; }}
            h1, h2, h3 {{ color: #333; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            th {{ background-color: #f2f2f2; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
            .highlight {{ background-color: #e6f7ff; }}
            .chart {{ margin: 20px 0; max-width: 100%; }}
            .multiplier-high {{ color: green; }}
            .multiplier-low {{ color: red; }}
            .multiplier-neutral {{ color: gray; }}
        </style>
    </head>
    <body>
        <h1>Genre-Specific Regression Analysis</h1>
        
        <h2>Overview</h2>
        <p>
            This analysis explores how the relationship between demand metrics and revenue varies by genre.
            By performing separate regression analyses for each genre, we can identify genre-specific multipliers
            that reflect the relative revenue-generating efficiency of content in different categories.
        </p>
        
        <div class="highlight">
            <p><strong>Key Concept:</strong> Genre multipliers represent how efficiently a show in a specific genre converts 
            demand into revenue, relative to the average across all shows.</p>
            <p><strong>Formula:</strong> Genre Multiplier = Revenue per Demand Point (Genre) / Revenue per Demand Point (All Shows)</p>
        </div>
        
        <h2>Regression Visualizations</h2>
        
        <div class="chart">
            <h3>Demand vs. Revenue by Genre</h3>
            <img src="../web/images/genre_regression_scatter.png" alt="Genre Regression Scatter Plot" style="max-width: 100%;">
            <p>This scatter plot shows the relationship between demand and revenue for shows in different genres,
            with genre-specific regression lines illustrating different revenue generation patterns.</p>
        </div>
        
        <div class="chart">
            <h3>Genre Multipliers</h3>
            <img src="../web/images/genre_multipliers.png" alt="Genre Multipliers" style="max-width: 100%;">
            <p>The chart above shows the multiplier for each genre relative to the average revenue per demand point across all shows.</p>
        </div>
        
        <div class="chart">
            <h3>Actual vs. Predicted Revenue by Genre</h3>
            <img src="../web/images/genre_prediction_comparison.png" alt="Genre Prediction Comparison" style="max-width: 100%;">
            <p>This comparison shows how well our genre-specific models predict average revenue for each genre.</p>
        </div>
        
        <div class="chart">
            <h3>Model Performance by Genre</h3>
            <img src="../web/images/genre_model_performance.png" alt="Genre Model Performance" style="max-width: 100%;">
            <p>This chart compares the R² scores and RMSE values across genre-specific regression models,
            indicating how well demand explains revenue variation within each genre.</p>
        </div>
        
        <h2>Genre Multiplier Results</h2>
        <table>
            <tr>
                <th>Genre</th>
                <th>Multiplier</th>
                <th>Revenue per Demand Point</th>
                <th>Show Count</th>
                <th>R² Score</th>
                <th>RMSE</th>
            </tr>
    """
    
    # Add regression results to the table
    genres = [genre for genre in regression_results.keys() if genre != 'All Shows']
    # Sort by multiplier
    genres.sort(key=lambda g: regression_results[g]['multiplier'], reverse=True)
    
    for genre in genres:
        results = regression_results[genre]
        
        # Determine CSS class based on multiplier
        if results['multiplier'] > 1.1:
            multiplier_class = "multiplier-high"
        elif results['multiplier'] < 0.9:
            multiplier_class = "multiplier-low"
        else:
            multiplier_class = "multiplier-neutral"
        
        html_content += f"""
            <tr>
                <td>{genre}</td>
                <td class="{multiplier_class}">{results['multiplier']:.2f}x</td>
                <td>${results['coefficient']:.2f}</td>
                <td>{results['count']}</td>
                <td>{results['r2']:.3f}</td>
                <td>${results['rmse']:.2f}</td>
            </tr>
        """
    
    # Add the baseline (All Shows) row at the bottom
    all_results = regression_results['All Shows']
    html_content += f"""
        <tr class="highlight">
            <td><strong>All Shows (Baseline)</strong></td>
            <td><strong>1.00x</strong></td>
            <td><strong>${all_results['coefficient']:.2f}</strong></td>
            <td><strong>{all_results['count']}</strong></td>
            <td><strong>{all_results['r2']:.3f}</strong></td>
            <td><strong>${all_results['rmse']:.2f}</strong></td>
        </tr>
    """
    
    html_content += """
        </table>
        
        <h2>Sample Genre-Adjusted Valuations</h2>
        <p>
            The table below shows a sample of shows with their genre-adjusted valuations,
            demonstrating the impact of genre multipliers on content valuation.
        </p>
        <table>
            <tr>
                <th>Show</th>
                <th>Genre</th>
                <th>Weighted Demand</th>
                <th>Actual Revenue</th>
                <th>Predicted Revenue</th>
                <th>Genre Multiplier</th>
                <th>Genre-Adjusted Valuation</th>
            </tr>
    """
    
    # Add a sample of shows to the table
    if adjusted_valuations is not None:
        # Get top 5 shows by weighted demand
        top_demand = adjusted_valuations.nlargest(5, 'weighted_demand')
        
        # Get shows with highest genre multipliers
        high_multiplier = adjusted_valuations.nlargest(5, 'genre_multiplier')
        
        # Get shows with lowest genre multipliers
        low_multiplier = adjusted_valuations.nsmallest(5, 'genre_multiplier')
        
        # Combine and remove duplicates
        sample_shows = pd.concat([top_demand, high_multiplier, low_multiplier]).drop_duplicates()
        
        for _, show in sample_shows.iterrows():
            # Determine CSS class based on multiplier
            if show['genre_multiplier'] > 1.1:
                multiplier_class = "multiplier-high"
            elif show['genre_multiplier'] < 0.9:
                multiplier_class = "multiplier-low"
            else:
                multiplier_class = "multiplier-neutral"
            
            html_content += f"""
                <tr>
                    <td>{show['show_name']}</td>
                    <td>{show['genre']}</td>
                    <td>{show['weighted_demand']:.3f}</td>
                    <td>${show['gross_receipts']:.2f}</td>
                    <td>${show['predicted_revenue']:.2f}</td>
                    <td class="{multiplier_class}">{show['genre_multiplier']:.2f}x</td>
                    <td>${show['genre_adjusted_valuation']:.2f}</td>
                </tr>
            """
    
    html_content += """
        </table>
        
        <h2>Methodology</h2>
        <p>
            The genre-specific regression analysis follows these steps:
        </p>
        <ol>
            <li>Assign genres to shows based on available metadata or predefined mappings</li>
            <li>Perform linear regression analysis for each genre, predicting revenue from demand</li>
            <li>Calculate the revenue per demand point (regression coefficient) for each genre</li>
            <li>Derive genre multipliers by comparing genre-specific coefficients to the overall average</li>
            <li>Apply these multipliers to adjust content valuations based on genre</li>
        </ol>
        
        <h2>Applications</h2>
        <p>
            Genre-specific regression provides several advantages in content valuation:
        </p>
        <ul>
            <li>More accurate revenue predictions by accounting for genre-specific factors</li>
            <li>Recognition of genres that more efficiently convert audience demand to revenue</li>
            <li>Ability to compare shows across genres on an adjusted basis</li>
            <li>Better calibration of acquisition pricing by genre</li>
            <li>Identification of potentially undervalued content in high-multiplier genres</li>
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
    print("GENRE-SPECIFIC REGRESSION ANALYSIS")
    print("=" * 60)
    
    # Create output directory for visualizations
    os.makedirs('../web/images', exist_ok=True)
    
    # Load data
    demand_data, revenue_data, metadata = load_data()
    
    # Preprocess data
    processed_data = preprocess_data(demand_data, revenue_data, metadata)
    
    # Perform regression by genre
    regression_results = perform_regression_by_genre(processed_data)
    
    # Calculate adjusted valuations
    adjusted_valuations = calculate_adjusted_valuations(processed_data, regression_results)
    
    # Generate visualizations
    generate_visualizations(processed_data, regression_results)
    
    # Save results
    save_results(regression_results, adjusted_valuations)
    
    # Generate HTML report
    generate_html_report(regression_results, adjusted_valuations)
    
    print("\nGenre-specific regression analysis complete!")
    print("=" * 60) 