import pandas as pd
import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import os

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

def define_territory_weights():
    """Define territory importance weights based on market size and strategic value"""
    print("\nDefining territory weights...")
    
    # Define default territory weights
    # These weights reflect the relative importance of each market
    territory_weights = {
        'US': 0.40,        # United States (largest market)
        'UK': 0.20,        # United Kingdom
        'Canada': 0.15,    # Canada
        'Australia': 0.15, # Australia
        'France': 0.10,    # France
        'Default': 0.05    # Default weight for any other territory
    }
    
    print("Territory weights defined:")
    for territory, weight in territory_weights.items():
        print(f"- {territory}: {weight*100:.1f}%")
    
    return territory_weights

def calculate_weighted_demand(demand_data, territory_weights):
    """Calculate territory-weighted demand for all shows"""
    print("\nCalculating weighted demand metrics...")
    
    if demand_data is None:
        print("ERROR: No demand data available for weighting")
        return None
    
    # Check if territory column exists
    if 'territory' not in demand_data.columns:
        print("WARNING: No territory column found. Using raw demand scores.")
        demand_data['weighted_demand'] = demand_data['demand_score']
        demand_data['weight_applied'] = 1.0
        return demand_data
    
    # Create a copy to avoid modifying the original data
    weighted_data = demand_data.copy()
    
    # Add a column for the applied weight
    weighted_data['weight_applied'] = weighted_data['territory'].map(
        lambda t: territory_weights.get(t, territory_weights['Default'])
    )
    
    # Calculate weighted demand
    weighted_data['weighted_demand'] = weighted_data['demand_score'] * weighted_data['weight_applied']
    
    print(f"Applied weights to {len(weighted_data)} demand records")
    
    # Summarize the effect of weighting
    territories = weighted_data['territory'].unique()
    print(f"Found {len(territories)} territories in the data")
    
    # Calculate the average weight effect by territory
    territory_summary = weighted_data.groupby('territory').agg({
        'demand_score': 'mean',
        'weighted_demand': 'mean',
        'weight_applied': 'first'
    }).reset_index()
    
    print("\nTerritory weighting summary:")
    for _, row in territory_summary.iterrows():
        print(f"- {row['territory']}: {row['demand_score']:.3f} → {row['weighted_demand']:.3f} (weight: {row['weight_applied']:.2f})")
    
    return weighted_data

def analyze_territory_impact(weighted_data):
    """Analyze the impact of territory weighting on show rankings"""
    print("\nAnalyzing territory weighting impact...")
    
    if weighted_data is None:
        print("ERROR: No weighted data available for analysis")
        return None
    
    # Calculate average demand by show (both raw and weighted)
    show_summary = weighted_data.groupby('show_name').agg({
        'demand_score': 'mean',
        'weighted_demand': 'mean'
    }).reset_index()
    
    # Sort by raw demand and weighted demand to compare rankings
    raw_ranking = show_summary.sort_values('demand_score', ascending=False).reset_index(drop=True)
    raw_ranking['raw_rank'] = raw_ranking.index + 1
    
    weighted_ranking = show_summary.sort_values('weighted_demand', ascending=False).reset_index(drop=True)
    weighted_ranking['weighted_rank'] = weighted_ranking.index + 1
    
    # Merge the rankings
    combined = pd.merge(
        raw_ranking[['show_name', 'raw_rank', 'demand_score']], 
        weighted_ranking[['show_name', 'weighted_rank', 'weighted_demand']],
        on='show_name'
    )
    
    # Calculate rank change
    combined['rank_change'] = combined['raw_rank'] - combined['weighted_rank']
    
    print("\nTop 5 shows by raw demand:")
    for i, row in raw_ranking.head(5).iterrows():
        print(f"{i+1}. {row['show_name']}: {row['demand_score']:.3f}")
    
    print("\nTop 5 shows by weighted demand:")
    for i, row in weighted_ranking.head(5).iterrows():
        print(f"{i+1}. {row['show_name']}: {row['weighted_demand']:.3f}")
    
    print("\nShows with biggest rank changes due to territory weighting:")
    biggest_changes = combined.sort_values('rank_change', ascending=False)
    for _, row in biggest_changes.head(5).iterrows():
        direction = "up" if row['rank_change'] > 0 else "down"
        print(f"- {row['show_name']}: moved {abs(row['rank_change'])} positions {direction}")
    
    return combined

def calculate_territory_contribution(weighted_data):
    """Calculate the contribution of each territory to overall weighted demand"""
    print("\nCalculating territory contribution to overall demand...")
    
    if weighted_data is None:
        print("ERROR: No weighted data available for analysis")
        return None
    
    # Calculate total weighted demand by territory
    territory_contribution = weighted_data.groupby('territory').agg({
        'weighted_demand': 'sum',
        'weight_applied': 'first'
    }).reset_index()
    
    # Calculate percentage of total
    total_weighted_demand = territory_contribution['weighted_demand'].sum()
    territory_contribution['contribution_pct'] = territory_contribution['weighted_demand'] / total_weighted_demand * 100
    
    # Sort by contribution
    territory_contribution = territory_contribution.sort_values('contribution_pct', ascending=False)
    
    print("Territory contribution to total weighted demand:")
    for _, row in territory_contribution.iterrows():
        print(f"- {row['territory']}: {row['contribution_pct']:.1f}% (weight: {row['weight_applied']:.2f})")
    
    return territory_contribution

def generate_visualizations(weighted_data, territory_contribution, ranking_changes, output_dir='../web/images'):
    """Generate visualizations for market-weighted demand analysis"""
    print("\nGenerating market weighting visualizations...")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Territory Contribution Pie Chart
    plt.figure(figsize=(10, 8))
    plt.pie(
        territory_contribution['contribution_pct'], 
        labels=territory_contribution['territory'],
        autopct='%1.1f%%',
        startangle=90,
        shadow=True,
        explode=[0.05] * len(territory_contribution)
    )
    plt.title('Territory Contribution to Total Weighted Demand', fontsize=16)
    plt.axis('equal')
    
    # Save figure
    chart_path = f"{output_dir}/territory_contribution_pie.png"
    plt.savefig(chart_path)
    print(f"Saved territory contribution pie chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # 2. Ranking Changes - Top Movers
    top_movers = ranking_changes.sort_values('rank_change', ascending=False)
    movers_to_show = 10  # Show the 10 shows with biggest changes
    
    plt.figure(figsize=(12, 8))
    
    # Show both positive and negative movers
    positive_movers = top_movers.head(movers_to_show)
    negative_movers = top_movers.tail(movers_to_show).iloc[::-1]  # Reverse order
    
    # Combine both sets
    movers = pd.concat([positive_movers.head(5), negative_movers.head(5)])
    
    # Create bars with different colors for up/down movement
    colors = ['green' if x > 0 else 'red' for x in movers['rank_change']]
    plt.bar(movers['show_name'], movers['rank_change'], color=colors)
    
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.title('Shows with Biggest Rank Changes Due to Market Weighting', fontsize=16)
    plt.xlabel('Show', fontsize=14)
    plt.ylabel('Rank Change (positions)', fontsize=14)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/rank_changes_bar.png"
    plt.savefig(chart_path)
    print(f"Saved rank changes bar chart to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # 3. Weight vs Raw Score Scatter Plot
    plt.figure(figsize=(12, 8))
    
    # Calculate average score by show
    show_avg = weighted_data.groupby('show_name').agg({
        'demand_score': 'mean',
        'weighted_demand': 'mean'
    }).reset_index()
    
    # Create scatter plot
    plt.scatter(
        show_avg['demand_score'], 
        show_avg['weighted_demand'],
        alpha=0.7,
        s=80,
        c=show_avg['weighted_demand']/show_avg['demand_score'],
        cmap='viridis'
    )
    
    # Add color bar to show weight impact
    cbar = plt.colorbar()
    cbar.set_label('Weight Impact Ratio', rotation=270, labelpad=20)
    
    # Add reference line (y=x)
    max_val = max(show_avg['demand_score'].max(), show_avg['weighted_demand'].max())
    plt.plot([0, max_val], [0, max_val], 'k--', alpha=0.5)
    
    # Annotate a few interesting points
    for _, row in show_avg.nlargest(5, 'weighted_demand').iterrows():
        plt.annotate(
            row['show_name'],
            xy=(row['demand_score'], row['weighted_demand']),
            xytext=(10, 5),
            textcoords='offset points'
        )
    
    plt.title('Raw Demand vs. Weighted Demand by Show', fontsize=16)
    plt.xlabel('Raw Demand Score', fontsize=14)
    plt.ylabel('Weighted Demand Score', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/raw_vs_weighted_scatter.png"
    plt.savefig(chart_path)
    print(f"Saved raw vs weighted scatter plot to {chart_path}")
    
    # Close figure to free memory
    plt.close()
    
    # 4. Heatmap of Territory by Show
    # Get the top 10 shows by weighted demand
    top_shows = weighted_data.groupby('show_name')['weighted_demand'].mean().nlargest(10).index
    
    # Filter data for these shows
    top_shows_data = weighted_data[weighted_data['show_name'].isin(top_shows)]
    
    # Create pivot table: shows x territories
    pivot = top_shows_data.pivot_table(
        index='show_name',
        columns='territory',
        values='demand_score',
        aggfunc='mean'
    ).fillna(0)
    
    # Create heatmap
    plt.figure(figsize=(14, 8))
    sns.heatmap(
        pivot, 
        annot=True, 
        cmap='YlGnBu', 
        fmt='.2f',
        cbar_kws={'label': 'Raw Demand Score'}
    )
    
    plt.title('Demand Score by Territory for Top Shows', fontsize=16)
    plt.tight_layout()
    
    # Save figure
    chart_path = f"{output_dir}/territory_show_heatmap.png"
    plt.savefig(chart_path)
    print(f"Saved territory-show heatmap to {chart_path}")
    
    # Close figure to free memory
    plt.close()

def save_results(weighted_data, territory_weights, territory_contribution, output_dir='../dataprocessed'):
    """Save the market weighting analysis results"""
    print(f"\nSaving market weighting analysis results to {output_dir}")
    
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Save territory weights
    with open(f"{output_dir}/territory_weights.json", 'w') as f:
        json.dump(territory_weights, f, indent=2)
    
    # 2. Save territory contribution
    territory_contribution_dict = territory_contribution.to_dict(orient='records')
    with open(f"{output_dir}/territory_contribution.json", 'w') as f:
        json.dump(territory_contribution_dict, f, indent=2)
    
    # 3. Save aggregated weighted demand by show
    show_weighted_demand = weighted_data.groupby('show_name').agg({
        'demand_score': 'mean',
        'weighted_demand': 'mean'
    }).reset_index()
    
    show_weighted_dict = show_weighted_demand.to_dict(orient='records')
    with open(f"{output_dir}/show_weighted_demand.json", 'w') as f:
        json.dump(show_weighted_dict, f, indent=2)
    
    print("Market weighting analysis results saved successfully")

def generate_html_report(weighted_data, territory_weights, territory_contribution, rank_changes, output_path='../correlation_analysis/market_weighted_report.html'):
    """Generate an HTML report of the market weighting analysis"""
    print(f"\nGenerating HTML report to {output_path}")
    
    # Create an HTML report with findings
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Market-Weighted Demand Analysis</title>
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
        <h1>Market-Weighted Demand Analysis</h1>
        
        <h2>Overview</h2>
        <p>
            This analysis applies territory-specific weights to demand metrics, reflecting the relative 
            importance of different markets. The weighting system adjusts raw demand scores based on 
            strategic priorities and market size.
        </p>
        
        <h2>Territory Weights</h2>
        <table>
            <tr>
                <th>Territory</th>
                <th>Weight</th>
                <th>Percentage</th>
            </tr>
    """
    
    # Add territory weights to the table
    for territory, weight in territory_weights.items():
        html_content += f"""
            <tr>
                <td>{territory}</td>
                <td>{weight:.2f}</td>
                <td>{weight*100:.1f}%</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Territory Contribution</h2>
        <p>
            After weighting, each territory contributes differently to the overall weighted demand.
            The chart below shows the proportion of total weighted demand by territory.
        </p>
        
        <div class="chart">
            <img src="../web/images/territory_contribution_pie.png" alt="Territory Contribution" style="max-width: 100%;">
        </div>
        
        <h2>Impact on Show Rankings</h2>
        <p>
            Territory weighting affects the relative importance of shows based on their performance in 
            high-priority markets. The chart below illustrates shows with the most significant rank changes.
        </p>
        
        <div class="chart">
            <img src="../web/images/rank_changes_bar.png" alt="Rank Changes" style="max-width: 100%;">
        </div>
        
        <h2>Raw vs. Weighted Demand</h2>
        <p>
            Shows perform differently across territories. The scatter plot below compares raw demand 
            scores with weighted demand scores, highlighting the effect of territory weighting.
        </p>
        
        <div class="chart">
            <img src="../web/images/raw_vs_weighted_scatter.png" alt="Raw vs Weighted Scatter" style="max-width: 100%;">
        </div>
        
        <h2>Territory Performance by Show</h2>
        <p>
            The heatmap below shows demand performance across territories for top shows, revealing 
            market-specific patterns.
        </p>
        
        <div class="chart">
            <img src="../web/images/territory_show_heatmap.png" alt="Territory-Show Heatmap" style="max-width: 100%;">
        </div>
        
        <h2>Top Rank Changers</h2>
        <p>
            The table below lists shows with the most significant changes in ranking after territory weighting.
        </p>
        
        <table>
            <tr>
                <th>Show</th>
                <th>Raw Rank</th>
                <th>Weighted Rank</th>
                <th>Rank Change</th>
                <th>Raw Demand</th>
                <th>Weighted Demand</th>
            </tr>
    """
    
    # Add top rank changers to the table
    top_movers = rank_changes.sort_values('rank_change', ascending=False)
    for _, row in top_movers.head(10).iterrows():
        highlight = " class='highlight'" if abs(row['rank_change']) > 2 else ""
        html_content += f"""
            <tr{highlight}>
                <td>{row['show_name']}</td>
                <td>{row['raw_rank']}</td>
                <td>{row['weighted_rank']}</td>
                <td>{row['rank_change']}</td>
                <td>{row['demand_score']:.3f}</td>
                <td>{row['weighted_demand']:.3f}</td>
            </tr>
        """
    
    html_content += """
        </table>
        
        <h2>Methodology</h2>
        <p>
            The market-weighted demand analysis follows these steps:
        </p>
        <ol>
            <li>Define territory weights based on strategic importance and market size</li>
            <li>Apply weights to raw demand scores for each territory</li>
            <li>Calculate aggregate weighted demand for each show</li>
            <li>Compare rankings before and after weighting to identify impact</li>
        </ol>
        
        <h2>Applications</h2>
        <p>
            Market-weighted demand metrics provide several advantages:
        </p>
        <ul>
            <li>More accurately reflects the commercial value of content</li>
            <li>Prioritizes performance in strategically important markets</li>
            <li>Provides insight into territory-specific demand patterns</li>
            <li>Enables more targeted acquisition and distribution strategies</li>
            <li>Helps identify content with strong performance in key territories</li>
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
    print("MARKET-WEIGHTED DEMAND ANALYSIS")
    print("=" * 60)
    
    # Create output directory for visualizations
    os.makedirs('../web/images', exist_ok=True)
    
    # Load data
    demand_data = load_data()
    
    # Define territory weights
    territory_weights = define_territory_weights()
    
    # Calculate weighted demand
    weighted_data = calculate_weighted_demand(demand_data, territory_weights)
    
    # Analyze territory impact
    rank_changes = analyze_territory_impact(weighted_data)
    
    # Calculate territory contribution
    territory_contribution = calculate_territory_contribution(weighted_data)
    
    # Generate visualizations
    generate_visualizations(weighted_data, territory_contribution, rank_changes)
    
    # Save results
    save_results(weighted_data, territory_weights, territory_contribution)
    
    # Generate HTML report
    generate_html_report(weighted_data, territory_weights, territory_contribution, rank_changes)
    
    print("\nMarket-weighted demand analysis complete!")
    print("=" * 60) 