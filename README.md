# Content Demand Index - Proof of Concept

## Purpose

This project is an internal proof-of-concept exploring the potential correlations between available digital demand signals (e.g., weighted scores derived from Parrot Analytics data, potentially web scrapes in the future) and internal quarterly revenue data for TV content. 

**Goal:** To develop a supplementary internal metric that *might* aid in understanding content value trends, supplementing commercial tools like Parrot Analytics Demand 360. **It is NOT intended as a replacement** for established commercial datasets or valuation methods at this stage.

## Methodology (High-Level)

1.  **Data Ingestion:** The core script (`scripts/process_data.py`) loads data from `dataraw/show_analysis.xlsx`, expecting specific sheets:
    *   `demand_data`: Contains monthly demand scores per show/territory (requires columns: `show_name`, `date`, `demand_score`, `territory`).
    *   `revenue_data`: Contains **quarterly** gross receipts per show (requires columns: `show_name`, `date`, `gross_receipts`). **Crucially, this data must be clean quarterly data; do not input monthly revenue derived from demand.**
    *   `show_metadata`: Optional sheet for genre information (requires columns: `show`, `genre`).

2.  **Demand Weighting:** Applies weights to demand scores based on territory importance (defaults defined in the script).

3.  **Quarterly Alignment:** Aggregates monthly weighted demand to a quarterly frequency (Quarter End).

4.  **Time Lag Correlation:** Calculates the correlation between quarterly revenue and lagged quarterly demand (up to 4 quarters) to find the 'optimal' lag period where the correlation is strongest for each show.

5.  **Momentum Calculation:** Calculates a momentum factor based on recent monthly demand trends (comparing the last 90 days vs. the previous 90 days).

6.  **Valuation Formula:** Calculates a predicted valuation using the formula:
    *Value = Current Monthly Demand × (Avg Quarterly Revenue / Avg Quarterly Demand) × Momentum Factor × Genre Multiplier × Volatility Discount*
    *   *Genre Multiplier*: Hardcoded factor based on genre (defined in `calculate_valuations`).
    *   *Volatility Discount*: Derived from the strength of the quarterly lag correlation (higher correlation = lower discount).

7.  **Output:** Saves the results into JSON files in the `dataprocessed/` directory:
    *   `valuations.json`: Final calculated metrics and valuation per show.
    *   `lag_results.json`: Optimal quarterly lag and correlation per show.
    *   `momentum_results.json`: Calculated momentum factor per show.

## Current Status & Limitations

*   **Proof of Concept:** This is an early-stage internal tool.
*   **Data Dependency:** Requires clean, properly formatted input data in `dataraw/show_analysis.xlsx`. **Only processes shows with available quarterly revenue data.**
*   **Limited Scope:** The model is simplified and does not capture all factors influencing content value (e.g., piracy, market specifics, contract terms).
*   **Correlation != Causation:** Any observed correlations require careful interpretation and are not definitive proof of a causal link.
*   **Hardcoded Factors:** Genre multipliers and territory weights are currently hardcoded defaults and may need refinement.

## How to Use

1.  **Prepare Data:** Ensure your `demand_data` and `revenue_data` (quarterly only!) sheets are correctly formatted in `dataraw/show_analysis.xlsx`.
2.  **Install Dependencies:** If you haven't already, install `pandas` (likely via `pip install pandas openpyxl`).
3.  **Run Script:** Open a terminal/command prompt in the project's root directory (`demand-index`) and run:
    ```bash
    python scripts/process_data.py
    ```
4.  **View Output:** Check the console for logs and find the generated JSON files in the `dataprocessed/` folder.

## Interpreting Output (`valuations.json`)

*   `show_name`: The name of the show.
*   `current_demand`: The most recent monthly weighted demand score.
*   `momentum_factor`: Recent demand trend (>1 is growing).
*   `revenue_per_demand`: The calculated ratio of average quarterly revenue to average quarterly demand. **Shows with $0 for this likely lacked sufficient revenue data.**
*   `genre`, `genre_multiplier`: Genre assigned and its corresponding valuation multiplier.
*   `optimal_lag_quarters`: The lag (in quarters) where quarterly demand best correlated with quarterly revenue.
*   `lag_correlation`: The correlation coefficient at the optimal lag.
*   `volatility_discount`: Discount factor based on correlation strength (closer to 1 means less discount).
*   `predicted_valuation`: The final calculated valuation based on the formula. Treat this as an *indicative internal metric*, not a market valuation.

## Frontend Viewer (Optional)

A simple SvelteKit frontend is available in the `frontend/` directory to view the `valuations.json` results in a table.

1.  **Ensure `valuations.json` exists:** Run the main Python script first.
2.  **Copy Data:** Copy the generated `dataprocessed/valuations.json` file to `frontend/static/valuations.json`.
3.  **Navigate:** `cd frontend`
4.  **Install Dependencies:** `npm install` (or `yarn` or `pnpm install`)
5.  **Run Locally:** `npm run dev`
6.  Open the provided localhost URL in your browser.

# demand-index
 
