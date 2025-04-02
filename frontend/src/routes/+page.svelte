<script lang="ts">
	import { onMount } from 'svelte';

	// Interface for the structure within valuations.json
	interface ValuationEntry {
		show_name: string;
		current_demand: number;
		weighted_demand: number; // Keep for potential future use
		momentum_factor: number;
		revenue_per_demand: number;
		genre: string;
		genre_multiplier: number;
		optimal_lag_quarters: number; // Placeholder, will be overridden by lag_results
		lag_correlation: number; // Placeholder, will be overridden by lag_results
		volatility_discount: number;
		predicted_valuation: number;
	}

	// Interface for the structure within lag_results.json
	interface LagResult {
		optimal_lag_quarters: number;
		correlation: number;
	}

	// Combined interface for display
	interface DisplayData extends ValuationEntry { 
        // Inherits all from ValuationEntry, lag/corr will be populated correctly
    }

	let displayData: DisplayData[] = []; // Use this for the table
	let loading: boolean = true;
	let error: string | null = null;
	let sortColumn: keyof DisplayData | null = 'predicted_valuation'; // Default sort
	let sortDirection: 'asc' | 'desc' = 'desc'; // Default sort direction

	onMount(async () => {
		try {
			// Fetch both files concurrently
			const [valuationsResponse, lagResponse] = await Promise.all([
				fetch('/valuations.json'),
				fetch('/lag_results.json')
			]);

			if (!valuationsResponse.ok) {
				throw new Error(`Valuations fetch error! status: ${valuationsResponse.status}`);
			}
			if (!lagResponse.ok) {
				throw new Error(`Lag results fetch error! status: ${lagResponse.status}`);
			}

			const valuationsData: ValuationEntry[] = await valuationsResponse.json();
			const lagData: { [key: string]: LagResult } = await lagResponse.json();

            // Merge the data
            displayData = valuationsData.map(valuation => {
                const showLag = lagData[valuation.show_name];
                return {
                    ...valuation,
                    // Override with actual lag data if found, otherwise keep defaults/placeholders from valuation file
                    optimal_lag_quarters: showLag?.optimal_lag_quarters ?? valuation.optimal_lag_quarters ?? 0,
                    lag_correlation: showLag?.correlation ?? valuation.lag_correlation ?? 0
                };
            });

			sortData(); // Apply initial sort to the merged displayData
		} catch (e) {
			console.error("Failed to load or merge data:", e);
			error = e instanceof Error ? e.message : 'Failed to load or process data.';
		} finally {
			loading = false;
		}
	});

	function formatCurrency(value: number): string {
		if (value === 0) return '$0.00M';
		return `$${(value / 1000000).toFixed(2)}M`;
	}

	function formatNumber(value: number | undefined | null, digits: number = 2): string {
        if (value === undefined || value === null) return 'N/A';
		return value.toFixed(digits);
	}

	function handleSort(column: keyof DisplayData) { // Update keyof type
		if (sortColumn === column) {
			sortDirection = sortDirection === 'asc' ? 'desc' : 'asc';
		} else {
			sortColumn = column;
			sortDirection = 'desc'; // Default to descending for new column
		}
		sortData();
	}

    function sortData() {
        if (!sortColumn) return;

        displayData.sort((a, b) => { // Sort displayData
            const aValue = a[sortColumn!];
            const bValue = b[sortColumn!];

            if (typeof aValue === 'string' && typeof bValue === 'string') {
                return sortDirection === 'asc' ? aValue.localeCompare(bValue) : bValue.localeCompare(aValue);
            } else if (typeof aValue === 'number' && typeof bValue === 'number') {
                return sortDirection === 'asc' ? aValue - bValue : bValue - aValue;
            } else {
                if (aValue < bValue) return sortDirection === 'asc' ? -1 : 1;
                if (aValue > bValue) return sortDirection === 'asc' ? 1 : -1;
                return 0;
            }
        });
		// Trigger reactivity by reassigning the array
		displayData = [...displayData]; // Update displayData
    }

    // Update keyof type here too
	const columns: { key: keyof DisplayData; label: string; format?: (value: any) => string }[] = [
		{ key: 'show_name', label: 'Show Name' },
		{ key: 'predicted_valuation', label: 'Valuation', format: formatCurrency },
        { key: 'current_demand', label: 'Current Demand', format: (v) => formatNumber(v, 2) },
		{ key: 'momentum_factor', label: 'Momentum', format: (v) => formatNumber(v, 2) + 'x' },
		{ key: 'revenue_per_demand', label: 'Rev/Demand (Q)', format: formatCurrency },
		{ key: 'optimal_lag_quarters', label: 'Lag (Qtrs)' }, // Now correctly populated
		{ key: 'lag_correlation', label: 'Lag Corr.', format: (v) => formatNumber(v, 2) }, // Now correctly populated
        { key: 'volatility_discount', label: 'Vol. Discount', format: (v) => formatNumber(v, 2) },
		{ key: 'genre', label: 'Genre' },
		{ key: 'genre_multiplier', label: 'Genre Mult.', format: (v) => formatNumber(v, 2) + 'x' },
	];

</script>

<div class="container mx-auto p-4 md:p-8 bg-gray-50 min-h-screen">
	<h1 class="text-3xl font-bold mb-6 text-gray-800">Content Demand Index Valuations</h1>

	{#if loading}
		<p class="text-center text-gray-600">Loading valuation data...</p>
	{:else if error}
		<p class="text-center text-red-600">Error loading data: {error}</p>
		<p class="text-center text-gray-500 mt-2">Ensure <code>valuations.json</code> and <code>lag_results.json</code> exist in the <code>/static</code> folder and are valid JSON.</p>
	{:else if displayData.length === 0}
		<p class="text-center text-gray-600">No valuation data found. Run the backend script to generate data and copy it to <code>frontend/static/</code>.</p>
	{:else}
		<div class="overflow-x-auto shadow-md rounded-lg bg-white">
			<table class="min-w-full divide-y divide-gray-200">
				<thead class="bg-gray-100">
					<tr>
						{#each columns as col}
							<th
								scope="col"
								class="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider cursor-pointer whitespace-nowrap hover:bg-gray-200 transition-colors duration-150"
                                on:click={() => handleSort(col.key)}
							>
								{col.label}
                                {#if sortColumn === col.key}
                                    <span class="ml-1">{sortDirection === 'asc' ? '↑' : '↓'}</span>
                                {/if}
							</th>
						{/each}
					</tr>
				</thead>
				<tbody class="bg-white divide-y divide-gray-200">
                    <!-- Iterate over displayData -->
					{#each displayData as item (item.show_name)}
						<tr class="hover:bg-gray-50 transition-colors duration-150">
							{#each columns as col}
								<td class="px-4 py-3 whitespace-nowrap text-sm text-gray-700">
                                    <!-- Access item properties -->
									{col.format ? col.format(item[col.key]) : item[col.key] ?? 'N/A'}
								</td>
							{/each}
						</tr>
					{/each}
				</tbody>
			</table>
		</div>
	{/if}
</div>

<style>
	/* Optional: Add any global styles or overrides here if needed */
	/* Tailwind handles most styling via utility classes */
</style>
