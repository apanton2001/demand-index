import adapter from '@sveltejs/adapter-static';
import { vitePreprocess } from '@sveltejs/vite-plugin-svelte';

/** @type {import('@sveltejs/kit').Config} */
const config = {
	// Consult https://svelte.dev/docs/kit/integrations
	// for more information about preprocessors
	preprocess: vitePreprocess(),

	kit: {
		// adapter-auto only supports some environments, see https://svelte.dev/docs/kit/adapter-auto for a list.
		// If your environment is not supported, or you settled on a specific environment, switch out the adapter.
		// See https://svelte.dev/docs/kit/adapters for more information about adapters.
		adapter: adapter({
			// default options are suitable for Vercel static deployment
			pages: 'build', // Output directory
			assets: 'build',
			fallback: undefined, // No fallback needed for pure static
			precompress: false,
			strict: true
		}),
		// Ensure paths are relative for static export
		paths: {
			base: process.env.NODE_ENV === 'production' ? '' : '' // Keep empty for root deployment
		}
	}
};

export default config;
