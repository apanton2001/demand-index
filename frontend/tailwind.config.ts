import type { Config } from 'tailwindcss'
import typography from '@tailwindcss/typography';

export default {
	content: ['./src/**/*.{html,js,svelte,ts}'],
	theme: {
		extend: {},
	},
	plugins: [
		typography, // Add the typography plugin we selected
	],
} satisfies Config; 