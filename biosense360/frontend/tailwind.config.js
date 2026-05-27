/** @type {import('tailwindcss').Config} */
export default {
  content: [
    './index.html',
    './src/**/*.{js,jsx,ts,tsx}',
  ],
  theme: {
    extend: {
      colors: {
        // BioSense360 design tokens
        bg: {
          page: '#0f1117',
          card: '#1a1f2e',
        },
        border: {
          card: '#2d3548',
        },
        accent: {
          blue: '#38bdf8',
        },
        text: {
          primary: '#f1f5f9',
          muted: '#94a3b8',
        },
        // Thermal comfort scale
        thermal: {
          comfortable:    '#22c55e',
          caution:        '#eab308',
          'extreme-caution': '#f97316',
          danger:         '#ef4444',
          'extreme-danger': '#7f1d1d',
        },
        // Risk level scale
        risk: {
          low:      '#22c55e',
          moderate: '#eab308',
          high:     '#f97316',
          critical: '#7f1d1d',
        },
      },
      fontFamily: {
        sans: ['Inter', 'system-ui', 'sans-serif'],
        mono: ['JetBrains Mono', 'monospace'],
      },
    },
  },
  plugins: [],
};
