/**
 * Axios HTTP client for BioSense360.
 *
 * Base URL resolution order:
 *   1. VITE_API_URL env var (set at build time), if provided.
 *   2. In a production build, the deployed Render backend.
 *   3. Otherwise localhost:8000 for local development.
 *
 * A 60-second timeout is applied so the first request after the free-tier
 * backend has spun down (cold start, ~50s) still completes instead of
 * failing with a misleading "Network Error".
 */

import axios from 'axios';

// Render's free backend service URL — used when no VITE_API_URL is supplied
// to the production build (so the deployed site never falls back to localhost).
const PROD_API_URL = 'https://biobiosense360-backend.onrender.com';

const baseURL =
  import.meta.env.VITE_API_URL ||
  (import.meta.env.PROD ? PROD_API_URL : 'http://localhost:8000');

const client = axios.create({
  baseURL,
  timeout: 60000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Response interceptor: normalise error shape for components
client.interceptors.response.use(
  (response) => response,
  (error) => {
    const message =
      error.response?.data?.detail ||
      error.response?.data?.message ||
      error.message ||
      'An unexpected error occurred';
    return Promise.reject(new Error(message));
  }
);

export default client;
