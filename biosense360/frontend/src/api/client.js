/**
 * Axios HTTP client for BioSense360.
 *
 * Base URL is set from the VITE_API_URL environment variable,
 * defaulting to localhost:8000 for local development.
 *
 * A 10-second timeout is applied to all requests to prevent
 * the UI from hanging when the backend is slow or unreachable.
 */

import axios from 'axios';

const client = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  timeout: 10000,
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
