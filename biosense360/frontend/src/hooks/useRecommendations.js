/**
 * useRecommendations — wrapper around the human and irrigation recommendation endpoints.
 *
 * Returns imperative fetch functions rather than auto-fetching on mount,
 * because these endpoints are triggered by form submissions.
 */

import { useState } from 'react';
import client from '../api/client';

export function useHumanRecommendation() {
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Fetch a human recommendation.
   *
   * @param {object} params - { humidex, comfort_class, age_group, activity_type }
   */
  async function fetchRecommendation(params) {
    setLoading(true);
    setError(null);
    setRecommendation(null);

    try {
      const { data } = await client.post('/api/recommendations/human', params);
      setRecommendation(data);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }

  return { recommendation, loading, error, fetchRecommendation };
}

export function useIrrigationRecommendation() {
  const [recommendation, setRecommendation] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  /**
   * Fetch an irrigation recommendation.
   *
   * @param {object} params - { date, station_id, plant_type, soil_type }
   */
  async function fetchRecommendation(params) {
    setLoading(true);
    setError(null);
    setRecommendation(null);

    try {
      const { data } = await client.post('/api/recommendations/irrigation', params);
      setRecommendation(data);
      return data;
    } catch (err) {
      setError(err.message);
      return null;
    } finally {
      setLoading(false);
    }
  }

  return { recommendation, loading, error, fetchRecommendation };
}
