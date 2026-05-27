/**
 * useRiskSummary — fetches daily risk summary for a given date and optional station.
 * Re-fetches whenever date or stationId changes.
 */

import { useEffect, useState } from 'react';
import client from '../api/client';

export function useRiskSummary(date, stationId) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!date) return;

    let cancelled = false;

    async function fetchSummary() {
      setLoading(true);
      setError(null);
      setSummary(null);

      try {
        const params = { date };
        if (stationId) params.station_id = stationId;

        const { data } = await client.get('/api/risk-summary', { params });
        if (!cancelled) {
          setSummary(data);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err.message);
        }
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    }

    fetchSummary();
    return () => { cancelled = true; };
  }, [date, stationId]);

  return { summary, loading, error };
}
