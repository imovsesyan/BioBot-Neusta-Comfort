/**
 * useForecast — fetches 24-hour time-slot forecast for a given station and date.
 * Re-fetches whenever stationId or date changes.
 */

import { useEffect, useState } from 'react';
import client from '../api/client';

export function useForecast(stationId, date) {
  const [forecast, setForecast] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!stationId || !date) return;

    let cancelled = false;

    async function fetchForecast() {
      setLoading(true);
      setError(null);
      setForecast(null);

      try {
        const { data } = await client.get('/api/forecast/time-slots', {
          params: { station_id: stationId, date },
        });
        if (!cancelled) {
          setForecast(data);
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

    fetchForecast();
    return () => { cancelled = true; };
  }, [stationId, date]);

  return { forecast, loading, error };
}
