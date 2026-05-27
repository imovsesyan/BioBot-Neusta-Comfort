/**
 * useStations — fetches and caches the list of all active stations.
 * Stations rarely change, so the list is fetched once on mount.
 */

import { useEffect, useState } from 'react';
import client from '../api/client';

export function useStations() {
  const [stations, setStations] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    let cancelled = false;

    async function fetchStations() {
      setLoading(true);
      setError(null);
      try {
        const { data } = await client.get('/api/stations');
        if (!cancelled) {
          setStations(data);
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

    fetchStations();
    return () => { cancelled = true; };
  }, []);

  return { stations, loading, error };
}
