/**
 * RiskMatrix — displays departure × arrival station risk grid.
 *
 * Used on the MapMobility page to visualise inter-station mobility risk.
 */

import { useEffect, useState } from 'react';
import client from '../../api/client.js';
import { getRiskColor } from '../../constants/thermalColors.js';

export default function RiskMatrix({ date }) {
  const [matrix, setMatrix] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (!date) return;
    let cancelled = false;

    async function fetchMatrix() {
      setLoading(true);
      setError(null);
      try {
        const { data } = await client.get('/api/risk-matrix', { params: { date } });
        if (!cancelled) setMatrix(data.matrix || []);
      } catch (err) {
        if (!cancelled) setError(err.message);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }

    fetchMatrix();
    return () => { cancelled = true; };
  }, [date]);

  if (loading) {
    return <p className="text-[#94a3b8] text-sm">Loading risk matrix...</p>;
  }
  if (error) {
    return <p className="text-[#ef4444] text-sm">Error: {error}</p>;
  }
  if (matrix.length === 0) {
    return <p className="text-[#94a3b8] text-sm">No matrix data for {date}.</p>;
  }

  // Get unique departure and arrival names for building the table headers
  const departures = [...new Set(matrix.map((c) => c.departure))];
  const arrivals   = [...new Set(matrix.map((c) => c.arrival))];

  // Build a fast lookup: "departure||arrival" → cell
  const cellMap = {};
  matrix.forEach((c) => {
    cellMap[`${c.departure}||${c.arrival}`] = c;
  });

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-xs border-collapse">
        <thead>
          <tr>
            <th className="text-left text-[#94a3b8] p-2 border border-[#2d3548] bg-[#1a1f2e]">
              From / To
            </th>
            {arrivals.map((arr) => (
              <th
                key={arr}
                className="text-center text-[#94a3b8] p-2 border border-[#2d3548] bg-[#1a1f2e] max-w-[100px]"
              >
                <span className="block truncate" title={arr}>{arr.split(' ')[0]}</span>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {departures.map((dep) => (
            <tr key={dep}>
              <td className="text-[#94a3b8] p-2 border border-[#2d3548] bg-[#1a1f2e] max-w-[120px]">
                <span className="block truncate" title={dep}>{dep.split(' ')[0]}</span>
              </td>
              {arrivals.map((arr) => {
                const cell = cellMap[`${dep}||${arr}`];
                if (!cell) {
                  return (
                    <td key={arr} className="p-2 border border-[#2d3548] text-center text-[#2d3548]">
                      —
                    </td>
                  );
                }
                const color = getRiskColor(cell.risk_level);
                return (
                  <td
                    key={arr}
                    className="p-2 border border-[#2d3548] text-center font-semibold"
                    style={{ backgroundColor: color + '30', color }}
                    title={cell.recommendation}
                  >
                    {cell.risk_level}
                    <div className="text-[10px] font-normal text-[#94a3b8]">
                      {cell.avg_humidex.toFixed(0)}
                    </div>
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
