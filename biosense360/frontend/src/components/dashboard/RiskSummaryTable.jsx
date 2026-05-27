/**
 * RiskSummaryTable — tabular risk breakdown for multiple stations.
 *
 * Includes a clipboard copy button that serialises the table to CSV.
 */

import { useState } from 'react';
import DangerBadge from '../risk/DangerBadge.jsx';

export default function RiskSummaryTable({ summary }) {
  const [copied, setCopied] = useState(false);

  if (!summary || !summary.stations_breakdown?.length) {
    return (
      <p className="text-[#94a3b8] text-sm">No station breakdown available.</p>
    );
  }

  function copyToClipboard() {
    const header = 'Station ID,Risk Level,Avg Humidex,Avg Temp,Favorable Hours,Dangerous Hours';
    const rows = summary.stations_breakdown.map((s) =>
      [
        s.station_id,
        s.risk_level,
        s.avg_humidex?.toFixed(1) ?? '',
        s.avg_temp?.toFixed(1) ?? '',
        (s.favorable_hours || []).join('; '),
        (s.dangerous_hours || []).join('; '),
      ].join(',')
    );
    navigator.clipboard.writeText([header, ...rows].join('\n')).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    });
  }

  return (
    <div className="space-y-2">
      <div className="flex justify-end">
        <button
          onClick={copyToClipboard}
          className="text-xs text-[#38bdf8] hover:text-[#f1f5f9] transition-colors px-3 py-1 border border-[#2d3548] rounded-md"
        >
          {copied ? 'Copied!' : 'Copy CSV'}
        </button>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="border-b border-[#2d3548]">
              <th className="text-left text-[#94a3b8] font-medium p-2">Station</th>
              <th className="text-left text-[#94a3b8] font-medium p-2">Risk</th>
              <th className="text-right text-[#94a3b8] font-medium p-2">Avg Humidex</th>
              <th className="text-right text-[#94a3b8] font-medium p-2">Avg Temp</th>
              <th className="text-left text-[#94a3b8] font-medium p-2">Favorable</th>
              <th className="text-left text-[#94a3b8] font-medium p-2">Dangerous</th>
            </tr>
          </thead>
          <tbody>
            {summary.stations_breakdown.map((s) => (
              <tr key={s.station_id} className="border-b border-[#2d3548] hover:bg-[#1a1f2e]">
                <td className="p-2 text-[#f1f5f9] text-xs">ID {s.station_id}</td>
                <td className="p-2">
                  <DangerBadge level={s.risk_level} size="sm" />
                </td>
                <td className="p-2 text-right text-[#f1f5f9] font-mono">
                  {s.avg_humidex?.toFixed(1) ?? '—'}
                </td>
                <td className="p-2 text-right text-[#f1f5f9] font-mono">
                  {s.avg_temp?.toFixed(1) ?? '—'}°C
                </td>
                <td className="p-2 text-[#22c55e] text-xs">
                  {(s.favorable_hours || []).slice(0, 3).join(', ')}
                  {(s.favorable_hours || []).length > 3 && '...'}
                </td>
                <td className="p-2 text-[#ef4444] text-xs">
                  {(s.dangerous_hours || []).slice(0, 3).join(', ')}
                  {(s.dangerous_hours || []).length > 3 && '...'}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
