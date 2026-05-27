/**
 * KPICards — row of 4 key performance indicator cards for the Dashboard page.
 *
 * Displays:
 *   1. Average Humidex today
 *   2. % of hours in the danger zone (Humidex >= 45)
 *   3. Hottest station (name + max humidex)
 *   4. Coolest station (name + min humidex)
 */

import { getRiskColor } from '../../constants/thermalColors.js';

function KPICard({ title, value, sub, color }) {
  return (
    <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-4 flex flex-col gap-1">
      <p className="text-[#94a3b8] text-xs font-medium uppercase tracking-wider">{title}</p>
      <p className="text-2xl font-bold" style={{ color: color || '#f1f5f9' }}>
        {value}
      </p>
      {sub && <p className="text-[#94a3b8] text-xs">{sub}</p>}
    </div>
  );
}

export default function KPICards({ summary }) {
  if (!summary) {
    return (
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl h-20 animate-pulse" />
        ))}
      </div>
    );
  }

  const breakdown = summary.stations_breakdown || [];
  const avgHumidex = summary.avg_humidex ?? 0;

  // % of hours in danger zone — use dangerous_hours from each station
  const totalDangerSlots = breakdown.reduce(
    (acc, s) => acc + (s.dangerous_hours?.length ?? 0), 0
  );
  const totalSlots = breakdown.length * 24;
  const dangerPct = totalSlots > 0 ? Math.round((totalDangerSlots / totalSlots) * 100) : 0;

  // Hottest / coolest station
  const hottest = breakdown.reduce(
    (best, s) => (!best || s.avg_humidex > best.avg_humidex ? s : best),
    null
  );
  const coolest = breakdown.reduce(
    (best, s) => (!best || s.avg_humidex < best.avg_humidex ? s : best),
    null
  );

  const riskColor = getRiskColor(summary.risk_level);

  return (
    <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
      <KPICard
        title="Avg Humidex Today"
        value={avgHumidex.toFixed(1)}
        sub={`Risk: ${summary.risk_level}`}
        color={riskColor}
      />
      <KPICard
        title="Hours in Danger Zone"
        value={`${dangerPct}%`}
        sub={`${totalDangerSlots} of ${totalSlots} station-hours`}
        color={dangerPct > 30 ? '#ef4444' : dangerPct > 10 ? '#f97316' : '#22c55e'}
      />
      <KPICard
        title="Hottest Station"
        value={hottest ? `${hottest.avg_humidex?.toFixed(1) ?? '—'}` : '—'}
        sub={hottest ? `Station #${hottest.station_id}` : 'No data'}
        color="#ef4444"
      />
      <KPICard
        title="Coolest Station"
        value={coolest ? `${coolest.avg_humidex?.toFixed(1) ?? '—'}` : '—'}
        sub={coolest ? `Station #${coolest.station_id}` : 'No data'}
        color="#22c55e"
      />
    </div>
  );
}
