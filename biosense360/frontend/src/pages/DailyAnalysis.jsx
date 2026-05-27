/**
 * DailyAnalysis page — F14-UC1, F14-UC5
 *
 * Features:
 *   - Date picker + station selector
 *   - 24-hour TimeSlotBar colored by comfort class
 *   - Favorable windows badge
 *   - Day-level DangerBadge
 *   - Risk Summary Table with clipboard copy
 *   - Risk Period Timeline
 *   - 24h temperature/humidex evolution chart
 */

import { useState } from 'react';
import { useStations } from '../hooks/useStations.js';
import { useForecast } from '../hooks/useForecast.js';
import { useRiskSummary } from '../hooks/useRiskSummary.js';
import DangerBadge from '../components/risk/DangerBadge.jsx';
import TimeSlotBar from '../components/risk/TimeSlotBar.jsx';
import RiskTimeline from '../components/dashboard/RiskTimeline.jsx';
import RiskSummaryTable from '../components/dashboard/RiskSummaryTable.jsx';
import TimeEvolution from '../components/charts/TimeEvolution.jsx';

// Default: August 12, 2025 — peak heat day in the real Météo France dataset
const DEFAULT_DATE = "2025-08-12";

export default function DailyAnalysis() {
  const [date, setDate] = useState(DEFAULT_DATE);
  const [stationId, setStationId] = useState('');

  const { stations, loading: stLoading } = useStations();
  const { forecast, loading: fcLoading } = useForecast(stationId || null, date);
  const { summary, loading: sumLoading } = useRiskSummary(date, stationId || null);

  const selectedStation = stations.find((s) => s.id === parseInt(stationId)) ?? null;
  const loading = stLoading || fcLoading || sumLoading;

  return (
    <div className="space-y-6">
      {/* Controls */}
      <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-4">
        <div className="flex flex-wrap gap-4 items-end">
          <div>
            <label className="text-[#94a3b8] text-xs block mb-1">Date</label>
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8]"
            />
          </div>
          <div>
            <label className="text-[#94a3b8] text-xs block mb-1">Station</label>
            <select
              value={stationId}
              onChange={(e) => setStationId(e.target.value)}
              className="bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8] min-w-[200px]"
            >
              <option value="">All Stations</option>
              {stations.map((s) => (
                <option key={s.id} value={s.id}>
                  {s.name} ({s.type})
                </option>
              ))}
            </select>
          </div>

          {summary && (
            <div className="flex items-center gap-2">
              <span className="text-[#94a3b8] text-xs">Day Risk:</span>
              <DangerBadge level={summary.risk_level} size="lg" />
            </div>
          )}
        </div>
      </div>

      {loading && (
        <div className="text-[#94a3b8] text-sm text-center py-6">Loading data...</div>
      )}

      {/* 24h TimeSlot Bar */}
      {forecast && (
        <Card title="24-Hour Comfort Timeline">
          <TimeSlotBar slots={forecast.slots} />

          {/* Favorable windows */}
          <div className="mt-3 flex flex-wrap gap-2 items-center">
            <span className="text-[#94a3b8] text-xs">Favorable windows:</span>
            {forecast.favorable_hours.length > 0 ? (
              forecast.favorable_hours.map((h) => (
                <span key={h} className="text-xs bg-[#22c55e22] text-[#22c55e] px-2 py-0.5 rounded-full border border-[#22c55e44]">
                  {h}
                </span>
              ))
            ) : (
              <span className="text-[#94a3b8] text-xs">None today</span>
            )}

            {forecast.dangerous_hours.length > 0 && (
              <>
                <span className="text-[#94a3b8] text-xs ml-4">Dangerous hours:</span>
                {forecast.dangerous_hours.slice(0, 6).map((h) => (
                  <span key={h} className="text-xs bg-[#ef444422] text-[#ef4444] px-2 py-0.5 rounded-full border border-[#ef444444]">
                    {h}
                  </span>
                ))}
              </>
            )}
          </div>
        </Card>
      )}

      {/* Temperature / Humidex evolution */}
      {forecast && (
        <Card title="Temperature & Humidex Evolution (24h)">
          <TimeEvolution slots={forecast.slots} />
        </Card>
      )}

      {/* Risk Period Timeline */}
      {forecast && (
        <Card title="Risk Period Timeline">
          <RiskTimeline forecast={forecast} />
        </Card>
      )}

      {/* Risk Summary Table */}
      {summary && (
        <Card title={`Risk Summary — ${date}`}>
          <div className="grid grid-cols-3 gap-4 mb-4">
            <MetricPill label="Avg Humidex" value={summary.avg_humidex?.toFixed(1)} unit="" />
            <MetricPill label="Avg Temp" value={summary.avg_temp?.toFixed(1)} unit="°C" />
            <MetricPill label="Risk Level" value={summary.risk_level} />
          </div>
          <RiskSummaryTable summary={summary} />
        </Card>
      )}

      {!loading && !forecast && !summary && (
        <div className="text-center py-12 text-[#94a3b8]">
          <p className="text-lg font-medium">No data for {date}</p>
          <p className="text-sm mt-1">Select a date with seeded data (2024-07-01 to 2024-07-30)</p>
        </div>
      )}
    </div>
  );
}

function Card({ title, children }) {
  return (
    <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-5 space-y-3">
      <h2 className="text-[#f1f5f9] font-semibold text-sm">{title}</h2>
      {children}
    </div>
  );
}

function MetricPill({ label, value, unit }) {
  return (
    <div className="bg-[#0f1117] rounded-lg p-3 border border-[#2d3548] text-center">
      <p className="text-[#94a3b8] text-xs">{label}</p>
      <p className="text-[#38bdf8] font-bold text-lg mt-0.5">
        {value ?? '—'}{unit && value ? unit : ''}
      </p>
    </div>
  );
}
