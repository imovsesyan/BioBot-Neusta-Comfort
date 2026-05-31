/**
 * Dashboard page — thermal comfort overview.
 *
 * Features:
 *   - Station selector + date picker
 *   - KPI Cards (avg humidex, risk level, temp)
 *   - Monthly heatmap calendar (click a day to update the date)
 */

import { useEffect, useState } from 'react';
import { useStations } from '../hooks/useStations.js';
import { useRiskSummary } from '../hooks/useRiskSummary.js';
import KPICards from '../components/dashboard/KPICards.jsx';
import ThermalCalendar from '../components/calendar/ThermalCalendar.jsx';

/**
 * Default date: August 12, 2025 — peak heat day in the real Météo France 2025 dataset.
 * The DB is seeded with real measurements for all of 2025.
 */
const DEFAULT_DATE = "2025-08-12";

export default function Dashboard() {
  const [selectedDate, setSelectedDate] = useState(DEFAULT_DATE);
  const [selectedStationId, setSelectedStationId] = useState(null);

  const { stations } = useStations();
  const { summary, loading: sumLoading } = useRiskSummary(selectedDate, selectedStationId);
  // All-stations summary — drives the cross-station Hottest/Coolest/Danger KPIs
  // (the single-station selector above only controls the average + calendar).
  const { summary: allSummary } = useRiskSummary(selectedDate, null);

  // Auto-select first station on load
  useEffect(() => {
    if (stations.length > 0 && !selectedStationId) {
      setSelectedStationId(stations[0].id);
    }
  }, [stations]);

  return (
    <div className="space-y-6">

      {/* Controls: station + date */}
      <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-4 flex flex-wrap gap-4 items-end">
        <div>
          <label className="text-[#94a3b8] text-xs block mb-1">Station</label>
          <select
            value={selectedStationId ?? ''}
            onChange={(e) => setSelectedStationId(parseInt(e.target.value) || null)}
            className="bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8] min-w-[220px]"
          >
            {stations.map((s) => (
              <option key={s.id} value={s.id}>{s.name}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-[#94a3b8] text-xs block mb-1">Date</label>
          <input
            type="date"
            value={selectedDate}
            min="2025-01-01"
            max="2025-12-31"
            onChange={(e) => e.target.value && setSelectedDate(e.target.value)}
            className="bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8]"
          />
        </div>

        <div className="text-xs text-[#64748b]">
          Data: January – December 2025 · Météo France
        </div>
      </div>

      {/* KPI Cards */}
      {sumLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl h-24 animate-pulse" />
          ))}
        </div>
      ) : (
        <KPICards
          summary={summary}
          allBreakdown={allSummary?.stations_breakdown}
          stations={stations}
        />
      )}

      {/* Monthly Heatmap Calendar */}
      <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-5 space-y-3">
        <div className="flex items-center justify-between">
          <h2 className="text-[#f1f5f9] font-semibold text-sm">Monthly Heatmap Calendar</h2>
          <span className="text-[#64748b] text-xs">Click any day to update the date</span>
        </div>
        <ThermalCalendar
          selectedDate={selectedDate}
          onDateSelect={setSelectedDate}
          stationId={selectedStationId}
        />
      </div>

    </div>
  );
}
