/**
 * MapMobility page — F14-UC2, F14-UC3
 *
 * Features:
 *   - react-leaflet map centered on Toulouse with colored station markers
 *   - Date picker + outdoor/indoor filter
 *   - Human Safety Recommendations
 *   - Irrigation Recommendations
 *   - Mobility Risk Matrix
 */

import { useEffect, useState } from 'react';
import { useStations } from '../hooks/useStations.js';
import { useRiskSummary } from '../hooks/useRiskSummary.js';
import ThermalMap from '../components/map/ThermalMap.jsx';
import RiskMatrix from '../components/risk/RiskMatrix.jsx';
import HumanRecoPanel from '../components/recommendations/HumanRecoPanel.jsx';
import IrrigationPanel from '../components/recommendations/IrrigationPanel.jsx';
import DangerBadge from '../components/risk/DangerBadge.jsx';

// Default to August 12, 2025 — peak heat day in the real Météo France dataset
const DEFAULT_DATE = "2025-08-12";

export default function MapMobility() {
  const [date, setDate] = useState(DEFAULT_DATE);
  const [typeFilter, setTypeFilter] = useState('all');
  const [selectedStationId, setSelectedStationId] = useState(null);

  const { stations } = useStations();
  const { summary } = useRiskSummary(date, null);

  // Build a map from station_id → risk data for marker coloring
  const riskByStation = {};
  if (summary?.stations_breakdown) {
    summary.stations_breakdown.forEach((s) => {
      riskByStation[s.station_id] = s;
    });
  }

  const filteredStations = stations.filter((s) =>
    typeFilter === 'all' ? true : s.type === typeFilter
  );

  // Auto-select first station
  useEffect(() => {
    if (stations.length > 0 && !selectedStationId) {
      setSelectedStationId(stations[0].id);
    }
  }, [stations]);

  const selectedRisk = selectedStationId ? riskByStation[selectedStationId] : null;
  const currentHumidex = selectedRisk?.avg_humidex ?? summary?.avg_humidex;
  const currentClass = selectedRisk?.risk_level ?? summary?.risk_level;

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
              min="2025-01-01"
              max="2025-12-31"
              onChange={(e) => e.target.value && setDate(e.target.value)}
              className="bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8]"
            />
          </div>

          <div>
            <label className="text-[#94a3b8] text-xs block mb-1">Station</label>
            <select
              value={selectedStationId ?? ''}
              onChange={(e) => setSelectedStationId(parseInt(e.target.value) || null)}
              className="bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8] min-w-[220px]"
            >
              {stations.map((s) => (
                <option key={s.id} value={s.id}>{s.name} ({s.type})</option>
              ))}
            </select>
          </div>

          <div className="flex gap-2">
            {['all', 'outdoor', 'indoor'].map((t) => (
              <button
                key={t}
                onClick={() => setTypeFilter(t)}
                className={[
                  'px-3 py-2 text-sm rounded-lg border transition-colors capitalize',
                  typeFilter === t
                    ? 'bg-[#38bdf8] text-[#0f1117] border-[#38bdf8] font-semibold'
                    : 'border-[#2d3548] text-[#94a3b8] hover:text-[#f1f5f9]',
                ].join(' ')}
              >
                {t}
              </button>
            ))}
          </div>

          {summary && (
            <div className="flex items-center gap-2">
              <span className="text-[#94a3b8] text-xs">Overall Risk:</span>
              <DangerBadge level={summary.risk_level} size="md" />
            </div>
          )}
        </div>
      </div>

      {/* Map */}
      <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-4">
        <ThermalMap stations={filteredStations} riskByStation={riskByStation} />
      </div>

      {/* Recommendation panels */}
      <div className="grid md:grid-cols-2 gap-6">
        <HumanRecoPanel currentHumidex={currentHumidex} currentClass={currentClass} />
        <IrrigationPanel date={date} stationId={selectedStationId} currentHumidex={currentHumidex ?? 30} />
      </div>

      {/* Risk Matrix */}
      <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-5 space-y-3">
        <h2 className="text-[#f1f5f9] font-semibold text-sm">Mobility Risk Matrix</h2>
        <p className="text-[#94a3b8] text-xs">
          Combined thermal risk for each departure → arrival station pair on {date}.
        </p>
        <RiskMatrix date={date} />
      </div>

    </div>
  );
}
