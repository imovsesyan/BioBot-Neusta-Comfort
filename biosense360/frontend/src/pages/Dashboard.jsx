/**
 * Dashboard page — F14-UC4
 *
 * Features:
 *   - KPI Cards row (avg humidex, danger %, hottest/coolest station)
 *   - Monthly heatmap calendar
 *   - Stacked bar chart (comfort hours per day)
 *   - Dual-axis line chart (temperature vs humidex, 30 days)
 *   - Human recommendation panel
 *   - Irrigation panel
 *   - Smart AI Advisor
 */

import { useEffect, useState } from 'react';
import { useStations } from '../hooks/useStations.js';
import { useRiskSummary } from '../hooks/useRiskSummary.js';
import KPICards from '../components/dashboard/KPICards.jsx';
import ThermalCalendar from '../components/calendar/ThermalCalendar.jsx';
import WeeklyBar from '../components/charts/WeeklyBar.jsx';
import DualAxis from '../components/charts/DualAxis.jsx';
import HumanRecoPanel from '../components/recommendations/HumanRecoPanel.jsx';
import IrrigationPanel from '../components/recommendations/IrrigationPanel.jsx';
import SmartAdvisor from '../components/recommendations/SmartAdvisor.jsx';
import client from '../api/client.js';

function todayISO() {
  return new Date().toISOString().slice(0, 10);
}

function thirtyDaysAgo() {
  const d = new Date();
  d.setDate(d.getDate() - 30);
  return d.toISOString().slice(0, 10);
}

export default function Dashboard() {
  const [selectedDate, setSelectedDate] = useState(todayISO());
  const [selectedStationId, setSelectedStationId] = useState(null);
  const [trends, setTrends] = useState([]);
  const [trendsLoading, setTrendsLoading] = useState(false);

  const { stations } = useStations();
  const { summary, loading: sumLoading } = useRiskSummary(selectedDate, selectedStationId);

  // Fetch 30-day trends when a station is selected
  useEffect(() => {
    if (!selectedStationId) {
      setTrends([]);
      return;
    }

    let cancelled = false;
    setTrendsLoading(true);

    client.get('/api/trends', {
      params: {
        station_id: selectedStationId,
        from: thirtyDaysAgo(),
        to: todayISO(),
      },
    })
      .then(({ data }) => {
        if (!cancelled) setTrends(data.daily || []);
      })
      .catch(() => {
        if (!cancelled) setTrends([]);
      })
      .finally(() => {
        if (!cancelled) setTrendsLoading(false);
      });

    return () => { cancelled = true; };
  }, [selectedStationId]);

  // Auto-select first station if none selected
  useEffect(() => {
    if (stations.length > 0 && !selectedStationId) {
      setSelectedStationId(stations[0].id);
    }
  }, [stations]);

  const selectedStation = stations.find((s) => s.id === selectedStationId);
  const currentHumidex = summary?.avg_humidex;
  const currentClass = summary?.risk_level;

  return (
    <div className="space-y-6">
      {/* Station selector */}
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
        <div className="text-xs text-[#94a3b8]">
          Selected date: <span className="text-[#38bdf8] font-mono">{selectedDate}</span>
        </div>
      </div>

      {/* KPI Cards */}
      {sumLoading ? (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {[1,2,3,4].map(i => (
            <div key={i} className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl h-24 animate-pulse" />
          ))}
        </div>
      ) : (
        <KPICards summary={summary} />
      )}

      {/* Calendar + Charts row */}
      <div className="grid lg:grid-cols-2 gap-6">
        {/* Heatmap Calendar */}
        <Card title="Monthly Heatmap Calendar">
          <p className="text-[#94a3b8] text-xs mb-3">
            Click a day to update the selected date across all panels.
          </p>
          <ThermalCalendar
            selectedDate={selectedDate}
            onDateSelect={setSelectedDate}
            stationId={selectedStationId}
          />
        </Card>

        {/* Stacked bar chart */}
        <Card title="Comfort Class Distribution (by Day)">
          <WeeklyBar trendsData={trends} />
        </Card>
      </div>

      {/* 30-day dual axis trend chart */}
      <Card title={`30-Day Temperature & Humidex Trend${selectedStation ? ` — ${selectedStation.name}` : ''}`}>
        {trendsLoading ? (
          <div className="h-48 flex items-center justify-center text-[#94a3b8] text-sm">
            Loading trend data...
          </div>
        ) : (
          <DualAxis trendsData={trends} />
        )}
      </Card>

      {/* Recommendation panels */}
      <div className="grid md:grid-cols-2 gap-6">
        <HumanRecoPanel currentHumidex={currentHumidex} currentClass={currentClass} />
        <IrrigationPanel date={selectedDate} stationId={selectedStationId} currentHumidex={currentHumidex} />
      </div>

      {/* AI Smart Advisor */}
      <SmartAdvisor
        date={selectedDate}
        station={selectedStation?.name}
        humidex={currentHumidex}
        comfortClass={currentClass}
      />
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
