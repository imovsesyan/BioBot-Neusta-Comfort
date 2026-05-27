/**
 * ThermalCalendar — monthly heatmap calendar.
 *
 * Fetches risk summaries for each day of the displayed month and renders
 * a grid of HeatmapCells colored by average daily Humidex.
 *
 * Clicking a day fires onDateSelect(dateString) to sync the DailyAnalysis page.
 */

import { useEffect, useState } from 'react';
import client from '../../api/client.js';
import HeatmapCell from './HeatmapCell.jsx';

const WEEKDAYS = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];

function getDaysInMonth(year, month) {
  return new Date(year, month + 1, 0).getDate();
}

function getFirstDayOfWeek(year, month) {
  return new Date(year, month, 1).getDay();
}

export default function ThermalCalendar({ onDateSelect, selectedDate, stationId }) {
  // Initialise the calendar on the month of the selectedDate (or August 2025 by default).
  const initDate = selectedDate
    ? new Date(selectedDate + 'T12:00:00')
    : new Date('2025-08-12T12:00:00');
  const [year, setYear] = useState(initDate.getFullYear());
  const [month, setMonth] = useState(initDate.getMonth());
  const [dayData, setDayData] = useState({});
  const [loading, setLoading] = useState(false);

  const monthName = new Date(year, month, 1).toLocaleString('en-US', { month: 'long' });
  const daysInMonth = getDaysInMonth(year, month);
  const firstDow = getFirstDayOfWeek(year, month);

  // Fetch risk summaries for the whole month in one call per day that has data.
  // We batch-fetch by iterating days — the backend is fast (reads from risk_periods).
  useEffect(() => {
    let cancelled = false;

    async function fetchMonth() {
      setLoading(true);
      const results = {};

      // Fetch all 30/31 days in parallel (bounded by daysInMonth)
      const promises = Array.from({ length: daysInMonth }, (_, i) => {
        const d = i + 1;
        const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
        const params = { date: dateStr };
        if (stationId) params.station_id = stationId;

        return client.get('/api/risk-summary', { params })
          .then(({ data }) => { results[dateStr] = data; })
          .catch(() => {}); // silently skip days with no data
      });

      await Promise.all(promises);
      if (!cancelled) {
        setDayData(results);
        setLoading(false);
      }
    }

    fetchMonth();
    return () => { cancelled = true; };
  }, [year, month, stationId, daysInMonth]);

  function prevMonth() {
    if (month === 0) { setYear(y => y - 1); setMonth(11); }
    else setMonth(m => m - 1);
  }

  function nextMonth() {
    if (month === 11) { setYear(y => y + 1); setMonth(0); }
    else setMonth(m => m + 1);
  }

  return (
    <div className="space-y-3">
      {/* Month navigation */}
      <div className="flex items-center justify-between">
        <button
          onClick={prevMonth}
          className="text-[#94a3b8] hover:text-[#38bdf8] transition-colors px-2"
        >
          &larr;
        </button>
        <div className="text-[#f1f5f9] font-semibold">
          {monthName} {year}
          {loading && <span className="ml-2 text-[#94a3b8] text-xs">loading...</span>}
        </div>
        <button
          onClick={nextMonth}
          className="text-[#94a3b8] hover:text-[#38bdf8] transition-colors px-2"
        >
          &rarr;
        </button>
      </div>

      {/* Weekday headers */}
      <div className="grid grid-cols-7 gap-1">
        {WEEKDAYS.map((wd) => (
          <div key={wd} className="text-center text-[10px] text-[#94a3b8] font-medium py-1">
            {wd}
          </div>
        ))}

        {/* Empty cells before the 1st */}
        {Array.from({ length: firstDow }, (_, i) => (
          <div key={`empty-${i}`} />
        ))}

        {/* Day cells */}
        {Array.from({ length: daysInMonth }, (_, i) => {
          const d = i + 1;
          const dateStr = `${year}-${String(month + 1).padStart(2, '0')}-${String(d).padStart(2, '0')}`;
          const data = dayData[dateStr];

          return (
            <HeatmapCell
              key={dateStr}
              date={dateStr}
              avgHumidex={data?.avg_humidex ?? null}
              riskLevel={data?.risk_level ?? null}
              isSelected={selectedDate === dateStr}
              onSelect={onDateSelect}
            />
          );
        })}
      </div>
    </div>
  );
}
