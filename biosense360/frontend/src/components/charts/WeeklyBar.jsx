/**
 * WeeklyBar — stacked bar chart showing hours per comfort class per day.
 *
 * X axis = date labels, Y axis = hours (0–24), stacked by comfort class.
 * Data is derived from the 30-day trends endpoint.
 */

import {
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import { COMFORT_CLASS_ORDER, THERMAL_COLORS } from '../../constants/thermalColors.js';

export default function WeeklyBar({ trendsData, forecastSlots }) {
  // Build weekly data from either trends or forecast slots
  // If trendsData is provided (array of daily aggregates), use that.
  // If only forecastSlots (24 hourly), show comfort class counts for that day.

  let data = [];

  if (forecastSlots && forecastSlots.length > 0) {
    // Group by comfort class count for a single day
    const counts = {};
    COMFORT_CLASS_ORDER.forEach((cls) => { counts[cls] = 0; });
    forecastSlots.forEach((s) => {
      if (counts[s.comfort_class] !== undefined) {
        counts[s.comfort_class]++;
      }
    });
    data = [{ day: 'Today', ...counts }];
  } else if (trendsData && trendsData.length > 0) {
    // Map each daily trend to a bar entry — we don't have hourly breakdown per class
    // from the trends endpoint, so we approximate via risk_level color buckets.
    data = trendsData.map((t) => {
      const label = t.date ? t.date.slice(5) : ''; // MM-DD
      // Estimate: assign all 24 hours to one bucket based on risk_level
      const entry = { day: label, Comfortable: 0, Caution: 0, 'Extreme Caution': 0, Danger: 0, 'Extreme Danger': 0 };
      const mapping = { LOW: 'Comfortable', MODERATE: 'Caution', HIGH: 'Extreme Caution', CRITICAL: 'Danger' };
      const cls = mapping[t.risk_level] ?? 'Caution';
      entry[cls] = 24;
      return entry;
    });
  }

  if (data.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-[#94a3b8] text-sm">
        No data to display
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} margin={{ top: 5, right: 10, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" vertical={false} />
        <XAxis
          dataKey="day"
          tick={{ fill: '#94a3b8', fontSize: 10 }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        <YAxis
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          tickLine={false}
          unit="h"
          width={32}
          domain={[0, 24]}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #2d3548', borderRadius: 8 }}
          labelStyle={{ color: '#38bdf8' }}
          itemStyle={{ color: '#f1f5f9' }}
        />
        <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 11 }} />
        {COMFORT_CLASS_ORDER.map((cls) => (
          <Bar key={cls} dataKey={cls} stackId="a" fill={THERMAL_COLORS[cls]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  );
}
