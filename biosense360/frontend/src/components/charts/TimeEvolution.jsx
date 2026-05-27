/**
 * TimeEvolution — line chart showing temperature and humidex over 24 hours.
 * Used on the DailyAnalysis page.
 */

import {
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';

export default function TimeEvolution({ slots }) {
  if (!slots || slots.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-[#94a3b8] text-sm">
        No data to display
      </div>
    );
  }

  const data = slots.map((s) => ({
    hour: s.label,
    Temperature: parseFloat(s.temperature.toFixed(1)),
    Humidex: parseFloat(s.humidex.toFixed(1)),
    Humidity: parseFloat(s.humidity.toFixed(1)),
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" />
        <XAxis
          dataKey="hour"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          tickLine={false}
          interval={3}
        />
        <YAxis
          yAxisId="temp"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          tickLine={false}
          unit="°C"
          width={40}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#1a1f2e', border: '1px solid #2d3548', borderRadius: 8 }}
          labelStyle={{ color: '#38bdf8' }}
          itemStyle={{ color: '#f1f5f9' }}
        />
        <Legend wrapperStyle={{ color: '#94a3b8', fontSize: 12 }} />
        <Line
          yAxisId="temp"
          type="monotone"
          dataKey="Temperature"
          stroke="#f97316"
          strokeWidth={2}
          dot={false}
        />
        <Line
          yAxisId="temp"
          type="monotone"
          dataKey="Humidex"
          stroke="#ef4444"
          strokeWidth={2}
          dot={false}
          strokeDasharray="5 3"
        />
        <Line
          yAxisId="temp"
          type="monotone"
          dataKey="Humidity"
          stroke="#38bdf8"
          strokeWidth={1.5}
          dot={false}
          strokeDasharray="2 4"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
