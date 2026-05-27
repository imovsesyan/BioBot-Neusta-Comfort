/**
 * DualAxis — 30-day line chart with temperature on the left Y axis
 * and humidex on the right Y axis.
 *
 * Consumes the /api/trends response.
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

export default function DualAxis({ trendsData }) {
  if (!trendsData || trendsData.length === 0) {
    return (
      <div className="h-48 flex items-center justify-center text-[#94a3b8] text-sm">
        No trend data available
      </div>
    );
  }

  const data = trendsData.map((t) => ({
    date: t.date ? t.date.slice(5) : '', // MM-DD label
    Temperature: t.avg_temp != null ? parseFloat(t.avg_temp.toFixed(1)) : null,
    Humidex:     t.avg_humidex != null ? parseFloat(t.avg_humidex.toFixed(1)) : null,
  }));

  return (
    <ResponsiveContainer width="100%" height={220}>
      <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#2d3548" />
        <XAxis
          dataKey="date"
          tick={{ fill: '#94a3b8', fontSize: 10 }}
          tickLine={false}
          interval="preserveStartEnd"
        />
        {/* Left Y axis — Temperature */}
        <YAxis
          yAxisId="temp"
          tick={{ fill: '#f97316', fontSize: 11 }}
          tickLine={false}
          unit="°C"
          width={38}
        />
        {/* Right Y axis — Humidex */}
        <YAxis
          yAxisId="humidex"
          orientation="right"
          tick={{ fill: '#ef4444', fontSize: 11 }}
          tickLine={false}
          width={38}
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
          yAxisId="humidex"
          type="monotone"
          dataKey="Humidex"
          stroke="#ef4444"
          strokeWidth={2}
          dot={false}
          strokeDasharray="5 3"
        />
      </LineChart>
    </ResponsiveContainer>
  );
}
