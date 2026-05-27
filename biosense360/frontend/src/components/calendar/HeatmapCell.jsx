/**
 * HeatmapCell — a single day cell in the ThermalCalendar heatmap.
 *
 * Colored by the average humidex for that day.
 * Clicking the cell propagates the date upward via onSelect.
 */

import { getRiskColor } from '../../constants/thermalColors.js';

export default function HeatmapCell({ date, avgHumidex, riskLevel, isSelected, onSelect }) {
  const color = getRiskColor(riskLevel ?? 'LOW');
  const day = new Date(date + 'T00:00:00').getDate();

  return (
    <button
      onClick={() => onSelect(date)}
      className={[
        'w-full aspect-square rounded-md flex flex-col items-center justify-center',
        'transition-all duration-100 focus:outline-none',
        isSelected ? 'ring-2 ring-[#38bdf8] ring-offset-1 ring-offset-[#0f1117]' : '',
      ].join(' ')}
      style={{
        backgroundColor: avgHumidex != null ? color + '44' : '#2d3548',
        borderColor: color,
        borderWidth: 1,
        borderStyle: 'solid',
      }}
      title={avgHumidex != null ? `${date}: Humidex ${avgHumidex.toFixed(1)} (${riskLevel})` : date}
    >
      <span className="text-xs font-semibold" style={{ color: avgHumidex != null ? color : '#94a3b8' }}>
        {day}
      </span>
      {avgHumidex != null && (
        <span className="text-[9px]" style={{ color }}>
          {avgHumidex.toFixed(0)}
        </span>
      )}
    </button>
  );
}
