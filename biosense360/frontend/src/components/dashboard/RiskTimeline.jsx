/**
 * RiskTimeline — visual timeline of risk periods across the day.
 *
 * Renders a horizontal timeline with color-coded blocks for
 * favorable, neutral, and dangerous hours.
 */

import { getRiskColor } from '../../constants/thermalColors.js';

export default function RiskTimeline({ forecast }) {
  if (!forecast || !forecast.slots?.length) {
    return (
      <p className="text-[#94a3b8] text-sm">No timeline data available.</p>
    );
  }

  return (
    <div className="space-y-3">
      {/* Timeline track */}
      <div className="relative">
        <div className="flex h-6 rounded overflow-hidden gap-px">
          {forecast.slots.map((slot) => {
            const color = getRiskColor(slot.risk_level);
            return (
              <div
                key={slot.hour}
                className="flex-1"
                style={{ backgroundColor: color }}
                title={`${slot.label}: Humidex ${slot.humidex.toFixed(1)} (${slot.risk_level})`}
              />
            );
          })}
        </div>

        {/* Labels at key hours */}
        <div className="flex justify-between mt-1">
          {[0, 6, 12, 18, 23].map((h) => (
            <span key={h} className="text-[10px] text-[#94a3b8]">
              {String(h).padStart(2, '0')}h
            </span>
          ))}
        </div>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-3 text-xs">
        {['LOW', 'MODERATE', 'HIGH', 'CRITICAL'].map((level) => (
          <div key={level} className="flex items-center gap-1">
            <div
              className="w-3 h-3 rounded-sm"
              style={{ backgroundColor: getRiskColor(level) }}
            />
            <span className="text-[#94a3b8]">{level}</span>
          </div>
        ))}
      </div>

      {/* Slots detail */}
      <div className="grid grid-cols-6 sm:grid-cols-8 md:grid-cols-12 gap-1">
        {forecast.slots.map((slot) => (
          <div
            key={slot.hour}
            className="text-center p-1 rounded text-xs"
            style={{ backgroundColor: getRiskColor(slot.risk_level) + '22' }}
          >
            <div className="font-mono text-[#94a3b8]">{slot.label}</div>
            <div className="font-semibold" style={{ color: getRiskColor(slot.risk_level) }}>
              {slot.humidex.toFixed(0)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
