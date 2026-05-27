/**
 * TimeSlotBar — 24-hour horizontal bar showing comfort class per hour.
 *
 * Each of the 24 segments is colored by the slot's comfort class.
 * Hovering a segment shows a tooltip with temperature, humidity, and humidex.
 */

import { useState } from 'react';
import { getComfortColor } from '../../constants/thermalColors.js';

export default function TimeSlotBar({ slots }) {
  const [tooltip, setTooltip] = useState(null);

  if (!slots || slots.length === 0) {
    return (
      <div className="h-10 bg-[#1a1f2e] rounded-lg flex items-center justify-center">
        <span className="text-[#94a3b8] text-sm">No hourly data available</span>
      </div>
    );
  }

  // Build a map from hour → slot for fast lookup (handle gaps with defaults)
  const slotByHour = {};
  slots.forEach((s) => { slotByHour[s.hour] = s; });

  return (
    <div className="relative">
      {/* Hour bar */}
      <div className="flex h-10 rounded-lg overflow-hidden gap-px">
        {Array.from({ length: 24 }, (_, h) => {
          const slot = slotByHour[h];
          const color = slot ? getComfortColor(slot.comfort_class) : '#2d3548';

          return (
            <div
              key={h}
              className="flex-1 cursor-pointer transition-opacity hover:opacity-80"
              style={{ backgroundColor: color }}
              onMouseEnter={(e) => {
                if (slot) {
                  const rect = e.currentTarget.getBoundingClientRect();
                  setTooltip({ slot, x: rect.left + rect.width / 2, y: rect.top });
                }
              }}
              onMouseLeave={() => setTooltip(null)}
            />
          );
        })}
      </div>

      {/* Hour labels */}
      <div className="flex justify-between mt-1 text-[10px] text-[#94a3b8]">
        {[0, 3, 6, 9, 12, 15, 18, 21, 23].map((h) => (
          <span key={h}>{String(h).padStart(2, '0')}h</span>
        ))}
      </div>

      {/* Tooltip */}
      {tooltip && (
        <div
          className="fixed z-50 bg-[#0f1117] border border-[#2d3548] rounded-lg p-2 text-xs text-[#f1f5f9] shadow-lg pointer-events-none"
          style={{ left: tooltip.x, top: tooltip.y - 80, transform: 'translateX(-50%)' }}
        >
          <div className="font-semibold text-[#38bdf8]">{tooltip.slot.label}</div>
          <div>{tooltip.slot.comfort_class}</div>
          <div>Humidex: {tooltip.slot.humidex.toFixed(1)}</div>
          <div>{tooltip.slot.temperature.toFixed(1)}°C / {tooltip.slot.humidity.toFixed(0)}%</div>
        </div>
      )}
    </div>
  );
}
