/**
 * StationMarker — Leaflet circle marker for a monitoring station.
 *
 * Color reflects the station's current comfort class.
 * Clicking opens a popup with full sensor data.
 */

import L from 'leaflet';
import { CircleMarker, Popup } from 'react-leaflet';
import { getComfortColor } from '../../constants/thermalColors.js';
import DangerBadge from '../risk/DangerBadge.jsx';

export default function StationMarker({ station, riskData }) {
  const comfortClass = riskData?.comfort_class ?? null;
  const color = comfortClass ? getComfortColor(comfortClass) : '#94a3b8';

  const humidex   = riskData?.avg_humidex ?? null;
  const temp      = riskData?.avg_temp ?? null;
  const riskLevel = riskData?.risk_level ?? null;

  return (
    <CircleMarker
      center={[station.lat, station.lon]}
      radius={station.type === 'outdoor' ? 14 : 10}
      pathOptions={{
        color,
        fillColor: color,
        fillOpacity: 0.7,
        weight: 2,
      }}
    >
      <Popup>
        <div className="min-w-[180px] text-sm" style={{ color: '#f1f5f9' }}>
          <p className="font-bold text-base mb-1" style={{ color: '#38bdf8' }}>
            {station.name}
          </p>
          <p className="text-xs mb-2" style={{ color: '#94a3b8' }}>
            {station.type.toUpperCase()} &bull; {station.lat.toFixed(4)}N, {station.lon.toFixed(4)}E
          </p>

          <div className="space-y-1">
            {humidex != null && (
              <div className="flex justify-between">
                <span style={{ color: '#94a3b8' }}>Humidex</span>
                <span className="font-semibold" style={{ color }}>{humidex.toFixed(1)}</span>
              </div>
            )}
            {temp != null && (
              <div className="flex justify-between">
                <span style={{ color: '#94a3b8' }}>Avg Temp</span>
                <span className="font-semibold" style={{ color: '#f1f5f9' }}>{temp.toFixed(1)}°C</span>
              </div>
            )}
            {riskLevel && (
              <div className="flex justify-between items-center mt-2">
                <span style={{ color: '#94a3b8' }}>Risk</span>
                <span
                  className="text-xs font-bold px-2 py-0.5 rounded-full"
                  style={{ backgroundColor: color, color: '#fff' }}
                >
                  {riskLevel}
                </span>
              </div>
            )}
          </div>
        </div>
      </Popup>
    </CircleMarker>
  );
}
