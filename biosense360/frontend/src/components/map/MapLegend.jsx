/**
 * MapLegend — overlay legend for the thermal comfort map.
 *
 * Positioned as a Leaflet custom control in the bottom-right corner.
 * Lists all 5 comfort classes with their colors.
 */

import { useEffect } from 'react';
import { useMap } from 'react-leaflet';
import L from 'leaflet';
import { COMFORT_CLASS_ORDER, THERMAL_COLORS } from '../../constants/thermalColors.js';

export default function MapLegend() {
  const map = useMap();

  useEffect(() => {
    // Create a Leaflet custom control for the legend
    const legend = L.control({ position: 'bottomright' });

    legend.onAdd = () => {
      const div = L.DomUtil.create('div');
      div.style.cssText = [
        'background:#1a1f2e',
        'border:1px solid #2d3548',
        'border-radius:8px',
        'padding:10px 12px',
        'font-size:11px',
        'color:#f1f5f9',
        'min-width:140px',
        'box-shadow:0 4px 12px rgba(0,0,0,0.4)',
      ].join(';');

      const title = document.createElement('p');
      title.textContent = 'Comfort Class';
      title.style.cssText = 'font-weight:700;color:#38bdf8;margin:0 0 6px;font-size:12px';
      div.appendChild(title);

      COMFORT_CLASS_ORDER.forEach((cls) => {
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:6px;margin-bottom:3px';

        const dot = document.createElement('div');
        dot.style.cssText = `width:10px;height:10px;border-radius:50%;background:${THERMAL_COLORS[cls]};flex-shrink:0`;

        const label = document.createElement('span');
        label.textContent = cls;
        label.style.color = '#94a3b8';

        row.appendChild(dot);
        row.appendChild(label);
        div.appendChild(row);
      });

      return div;
    };

    legend.addTo(map);
    return () => legend.remove();
  }, [map]);

  return null;
}
