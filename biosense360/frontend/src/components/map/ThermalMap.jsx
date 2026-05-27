/**
 * ThermalMap — react-leaflet map showing all station markers.
 *
 * Centered on Toulouse (43.6047°N, 1.4442°E), zoom 12.
 * Marker colors reflect each station's current comfort class.
 */

import { MapContainer, TileLayer } from 'react-leaflet';
import MapLegend from './MapLegend.jsx';
import StationMarker from './StationMarker.jsx';

// Toulouse city centre coordinates
const TOULOUSE_CENTER = [43.6047, 1.4442];
const DEFAULT_ZOOM = 12;

export default function ThermalMap({ stations, riskByStation }) {
  return (
    <MapContainer
      center={TOULOUSE_CENTER}
      zoom={DEFAULT_ZOOM}
      style={{ height: '420px', width: '100%', borderRadius: '0.75rem' }}
      className="border border-[#2d3548]"
    >
      {/* Dark-theme tile layer (CartoDB Dark Matter) */}
      <TileLayer
        url="https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png"
        attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> &copy; <a href="https://carto.com/attributions">CARTO</a>'
        subdomains="abcd"
        maxZoom={19}
      />

      {/* Station markers */}
      {stations.map((station) => (
        <StationMarker
          key={station.id}
          station={station}
          riskData={riskByStation?.[station.id] ?? null}
        />
      ))}

      {/* Legend overlay */}
      <MapLegend />
    </MapContainer>
  );
}
