/**
 * IrrigationPanel — form + results panel for irrigation recommendations.
 * Supports optional plant photo upload for PlantNet species identification.
 */
import { useRef, useState } from 'react';
import { useIrrigationRecommendation } from '../../hooks/useRecommendations.js';
import client from '../../api/client.js';

const PLANT_TYPES = ['Vegetables', 'Flowers', 'Crops', 'Grass'];
const SOIL_TYPES = ['Clay', 'Sandy', 'Loamy'];

export default function IrrigationPanel({ date, stationId, currentHumidex }) {
  const [plantType, setPlantType] = useState('Vegetables');
  const [soilType, setSoilType] = useState('Loamy');
  const { recommendation, loading, error, fetchRecommendation } = useIrrigationRecommendation();

  // Plant identification state
  const [photoFile, setPhotoFile] = useState(null);
  const [photoPreview, setPhotoPreview] = useState(null);
  const [identifying, setIdentifying] = useState(false);
  const [identification, setIdentification] = useState(null);
  const [identifyError, setIdentifyError] = useState(null);
  const fileRef = useRef();

  function handlePhotoChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setPhotoFile(file);
    setPhotoPreview(URL.createObjectURL(file));
    setIdentification(null);
    setIdentifyError(null);
  }

  async function handleIdentify() {
    if (!photoFile) return;
    setIdentifying(true);
    setIdentifyError(null);
    try {
      const form = new FormData();
      form.append('file', photoFile);
      const humidex = currentHumidex ?? 30;
      const { data } = await client.post(
        `/api/plants/identify?organ=leaf&avg_humidex=${humidex}`,
        form,
        { headers: { 'Content-Type': 'multipart/form-data' } }
      );
      setIdentification(data);
    } catch (err) {
      setIdentifyError(err?.response?.data?.detail || err.message || 'Identification failed');
    } finally {
      setIdentifying(false);
    }
  }

  async function handleFetch() {
    if (!date || !stationId) return;
    await fetchRecommendation({
      date,
      station_id: stationId,
      plant_type: plantType,
      soil_type: soilType,
    });
  }

  return (
    <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-5 space-y-4">
      <h3 className="text-[#f1f5f9] font-semibold">Irrigation Recommendations</h3>

      {/* Photo identification section */}
      <div className="border border-dashed border-[#2d3548] rounded-lg p-3 space-y-2">
        <p className="text-[#94a3b8] text-xs font-medium">Optional: Identify plant from photo</p>
        <div className="flex gap-2 items-center flex-wrap">
          <input
            ref={fileRef}
            type="file"
            accept="image/*"
            onChange={handlePhotoChange}
            className="hidden"
          />
          <button
            onClick={() => fileRef.current?.click()}
            className="text-xs border border-[#2d3548] text-[#94a3b8] hover:text-[#f1f5f9] rounded-lg px-3 py-1.5 transition-colors"
          >
            {photoFile ? photoFile.name.slice(0, 20) + '...' : 'Choose photo'}
          </button>
          {photoFile && (
            <button
              onClick={handleIdentify}
              disabled={identifying}
              className="text-xs bg-[#22c55e22] border border-[#22c55e44] text-[#22c55e] hover:bg-[#22c55e33] rounded-lg px-3 py-1.5 transition-colors disabled:opacity-50"
            >
              {identifying ? 'Identifying...' : 'Identify Species'}
            </button>
          )}
        </div>

        {photoPreview && (
          <img src={photoPreview} alt="plant" className="h-24 rounded-lg object-cover border border-[#2d3548]" />
        )}

        {identifyError && <p className="text-[#ef4444] text-xs">{identifyError}</p>}

        {identification && (
          <div className="space-y-1 pt-1">
            {identification.success && identification.top_result ? (
              <>
                <p className="text-[#f1f5f9] text-xs font-semibold italic">
                  {identification.top_result.species}
                  <span className="text-[#94a3b8] font-normal ml-1">
                    ({(identification.top_result.score * 100).toFixed(1)}% confidence)
                  </span>
                </p>
                {identification.top_result.common_names?.length > 0 && (
                  <p className="text-[#94a3b8] text-xs">
                    Common: {identification.top_result.common_names.slice(0, 2).join(', ')}
                  </p>
                )}
                <div className="flex items-center gap-2 flex-wrap">
                  <span className="text-[#94a3b8] text-xs">Water need:</span>
                  <span className="text-xs bg-[#38bdf822] text-[#38bdf8] px-2 py-0.5 rounded-full border border-[#38bdf844] capitalize">
                    {identification.water_need_class}
                  </span>
                  <span className="text-[#94a3b8] text-xs">Heat risk:</span>
                  <span className={`text-xs px-2 py-0.5 rounded-full border ${
                    identification.heat_stress_risk === 'LOW'
                      ? 'text-[#22c55e] bg-[#22c55e22] border-[#22c55e44]'
                      : identification.heat_stress_risk === 'MODERATE'
                        ? 'text-[#eab308] bg-[#eab30822] border-[#eab30844]'
                        : 'text-[#ef4444] bg-[#ef444422] border-[#ef444444]'
                  }`}>
                    {identification.heat_stress_risk}
                  </span>
                </div>
                {identification.care_advice && (
                  <p className="text-xs text-[#94a3b8] mt-1">{identification.care_advice}</p>
                )}
              </>
            ) : (
              <p className="text-[#94a3b8] text-xs">{identification.message}</p>
            )}
          </div>
        )}
      </div>

      {/* Existing form */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-[#94a3b8] text-xs block mb-1">Plant Type</label>
          <select
            value={plantType}
            onChange={(e) => setPlantType(e.target.value)}
            className="w-full bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8]"
          >
            {PLANT_TYPES.map((p) => <option key={p} value={p}>{p}</option>)}
          </select>
        </div>
        <div>
          <label className="text-[#94a3b8] text-xs block mb-1">Soil Type</label>
          <select
            value={soilType}
            onChange={(e) => setSoilType(e.target.value)}
            className="w-full bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8]"
          >
            {SOIL_TYPES.map((s) => <option key={s} value={s}>{s}</option>)}
          </select>
        </div>
      </div>

      <button
        onClick={handleFetch}
        disabled={loading || !stationId}
        className="w-full bg-[#38bdf8] hover:bg-[#0ea5e9] text-[#0f1117] font-semibold rounded-lg py-2 text-sm transition-colors disabled:opacity-50"
      >
        {loading ? 'Calculating...' : 'Calculate Irrigation Plan'}
      </button>

      {!stationId && (
        <p className="text-[#94a3b8] text-xs">Select a station to get irrigation advice.</p>
      )}

      {error && <p className="text-[#ef4444] text-xs">{error}</p>}

      {recommendation && (
        <div className="space-y-3 pt-2">
          <div
            className="flex items-center gap-2 p-3 rounded-lg border"
            style={{
              borderColor: recommendation.should_irrigate ? '#22c55e44' : '#ef444444',
              backgroundColor: recommendation.should_irrigate ? '#22c55e11' : '#ef444411',
            }}
          >
            <span
              className="font-semibold text-sm"
              style={{ color: recommendation.should_irrigate ? '#22c55e' : '#ef4444' }}
            >
              {recommendation.should_irrigate ? 'Irrigation Recommended' : 'No Irrigation Needed'}
            </span>
          </div>

          <div className="grid grid-cols-2 gap-3 text-sm">
            <InfoItem label="Water Volume" value={`${recommendation.water_liters} L/m²`} />
            <InfoItem label="Frequency" value={recommendation.frequency} />
          </div>

          <div>
            <p className="text-[#94a3b8] text-xs font-medium mb-1">Best Time Slots</p>
            <div className="flex flex-wrap gap-2">
              {recommendation.best_slots.map((slot) => (
                <span key={slot} className="text-xs bg-[#38bdf822] text-[#38bdf8] px-2 py-0.5 rounded-full border border-[#38bdf844]">
                  {slot}
                </span>
              ))}
            </div>
          </div>

          <p className="text-xs text-[#94a3b8]">{recommendation.reason}</p>

          {recommendation.alert && (
            <div className="bg-[#eab30822] border border-[#eab30844] rounded-lg p-3">
              <p className="text-xs text-[#eab308]">{recommendation.alert}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

function InfoItem({ label, value }) {
  return (
    <div className="bg-[#0f1117] rounded-lg p-3 border border-[#2d3548]">
      <p className="text-[#94a3b8] text-xs">{label}</p>
      <p className="text-[#38bdf8] font-semibold text-sm mt-0.5">{value}</p>
    </div>
  );
}
