/**
 * HumanRecoPanel — form + results panel for human thermal recommendations.
 *
 * User selects their profile (age group) and activity, then fetches
 * personalised advice from POST /api/recommendations/human.
 */

import { useState } from 'react';
import { useHumanRecommendation } from '../../hooks/useRecommendations.js';
import DangerBadge from '../risk/DangerBadge.jsx';

const AGE_GROUPS = ['Worker', 'Elderly', 'Child', 'Athlete'];
const ACTIVITY_TYPES = ['Outdoor work', 'Sports', 'Commuting', 'Rest'];

export default function HumanRecoPanel({ currentHumidex, currentClass }) {
  const [ageGroup, setAgeGroup] = useState('Worker');
  const [activity, setActivity] = useState('Outdoor work');
  const { recommendation, loading, error, fetchRecommendation } = useHumanRecommendation();

  const humidex = currentHumidex ?? 35;
  const comfortClass = currentClass ?? 'Caution';

  async function handleFetch() {
    await fetchRecommendation({
      humidex,
      comfort_class: comfortClass,
      age_group: ageGroup,
      activity_type: activity,
    });
  }

  return (
    <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-5 space-y-4">
      <h3 className="text-[#f1f5f9] font-semibold">Human Safety Recommendations</h3>

      {/* Inputs */}
      <div className="grid grid-cols-2 gap-3">
        <div>
          <label className="text-[#94a3b8] text-xs block mb-1">Age Group</label>
          <select
            value={ageGroup}
            onChange={(e) => setAgeGroup(e.target.value)}
            className="w-full bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8]"
          >
            {AGE_GROUPS.map((g) => (
              <option key={g} value={g}>{g}</option>
            ))}
          </select>
        </div>

        <div>
          <label className="text-[#94a3b8] text-xs block mb-1">Activity</label>
          <select
            value={activity}
            onChange={(e) => setActivity(e.target.value)}
            className="w-full bg-[#0f1117] border border-[#2d3548] rounded-lg text-[#f1f5f9] text-sm px-3 py-2 focus:outline-none focus:border-[#38bdf8]"
          >
            {ACTIVITY_TYPES.map((a) => (
              <option key={a} value={a}>{a}</option>
            ))}
          </select>
        </div>
      </div>

      <div className="text-xs text-[#94a3b8]">
        Current conditions: Humidex <span className="text-[#38bdf8] font-mono">{humidex.toFixed(1)}</span> ({comfortClass})
      </div>

      <button
        onClick={handleFetch}
        disabled={loading}
        className="w-full bg-[#38bdf8] hover:bg-[#0ea5e9] text-[#0f1117] font-semibold rounded-lg py-2 text-sm transition-colors disabled:opacity-50"
      >
        {loading ? 'Generating...' : 'Get Recommendations'}
      </button>

      {error && (
        <p className="text-[#ef4444] text-xs">{error}</p>
      )}

      {/* Results */}
      {recommendation && (
        <div className="space-y-3 pt-2">
          <div className="flex items-center gap-2">
            <span className="text-[#94a3b8] text-xs">Risk Level:</span>
            <DangerBadge level={recommendation.risk_level} size="sm" />
          </div>

          <div className="space-y-2 text-sm">
            <RecoItem label="Advice" value={recommendation.advice} />
            <RecoItem label="Hydration" value={recommendation.hydration} />
            <RecoItem label="Clothing" value={recommendation.clothing} />
            <div>
              <p className="text-[#94a3b8] text-xs font-medium mb-1">Safe Time Slots</p>
              <div className="flex flex-wrap gap-2">
                {recommendation.safe_slots.map((slot) => (
                  <span key={slot} className="text-xs bg-[#22c55e22] text-[#22c55e] px-2 py-0.5 rounded-full border border-[#22c55e44]">
                    {slot}
                  </span>
                ))}
              </div>
            </div>
            {recommendation.emergency_protocol && (
              <div className="bg-[#ef444422] border border-[#ef444444] rounded-lg p-3">
                <p className="text-[#94a3b8] text-xs font-medium mb-1">Emergency Protocol</p>
                <p className="text-xs text-[#f1f5f9]">{recommendation.emergency_protocol}</p>
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

function RecoItem({ label, value }) {
  return (
    <div>
      <p className="text-[#94a3b8] text-xs font-medium">{label}</p>
      <p className="text-[#f1f5f9] text-sm mt-0.5">{value}</p>
    </div>
  );
}
