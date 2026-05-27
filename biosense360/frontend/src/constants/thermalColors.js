/**
 * Thermal comfort and risk color scales for BioSense360.
 *
 * Colors align with universal hazard conventions:
 *   Green  → safe
 *   Yellow → caution
 *   Orange → high alert
 *   Red    → danger
 *   Dark red → extreme / life-threatening
 */

export const THERMAL_COLORS = {
  'Comfortable':     '#22c55e',
  'Caution':         '#eab308',
  'Extreme Caution': '#f97316',
  'Danger':          '#ef4444',
  'Extreme Danger':  '#7f1d1d',
};

export const RISK_COLORS = {
  LOW:      '#22c55e',
  MODERATE: '#eab308',
  HIGH:     '#f97316',
  CRITICAL: '#7f1d1d',
};

/** Text color (foreground) for each comfort class badge */
export const THERMAL_TEXT_COLORS = {
  'Comfortable':     '#ffffff',
  'Caution':         '#000000',
  'Extreme Caution': '#ffffff',
  'Danger':          '#ffffff',
  'Extreme Danger':  '#ffffff',
};

/** Risk level display labels */
export const RISK_LABELS = {
  LOW:      'Low',
  MODERATE: 'Moderate',
  HIGH:     'High',
  CRITICAL: 'Critical',
};

/**
 * Map a comfort class string to its display color.
 * Falls back to grey if class is unknown.
 */
export function getComfortColor(comfortClass) {
  return THERMAL_COLORS[comfortClass] ?? '#64748b';
}

/**
 * Map a risk level string to its display color.
 * Falls back to grey if level is unknown.
 */
export function getRiskColor(riskLevel) {
  return RISK_COLORS[riskLevel] ?? '#64748b';
}

/** Ordered list from safest to most severe — used for legend rendering */
export const COMFORT_CLASS_ORDER = [
  'Comfortable',
  'Caution',
  'Extreme Caution',
  'Danger',
  'Extreme Danger',
];

export const RISK_LEVEL_ORDER = ['LOW', 'MODERATE', 'HIGH', 'CRITICAL'];
