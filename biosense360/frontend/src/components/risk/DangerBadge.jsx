/**
 * DangerBadge — pill badge showing a risk level with its corresponding color.
 *
 * Accepts either a risk level ("LOW" | "MODERATE" | "HIGH" | "CRITICAL")
 * or a comfort class ("Comfortable" | "Caution" | ...).
 */

import { getRiskColor, getComfortColor } from '../../constants/thermalColors.js';

const RISK_LABELS = {
  LOW:      'LOW',
  MODERATE: 'MODERATE',
  HIGH:     'HIGH',
  CRITICAL: 'CRITICAL',
};

export default function DangerBadge({ level, type = 'risk', size = 'md' }) {
  const color = type === 'comfort' ? getComfortColor(level) : getRiskColor(level);
  const label = type === 'comfort' ? level : (RISK_LABELS[level] ?? level);

  const sizeClasses = {
    sm: 'text-xs px-2 py-0.5',
    md: 'text-sm px-3 py-1',
    lg: 'text-base px-4 py-1.5 font-bold',
  }[size] ?? 'text-sm px-3 py-1';

  return (
    <span
      className={`inline-flex items-center rounded-full font-semibold ${sizeClasses}`}
      style={{ backgroundColor: color, color: '#fff' }}
    >
      {label}
    </span>
  );
}
