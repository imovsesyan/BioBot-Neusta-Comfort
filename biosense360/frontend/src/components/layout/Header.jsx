/**
 * Header — persistent top bar for BioSense360.
 *
 * Displays the application title, subtitle, and a live UTC clock.
 */

import { useEffect, useState } from 'react';

export default function Header() {
  const [time, setTime] = useState(() => new Date().toUTCString());

  // Update clock every second
  useEffect(() => {
    const interval = setInterval(() => {
      setTime(new Date().toUTCString());
    }, 1000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="sticky top-0 z-50 bg-[#1a1f2e] border-b border-[#2d3548] shadow-lg">
      <div className="max-w-screen-2xl mx-auto px-6 py-3 flex items-center justify-between">
        {/* Brand */}
        <div className="flex items-center gap-3">
          <div className="flex flex-col">
            <span className="text-xl font-bold text-[#38bdf8] tracking-tight leading-none">
              BioSense360
            </span>
            <span className="text-xs text-[#94a3b8] leading-none mt-0.5">
              Thermal Comfort Monitor | Toulouse
            </span>
          </div>
        </div>

        {/* Right side info */}
        <div className="flex items-center gap-6 text-xs text-[#94a3b8]">
<span className="font-mono text-[#38bdf8]">{time}</span>
        </div>
      </div>
    </header>
  );
}
