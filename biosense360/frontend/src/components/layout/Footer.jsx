/**
 * Footer — persistent scientific references bar.
 *
 * Displays all relevant protocol citations and the Humidex formula
 * as required by the academic specification.
 */

export default function Footer() {
  return (
    <footer className="bg-[#1a1f2e] border-t border-[#2d3548] mt-auto">
      <div className="max-w-screen-2xl mx-auto px-6 py-3">
        <p className="text-[10px] text-[#94a3b8] text-center leading-relaxed">
          <span className="text-[#38bdf8] font-semibold">Thresholds:</span>{' '}
          OHCOW 2022 &bull; ACGIH TLV 2023 &bull; NIOSH 2016
          &ensp;|&ensp;
          <span className="text-[#38bdf8] font-semibold">Protocols:</span>{' '}
          HAS 2023 &bull; INRS R-447 &bull; Decree No. 2025-482
          &ensp;|&ensp;
          <span className="text-[#38bdf8] font-semibold">Humidex:</span>{' '}
          Masterton &amp; Richardson 1979 &mdash;{' '}
          <span className="font-mono">
            e = 6.112 &times; 10<sup>(7.5T/(237.7+T))</sup> &times; RH/100
          </span>
          &ensp;|&ensp;
          <span className="text-[#38bdf8] font-semibold">BioSense360 v3.0</span>
        </p>
      </div>
    </footer>
  );
}
