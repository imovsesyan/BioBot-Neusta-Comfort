/**
 * App — root component for BioSense360.
 *
 * Manages tab navigation state and renders the appropriate page
 * with a smooth fade transition. Header, TabNav, and Footer are always visible.
 */

import { useState } from 'react';
import Header from './components/layout/Header.jsx';
import Footer from './components/layout/Footer.jsx';
import TabNav from './components/layout/TabNav.jsx';
import DailyAnalysis from './pages/DailyAnalysis.jsx';
import MapMobility from './pages/MapMobility.jsx';
import Dashboard from './pages/Dashboard.jsx';
import Source from './pages/Source.jsx';

const PAGE_MAP = {
  daily:     DailyAnalysis,
  map:       MapMobility,
  dashboard: Dashboard,
  source:    Source,
};

export default function App() {
  const [activeTab, setActiveTab] = useState('daily');

  const PageComponent = PAGE_MAP[activeTab];

  return (
    <div className="min-h-screen bg-[#0f1117] flex flex-col">
      <Header />
      <TabNav activeTab={activeTab} onTabChange={setActiveTab} />

      {/* Page content with fade animation */}
      <main
        key={activeTab}
        className="flex-1 max-w-screen-2xl mx-auto w-full px-6 py-6"
        style={{
          animation: 'fadeIn 200ms ease forwards',
        }}
      >
        <PageComponent />
      </main>

      <Footer />

      {/* Inline keyframe for tab fade — avoids a separate CSS file */}
      <style>{`
        @keyframes fadeIn {
          from { opacity: 0; transform: translateY(6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}
