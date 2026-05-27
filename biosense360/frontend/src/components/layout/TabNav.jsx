/**
 * TabNav — primary navigation bar with 4 tabs.
 *
 * Uses simple index state rather than a router to keep the app
 * dependency-free from react-router for this embedded use case.
 */

const TABS = [
  { id: 'daily',     label: 'Daily Analysis' },
  { id: 'map',       label: 'Map & Mobility' },
  { id: 'dashboard', label: 'Dashboard' },
  { id: 'source',    label: 'Source' },
];

export default function TabNav({ activeTab, onTabChange }) {
  return (
    <nav className="bg-[#1a1f2e] border-b border-[#2d3548]">
      <div className="max-w-screen-2xl mx-auto px-6">
        <div className="flex gap-1">
          {TABS.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                onClick={() => onTabChange(tab.id)}
                className={[
                  'px-5 py-3 text-sm font-medium transition-all duration-150',
                  'border-b-2 focus:outline-none',
                  isActive
                    ? 'border-[#38bdf8] text-[#38bdf8]'
                    : 'border-transparent text-[#94a3b8] hover:text-[#f1f5f9] hover:border-[#2d3548]',
                ].join(' ')}
                aria-current={isActive ? 'page' : undefined}
              >
                {tab.label}
              </button>
            );
          })}
        </div>
      </div>
    </nav>
  );
}
