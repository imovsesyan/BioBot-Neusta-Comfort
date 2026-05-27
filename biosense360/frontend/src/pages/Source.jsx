/**
 * Source page — Scientific references and regulatory sources.
 */

const REFERENCES = [
  {
    key: 'OHCOW',
    short: 'OHCOW 2022',
    full: 'Ontario Occupational Health Clinics for Ontario Workers. (2022). Humidex Rating and Work: A Health and Safety Resource for Workers and Employers.',
    url: 'https://www.ohcow.on.ca/edit/files/general_handouts/humidex_brochure.pdf',
    applies: 'Workers, outdoor labour',
  },
  {
    key: 'ACGIH',
    short: 'ACGIH TLV 2023',
    full: 'American Conference of Governmental Industrial Hygienists. (2023). Threshold Limit Values for Chemical Substances and Physical Agents and Biological Exposure Indices.',
    url: 'https://www.acgih.org/tlv-bei-guidelines/',
    applies: 'All occupational heat exposure',
  },
  {
    key: 'NIOSH',
    short: 'NIOSH 2016',
    full: 'National Institute for Occupational Safety and Health. (2016). NIOSH Criteria for a Recommended Standard: Occupational Exposure to Heat and Hot Environments. DHHS Publication No. 2016-106.',
    url: 'https://www.cdc.gov/niosh/docs/2016-106/',
    applies: 'Occupational heat, all industries',
  },
  {
    key: 'HAS',
    short: 'HAS 2023',
    full: 'Haute Autorité de Santé. (2023). Prise en charge des personnes exposées à une chaleur extrême — Protocole de prévention et de traitement du coup de chaleur.',
    url: 'https://www.has-sante.fr/',
    applies: 'Elderly, children, vulnerable populations',
  },
  {
    key: 'INRS',
    short: 'INRS R-447',
    full: 'Institut National de Recherche et de Sécurité. Risques liés à la chaleur — Recommandation R-447.',
    url: 'https://www.inrs.fr/risques/chaleur.html',
    applies: 'French occupational safety — all workers',
  },
  {
    key: 'DECREE',
    short: 'Décret 2025-482',
    full: 'Décret n° 2025-482 du 27 mai 2025 relatif à la prévention des risques liés aux fortes chaleurs sur les lieux de travail. Journal Officiel de la République Française.',
    url: 'https://www.legifrance.gouv.fr/',
    applies: 'All French employers — legal obligation',
  },
  {
    key: 'MASTERTON',
    short: 'Masterton & Richardson 1979',
    full: 'Masterton, J.M. & Richardson, F.A. (1979). Humidex: A Method of Quantifying Human Discomfort Due to Excessive Heat and Humidity. CLI 1-79. Atmospheric Environment Service, Downsview, Ontario.',
    url: 'https://climate.weather.gc.ca/glossary_e.html#humidex',
    applies: 'Humidex formula definition',
  },
];

export default function Source() {
  return (
    <div className="space-y-8">
      <div className="bg-[#1a1f2e] border border-[#2d3548] rounded-xl p-5 space-y-4">
        <h2 className="text-[#f1f5f9] font-semibold">Scientific References & Regulatory Sources</h2>
        <div className="overflow-x-auto">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="border-b border-[#2d3548]">
                <th className="text-left text-[#94a3b8] font-medium p-3 w-[160px]">Reference</th>
                <th className="text-left text-[#94a3b8] font-medium p-3">Full Citation</th>
                <th className="text-left text-[#94a3b8] font-medium p-3 w-[220px]">Applies To</th>
              </tr>
            </thead>
            <tbody>
              {REFERENCES.map((ref) => (
                <tr key={ref.key} className="border-b border-[#2d3548] hover:bg-[#0f1117]">
                  <td className="p-3">
                    <a
                      href={ref.url}
                      target="_blank"
                      rel="noopener noreferrer"
                      className="text-[#38bdf8] hover:underline font-semibold text-xs"
                    >
                      {ref.short}
                    </a>
                  </td>
                  <td className="p-3 text-[#f1f5f9] text-xs leading-relaxed">{ref.full}</td>
                  <td className="p-3 text-[#94a3b8] text-xs">{ref.applies}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
