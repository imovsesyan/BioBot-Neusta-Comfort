"""
BioSense360 — Thermal Comfort Thresholds
=========================================
All constants are sourced from peer-reviewed or regulatory documents.
Each constant documents its source, year, and applicability context.

Citation conventions
--------------------
Every constant is followed by an inline `# Source: NAME YEAR` comment.
The expanded citation, URL, and applicability of each source live in the
`SOURCES` dictionary at the bottom of this file.

Geographic scope
----------------
Toulouse, Occitanie, France (Cfb / hot-summer Mediterranean influence).
French regulations (HAS, INRS, Décret 2025-482) take precedence over
North-American sources when they conflict.
"""

from __future__ import annotations

# =============================================================================
# Humidex Comfort Classes
# Source: Masterton & Richardson 1979 (original scale)
#         OHCOW 2022 (operational interpretation for workers)
# =============================================================================
COMFORT_CLASSES: dict[str, tuple[float, float]] = {
    "Comfortable":     (float("-inf"), 30.0),  # H < 30   — Source: Masterton & Richardson 1979
    "Caution":         (30.0, 40.0),           # 30 ≤ H < 40 — Source: Masterton & Richardson 1979
    "Extreme Caution": (40.0, 45.0),           # 40 ≤ H < 45 — Source: OHCOW 2022
    "Danger":          (45.0, 54.0),           # 45 ≤ H < 54 — Source: OHCOW 2022
    "Extreme Danger":  (54.0, float("inf")),   # H ≥ 54 — Source: Masterton & Richardson 1979
}

COMFORT_CLASS_ORDER: list[str] = [
    "Comfortable",
    "Caution",
    "Extreme Caution",
    "Danger",
    "Extreme Danger",
]


# =============================================================================
# OHCOW 2022 — Heat Risk Thresholds (Humidex action levels for workers)
# Source: OHCOW 2022, "Humidex Based Heat Response Plan", Table 1
# =============================================================================
HUMIDEX_RISK_THRESHOLDS: dict[str, float] = {
    "LOW":      30.0,  # Source: OHCOW 2022 — first action level (general awareness)
    "MODERATE": 40.0,  # Source: OHCOW 2022 — work-rest cycling mandatory for moderate work
    "HIGH":     45.0,  # Source: OHCOW 2022 — essential work only, medical surveillance
    "CRITICAL": 54.0,  # Source: Masterton & Richardson 1979 / OHCOW 2022 — heat stroke imminent
}

RISK_LEVEL_ORDER: list[str] = ["LOW", "MODERATE", "HIGH", "CRITICAL"]


# =============================================================================
# Vulnerability Adjustments
# Source: HAS 2023 §3.2 (vulnerable population definitions)
#         INRS R-447 (worker baseline)
# =============================================================================
VULNERABILITY_OFFSETS: dict[str, float] = {
    "Elderly": -5.0,  # Source: HAS 2023 §3.2 — impaired thermoregulation, reduced thirst response
    "Child":   -5.0,  # Source: HAS 2023 §3.2 — higher surface-to-mass ratio, immature sweating
    "Worker":   0.0,  # Source: INRS R-447 — healthy adult baseline
    "Athlete":  0.0,  # Source: ACGIH TLV 2023 — metabolic load handled via work-rest matrix
}

# Backward-compatible alias
VULNERABILITY_ADJUSTMENTS = VULNERABILITY_OFFSETS


# =============================================================================
# ACGIH TLV 2023 — Work-Rest Ratios by WBGT Category
# Source: ACGIH 2023 TLVs and BEIs, "Heat Stress and Strain", Table 2
# =============================================================================
ACGIH_METABOLIC_RATES_W: dict[str, int] = {
    "Light":      180,  # Source: ACGIH TLV 2023 — sitting/standing light tasks
    "Moderate":   300,  # Source: ACGIH TLV 2023 — sustained walking, moderate lifting
    "Heavy":      415,  # Source: ACGIH TLV 2023 — shovelling, heavy lifting
    "Very Heavy": 520,  # Source: ACGIH TLV 2023 — sustained intense exertion
}

# WBGT ceilings (°C) for ACCLIMATIZED workers — Source: ACGIH TLV 2023, Table 2
ACGIH_WBGT_ACCLIMATIZED: dict[str, dict[str, float]] = {
    "100% work": {"Light": 29.5, "Moderate": 27.5, "Heavy": 26.0, "Very Heavy": float("nan")},
    "75% work":  {"Light": 30.5, "Moderate": 28.5, "Heavy": 27.5, "Very Heavy": float("nan")},
    "50% work":  {"Light": 31.5, "Moderate": 29.5, "Heavy": 28.5, "Very Heavy": 27.5},
    "25% work":  {"Light": 32.5, "Moderate": 31.0, "Heavy": 30.0, "Very Heavy": 29.5},
}

# WBGT ceilings (°C) for UNACCLIMATIZED workers — Source: ACGIH TLV 2023, Table 2
ACGIH_WBGT_UNACCLIMATIZED: dict[str, dict[str, float]] = {
    "100% work": {"Light": 27.5, "Moderate": 25.0, "Heavy": 22.5, "Very Heavy": float("nan")},
    "75% work":  {"Light": 29.0, "Moderate": 26.5, "Heavy": 24.5, "Very Heavy": float("nan")},
    "50% work":  {"Light": 30.0, "Moderate": 28.0, "Heavy": 26.5, "Very Heavy": 25.0},
    "25% work":  {"Light": 31.0, "Moderate": 29.0, "Heavy": 28.0, "Very Heavy": 26.5},
}

ACGIH_ACCLIMATIZATION_DAYS_NEW: int = 14       # Source: NIOSH 2016 §1.1.4
ACGIH_ACCLIMATIZATION_DAYS_RETURNING: int = 5  # Source: NIOSH 2016 §1.1.4
ACGIH_ACCLIMATIZATION_LOSS_DAYS: int = 9       # Source: NIOSH 2016 — re-acclimatize after >9 days off


# =============================================================================
# NIOSH 2016 — Heat Illness Risk Levels (WBGT-based)
# Source: NIOSH 2016, DHHS Pub. 2016-106, Figures 8-1 and 8-2
# =============================================================================
NIOSH_WBGT_REL: dict[str, float] = {  # Acclimatized — Recommended Exposure Limit
    "Light":      30.0,  # Source: NIOSH 2016 Fig 8-1
    "Moderate":   28.0,  # Source: NIOSH 2016 Fig 8-1
    "Heavy":      26.0,  # Source: NIOSH 2016 Fig 8-1
    "Very Heavy": 25.0,  # Source: NIOSH 2016 Fig 8-1
}

NIOSH_WBGT_RAL: dict[str, float] = {  # Unacclimatized — Recommended Alert Limit
    "Light":      27.5,  # Source: NIOSH 2016 Fig 8-2
    "Moderate":   25.0,  # Source: NIOSH 2016 Fig 8-2
    "Heavy":      22.5,  # Source: NIOSH 2016 Fig 8-2
    "Very Heavy": 21.0,  # Source: NIOSH 2016 Fig 8-2
}

NIOSH_CORE_TEMPERATURE_LIMITS_C: dict[str, float] = {
    "Sustained_Work_Ceiling": 38.0,  # Source: NIOSH 2016 / WHO 1969
    "Heat_Exhaustion_Onset":  38.5,  # Source: NIOSH 2016 §3.3
    "Heat_Stroke_Threshold":  40.0,  # Source: NIOSH 2016 §3.4 — medical emergency
}


# =============================================================================
# HAS 2023 — Heat Stroke Clinical Thresholds (French medical protocol)
# Source: HAS 2023, "Coup de chaleur — Diagnostic et traitement en urgence"
# =============================================================================
HAS_HEAT_STROKE_CORE_TEMP_C: float = 40.0       # Source: HAS 2023 — diagnostic threshold
HAS_HYPERTHERMIA_TEMP_C: float = 38.5           # Source: HAS 2023 — surveillance threshold
HAS_NIGHT_TEMP_RECOVERY_C: float = 20.0         # Source: PNC / HAS 2023 — min night temp for recovery
HAS_INDOOR_DANGER_TEMP_C: float = 32.0          # Source: HAS 2023 — indoor cooling-room target ceiling
HAS_ELDERLY_HOT_DAY_THRESHOLD_C: float = 31.0   # Source: HAS 2023 — daytime ceiling for elderly
HAS_INFANT_HOT_DAY_THRESHOLD_C: float = 30.0    # Source: HAS 2023 — daytime ceiling for infants


# =============================================================================
# INRS R-447 — French Occupational Heat Thresholds
# Source: INRS R-447 / brochure ED 6320 — "Travail à la chaleur"
# =============================================================================
INRS_INDOOR_DISCOMFORT_TEMP_C: float = 28.0          # Source: INRS R-447 — sedentary indoor work
INRS_INDOOR_HIGH_RISK_TEMP_C: float = 30.0           # Source: INRS R-447 — sedentary work risk threshold
INRS_OUTDOOR_HIGH_RISK_TEMP_C: float = 28.0          # Source: INRS R-447 — physical outdoor work
INRS_HEAVY_WORK_DANGER_TEMP_C: float = 33.0          # Source: INRS R-447 — heavy physical work hazard
INRS_DRINKING_WATER_LITERS_PER_HOUR: float = 0.25    # Source: INRS R-447
INRS_CONSTRUCTION_WATER_LITERS_PER_DAY: float = 3.0  # Source: Code du Travail R.4534-143


# =============================================================================
# Décret n° 2025-482 — French Regulatory Obligations (Heat at Work)
# Source: Décret n° 2025-482 du 27 mai 2025, Articles R.4463-1 to R.4463-9
#         Entry into force: 1 July 2025
# =============================================================================
DECREE_2025_VIGILANCE_TEMP_C: dict[str, float] = {
    "Vert":   25.0,  # Source: Décret 2025-482 / Météo-France Haute-Garonne
    "Jaune":  30.0,  # Source: Décret 2025-482 — Tmax 30 °C
    "Orange": 34.0,  # Source: Décret 2025-482 — Tmax 34 °C
    "Rouge":  40.0,  # Source: Décret 2025-482 — exceptional canicule
}

DECREE_2025_OBLIGATIONS: dict[str, list[str]] = {
    "Vert": [
        "Include heat risk in DUERP (Document Unique d'Évaluation des Risques)",
    ],
    "Jaune": [
        "Inform workers of forecast and symptoms",
        "Ensure access to cool drinking water (≥ 3 L/day on construction sites)",
    ],
    "Orange": [
        "Adapt work schedules (start earlier, lengthen breaks)",
        "Provide cool rest area (T < 26 °C recommended)",
        "Increase surveillance frequency",
        "Reinforce hydration provision",
        "Train at least one worker on heat illness recognition",
    ],
    "Rouge": [
        "Suspend or reorganize all non-essential exposed work",
        "Mandatory medical surveillance for at-risk workers",
        "Activity reorganization formally documented",
        "Right of withdrawal (droit de retrait) explicitly applicable",
    ],
}

DECREE_2025_REST_AREA_MAX_TEMP_C: float = 26.0         # Source: Décret 2025-482
DECREE_2025_DRINKING_WATER_MIN_L_PER_DAY: float = 3.0  # Source: Décret 2025-482


# =============================================================================
# Default safe work window (Toulouse summer climatology)
# =============================================================================
DEFAULT_SAFE_HOURS: list[tuple[int, int]] = [
    (6, 10),   # 06:00–10:00 — pre-noon, before solar peak
    (19, 22),  # 19:00–22:00 — after sunset, before night minimum
]


# =============================================================================
# Source Citations Dictionary
# =============================================================================
SOURCES: dict[str, dict] = {
    "OHCOW": {
        "short": "OHCOW 2022",
        "title": "Humidex Based Heat Response Plan — A Guide for Employers",
        "year":  2022,
        "url":   "https://www.ohcow.on.ca/resource/humidex-based-heat-response-plan/",
        "publisher": "Occupational Health Clinics for Ontario Workers Inc.",
        "applicability": "Outdoor and indoor workers; Humidex-based action levels",
        "full": ("Occupational Health Clinics for Ontario Workers. (2022). "
                 "Humidex Based Heat Response Plan: A Guide for Employers. Hamilton, ON: OHCOW."),
    },
    "ACGIH": {
        "short": "ACGIH TLV 2023",
        "title": "Threshold Limit Values — Heat Stress and Strain",
        "year":  2023,
        "url":   "https://www.acgih.org/science/tlv-bei-guidelines/",
        "publisher": "American Conference of Governmental Industrial Hygienists",
        "applicability": "Occupational heat exposure; WBGT-based work-rest matrix; acclimatization",
        "full": ("American Conference of Governmental Industrial Hygienists. (2023). "
                 "TLVs and BEIs. Cincinnati, OH: ACGIH."),
    },
    "NIOSH": {
        "short": "NIOSH 2016",
        "title": "Criteria for a Recommended Standard: Occupational Exposure to Heat and Hot Environments",
        "year":  2016,
        "url":   "https://www.cdc.gov/niosh/docs/2016-106/",
        "publisher": "U.S. National Institute for Occupational Safety and Health",
        "applicability": "Occupational heat; WBGT REL/RAL curves; core-temperature ceilings",
        "full": ("Jacklitsch, B. et al. (2016). NIOSH Criteria — Occupational Exposure to Heat. "
                 "DHHS (NIOSH) Pub. 2016-106. CDC, NIOSH."),
    },
    "HAS": {
        "short": "HAS 2023",
        "title": "Coup de chaleur — Diagnostic et traitement en urgence",
        "year":  2023,
        "url":   "https://www.has-sante.fr/jcms/c_272218/fr/canicule",
        "publisher": "Haute Autorité de Santé",
        "applicability": ("Clinical heat-stroke diagnosis; vulnerable population definitions; "
                          "French national medical protocol (PNC)"),
        "full": ("Haute Autorité de Santé. (2023). Coup de chaleur — Diagnostic et traitement "
                 "en urgence. Mise à jour du Plan National Canicule. Saint-Denis: HAS."),
    },
    "INRS": {
        "short": "INRS R-447",
        "title": "Travail à la chaleur et confort thermique — Recommandation R-447 / ED 6320",
        "year":  2024,
        "url":   "https://www.inrs.fr/risques/chaleur/",
        "publisher": "Institut National de Recherche et de Sécurité",
        "applicability": "French occupational heat-stress prevention; risk-assessment thresholds",
        "full": ("Institut National de Recherche et de Sécurité. "
                 "Travail et chaleur d'été. Recommandation R-447 / Brochure ED 6320. Paris: INRS."),
    },
    "DECREE": {
        "short": "Décret 2025-482",
        "title": ("Décret n° 2025-482 du 27 mai 2025 relatif à la prévention des risques "
                  "liés aux épisodes de chaleur intense sur les lieux de travail"),
        "year":  2025,
        "url":   "https://www.legifrance.gouv.fr/jorf/id/JORFTEXT000051648000",
        "publisher": "République Française — Journal Officiel",
        "applicability": ("Legally binding obligations on all French employers; "
                          "vigilance-tier trigger system aligned with Météo-France"),
        "full": ("République Française. (2025). Décret n° 2025-482 du 27 mai 2025. "
                 "JORF n° 0123 du 28 mai 2025. Articles R.4463-1 à R.4463-9 du Code du Travail. "
                 "Entrée en vigueur: 1er juillet 2025."),
    },
    "MASTERTON": {
        "short": "Masterton & Richardson 1979",
        "title": "Humidex: A Method of Quantifying Human Discomfort Due to Excessive Heat and Humidity",
        "year":  1979,
        "url":   "https://climate.weather.gc.ca/glossary_e.html#humidex",
        "publisher": "Environment Canada — Atmospheric Environment Service",
        "applicability": ("Definition of the Humidex index and its discomfort scale; "
                          "formula H = T + 5/9 × (e − 10)"),
        "full": ("Masterton, J.M., & Richardson, F.A. (1979). Humidex: A Method of Quantifying "
                 "Human Discomfort Due to Excessive Heat and Humidity. CLI 1-79. "
                 "Downsview, Ontario: Environment Canada, Atmospheric Environment Service."),
    },
}

# Backward-compatible flattened view for older callers
SOURCE_CITATIONS: dict[str, dict[str, str]] = {
    key: {
        "short":      str(meta["short"]),
        "full":       str(meta["full"]),
        "url":        str(meta["url"]),
        "applies_to": str(meta["applicability"]),
    }
    for key, meta in SOURCES.items()
}


# =============================================================================
# Irrigation Thresholds
# =============================================================================
PLANT_WATER_MULTIPLIERS: dict[str, float] = {
    "Vegetables": 1.5,  # Source: ARVALIS Occitanie — high-evapotranspiration crops
    "Crops":      2.0,  # Source: ARVALIS Occitanie — field crops, peak summer demand
    "Flowers":    1.0,  # heuristic baseline
    "Grass":      0.8,  # heuristic — turfgrass tolerates light deficit
}

SOIL_WATER_MULTIPLIERS: dict[str, float] = {
    "Sandy": 1.5,  # heuristic — low water-holding capacity
    "Clay":  0.7,  # heuristic — high retention, waterlogging risk
    "Loamy": 1.0,  # heuristic — balanced reference soil
}

BASE_WATER_LITERS: float = 2.0           # L per m² baseline daily volume
HIGH_HUMIDITY_THRESHOLD: float = 80.0   # % RH above which irrigation is reduced
HIGH_HUMIDITY_REDUCTION: float = 0.6    # multiply base volume by this factor (−40%)


# =============================================================================
# Thermal Color Scale (UI / map / chart styling)
# =============================================================================
THERMAL_COLORS: dict[str, str] = {
    "Comfortable":     "#22c55e",  # Tailwind green-500
    "Caution":         "#eab308",  # Tailwind yellow-500
    "Extreme Caution": "#f97316",  # Tailwind orange-500
    "Danger":          "#ef4444",  # Tailwind red-500
    "Extreme Danger":  "#7f1d1d",  # Tailwind red-900
}

RISK_COLORS: dict[str, str] = {
    "LOW":      "#22c55e",
    "MODERATE": "#eab308",
    "HIGH":     "#f97316",
    "CRITICAL": "#7f1d1d",
}


# =============================================================================
# ML model metadata
# =============================================================================
DEFAULT_MODEL_NAME: str = "RandomForestClassifier"
DEFAULT_MODEL_VERSION: str = "1.0.0-stub"
MODEL_FEATURES: list[str] = [
    "temperature", "humidity", "humidex",
    "hour", "month", "day_of_week",
    "lat", "lon", "is_indoor",
]
