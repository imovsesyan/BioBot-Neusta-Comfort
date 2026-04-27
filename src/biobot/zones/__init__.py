"""F12 zones: temporal / spatial windowing and thermal protection slots.

Modules
-------
temporal           — diurnal-window definitions and humidex→plant-zone label mapping
                     (F12-UC1).
protection_slots   — rolling-window grouping of plant-zone labels into safe / danger /
                     transition slots and a daily action schedule (F12-UC2).

Scientific note
---------------
F12 is a **windowing and labeling layer over F10 humidex risk levels**. It does not
introduce a new model and does not consume `vivabilite_binary_mean` or any F10-UC4
classifier output (see CLAUDE.md target-leakage guardrails and Section 4 of
`docs/f13/F13_research_plants.md`).
"""

from biobot.zones.protection_slots import (
    generate_daily_schedule,
    generate_protection_slots,
)
from biobot.zones.temporal import (
    DIURNAL_WINDOWS,
    HUMIDEX_TO_PLANT_ZONE,
    PLANT_ZONE_LABELS,
    assign_diurnal_window,
    assign_plant_zone_label,
)

__all__ = [
    "DIURNAL_WINDOWS",
    "HUMIDEX_TO_PLANT_ZONE",
    "PLANT_ZONE_LABELS",
    "assign_diurnal_window",
    "assign_plant_zone_label",
    "generate_daily_schedule",
    "generate_protection_slots",
]
