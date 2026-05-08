"""
Plants router — POST /api/plants/identify

Accepts a photo upload and returns species identification via PlantNet API.
Also returns a basic heat-stress risk assessment derived from the current
humidex level and the identified species' water-need class.
"""

from fastapi import APIRouter, File, HTTPException, Query, UploadFile

from backend.services.humidex_service import humidex_to_risk_level
from backend.services.plantnet_service import identify_plant

router = APIRouter(prefix="/plants", tags=["plants"])

# ---------------------------------------------------------------------------
# Species → water need class mapping
# Keys may be full species names, family names, or genus names (case-insensitive).
# Classes: xeric (drought-tolerant) | mesic (average) | hydric (water-loving)
# ---------------------------------------------------------------------------
WATER_NEED: dict[str, str] = {
    # Xeric — drought-tolerant
    "Cactaceae": "xeric",
    "Agave": "xeric",
    "Sedum": "xeric",
    "Lavandula": "xeric",
    # Hydric — water-loving crops and vegetables
    "Solanum lycopersicum": "hydric",
    "Cucumis sativus": "hydric",
    "Lactuca sativa": "hydric",
    "Zea mays": "hydric",
    "Oryza sativa": "hydric",
}

# Care-advice matrix indexed by [water_need_class][risk_level]
_ADVICE: dict[str, dict[str, str]] = {
    "xeric": {
        "LOW":      "Drought-tolerant species. Water sparingly — once a week is usually sufficient.",
        "MODERATE": "Heat is elevated. Check soil moisture. Water early morning if soil is dry.",
        "HIGH":     "Significant heat stress. Even drought-tolerant plants need hydration now. Water at dawn.",
        "CRITICAL": "Emergency heat conditions. Water immediately at the coolest window (05:00–06:30). Apply mulch.",
    },
    "mesic": {
        "LOW":      "Normal conditions. Water 2–3 times per week as needed.",
        "MODERATE": "Elevated heat. Increase watering frequency. Avoid midday watering.",
        "HIGH":     "High heat stress risk. Water twice daily: early morning and late evening.",
        "CRITICAL": "Critical heat. Water three times daily. Shade if possible. Check for wilting.",
    },
    "hydric": {
        "LOW":      "Water-loving species. Keep soil consistently moist. Water daily.",
        "MODERATE": "Heat is elevated. Water daily, morning preferred to reduce evaporation.",
        "HIGH":     "High heat stress. Water twice daily. Consider temporary shading.",
        "CRITICAL": "Emergency: water 3x daily. These species are highly vulnerable to heat stress.",
    },
}


def _water_need_for_species(species: str, family: str, genus: str) -> str:
    """
    Look up water need class from the WATER_NEED table.
    Checks species, family, and genus in order; defaults to "mesic" if unknown.
    """
    for key, cls in WATER_NEED.items():
        if (
            key.lower() in species.lower()
            or key.lower() in family.lower()
            or key.lower() in genus.lower()
        ):
            return cls
    return "mesic"


@router.post("/identify")
async def identify_plant_endpoint(
    organ: str = Query(
        "leaf",
        description="Plant organ: leaf, flower, fruit, bark, habit, auto",
    ),
    avg_humidex: float = Query(
        30.0,
        description="Current average humidex at station (used for heat-stress advice)",
    ),
    file: UploadFile = File(..., description="Plant photo (JPG or PNG)"),
) -> dict:
    """
    Identify a plant from a photo using PlantNet API.

    Returns PlantNet identification results plus:
    - water_need_class: xeric | mesic | hydric (drives irrigation rule display)
    - heat_stress_risk: LOW | MODERATE | HIGH | CRITICAL
    - care_advice: advisory text combining species water needs and current conditions

    If PLANTNET_API_KEY is not configured, returns success=False with an
    informative message rather than raising an error.
    """
    if file.content_type not in (
        "image/jpeg",
        "image/jpg",
        "image/png",
        "image/webp",
    ):
        raise HTTPException(
            status_code=422,
            detail="File must be an image (JPG, PNG, or WebP)",
        )

    image_bytes = await file.read()
    if len(image_bytes) > 10 * 1024 * 1024:  # 10 MB limit
        raise HTTPException(status_code=413, detail="Image too large (max 10 MB)")

    result = identify_plant(image_bytes, organ=organ)

    water_need = "mesic"
    care_advice = "Unable to provide species-specific advice without identification."

    if result["success"] and result["top_result"]:
        top = result["top_result"]
        water_need = _water_need_for_species(
            top["species"],
            top.get("family", ""),
            top.get("genus", ""),
        )
        risk = humidex_to_risk_level(avg_humidex)
        care_advice = _ADVICE.get(water_need, {}).get(risk, "Monitor plant closely.")

    return {
        **result,
        "water_need_class": water_need,
        "heat_stress_risk": humidex_to_risk_level(avg_humidex),
        "care_advice": care_advice,
    }
