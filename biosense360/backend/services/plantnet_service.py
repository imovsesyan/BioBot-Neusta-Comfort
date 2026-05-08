"""
PlantNet API wrapper for BioSense360 plant species identification.

API: https://my-api.plantnet.org/v2/identify/all
Free tier: 500 identifications/day with a registered API key.
Docs: https://my.plantnet.org/doc/openapi

If PLANTNET_API_KEY is not set, returns a mock response with a clear message
so the photo upload UI remains functional without crashing.
"""

import os

import httpx

PLANTNET_API_KEY = os.getenv("PLANTNET_API_KEY", "")
PLANTNET_URL = "https://my-api.plantnet.org/v2/identify/all"


def identify_plant(image_bytes: bytes, organ: str = "leaf") -> dict:
    """
    Identify a plant from an image using the PlantNet REST API.

    Uses httpx (already a project dependency) for HTTP rather than requests.

    Args:
        image_bytes: Raw bytes of the plant photo (JPG / PNG / WebP).
        organ:       Plant organ submitted — "leaf", "flower", "fruit",
                     "bark", "habit", or "auto".

    Returns:
        dict with keys:
            success      (bool)
            top_result   (dict | None) — species, common_names, score, family, genus
            all_results  (list[dict])  — top 5 candidates with species + score
            message      (str)         — human-readable status / error
    """
    if not PLANTNET_API_KEY:
        return {
            "success": False,
            "top_result": None,
            "all_results": [],
            "message": "PLANTNET_API_KEY not configured. Add it to your .env file.",
        }

    try:
        files = {"images": ("plant.jpg", image_bytes, "image/jpeg")}
        data = {"organs": organ}
        params = {"api-key": PLANTNET_API_KEY}

        with httpx.Client(timeout=15.0) as client:
            resp = client.post(
                PLANTNET_URL,
                files=files,
                data=data,
                params=params,
            )
        resp.raise_for_status()
        body = resp.json()

        results = body.get("results", [])
        if not results:
            return {
                "success": False,
                "top_result": None,
                "all_results": [],
                "message": "No species identified.",
            }

        top = results[0]
        species_obj = top.get("species", {})

        return {
            "success": True,
            "top_result": {
                "species": species_obj.get("scientificNameWithoutAuthor", "Unknown"),
                "common_names": species_obj.get("commonNames", []),
                "score": round(top.get("score", 0.0), 4),
                "family": species_obj.get("family", {}).get("scientificNameWithoutAuthor", ""),
                "genus": species_obj.get("genus", {}).get("scientificNameWithoutAuthor", ""),
            },
            "all_results": [
                {
                    "species": r.get("species", {}).get("scientificNameWithoutAuthor", ""),
                    "score": round(r.get("score", 0.0), 4),
                }
                for r in results[:5]
            ],
            "message": "Identification successful.",
        }
    except httpx.HTTPStatusError as e:
        return {
            "success": False,
            "top_result": None,
            "all_results": [],
            "message": f"PlantNet API error: {e}",
        }
    except Exception as e:
        return {
            "success": False,
            "top_result": None,
            "all_results": [],
            "message": f"Error: {e}",
        }
