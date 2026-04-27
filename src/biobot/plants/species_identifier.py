"""F13-UC1 plant species identification via the PlantNet API.

Thin wrapper around the Pl@ntNet v2 HTTPS API
(https://my-api.plantnet.org/v2/identify/all).  Only the top ranked candidate
is returned.  The module supports a ``mock`` mode for tests and CI — no HTTP
call is issued in that path.

Scientific note
---------------
PlantNet scores are softmax-style confidences (0–1) produced by a CNN trained
on the Pl@ntNet reference image dataset (INRIA/CIRAD/INRAE/IRD consortium).
A score below 0.4 is insufficient for reliable species-level identification;
the caller should fall back to a genus-level or conservative default rule
(research recommendation: F13_research_plants.md, Section 1.3).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

PLANTNET_API_URL = "https://my-api.plantnet.org/v2/identify/all"
LOW_CONFIDENCE_THRESHOLD = 0.4

_MOCK_RESULT_SPECIES = "Ficus lyrata"
_MOCK_RESULT_COMMON = "Fiddle-leaf fig"
_MOCK_RESULT_FAMILY = "Moraceae"
_MOCK_RESULT_CONFIDENCE = 0.82


@dataclass
class SpeciesResult:
    """Identification result for a single plant image."""

    species_name: str
    common_name: str
    confidence: float
    family: str
    is_low_confidence: bool = field(init=False)

    def __post_init__(self) -> None:
        self.is_low_confidence = self.confidence < LOW_CONFIDENCE_THRESHOLD


def _mock_result() -> SpeciesResult:
    """Return a fixed, deterministic result used in tests and CI."""
    return SpeciesResult(
        species_name=_MOCK_RESULT_SPECIES,
        common_name=_MOCK_RESULT_COMMON,
        confidence=_MOCK_RESULT_CONFIDENCE,
        family=_MOCK_RESULT_FAMILY,
    )


def _empty_result() -> SpeciesResult:
    """Return a sentinel result when the API returns no candidates."""
    return SpeciesResult(
        species_name="unknown",
        common_name="unknown",
        confidence=0.0,
        family="unknown",
    )


def _call_plantnet_api(image_path: str, api_key: str) -> SpeciesResult:
    """Issue the multipart upload to PlantNet and parse the top candidate.

    The ``organs`` hint is set to ``["auto"]`` so PlantNet selects the
    appropriate organ classifier without requiring the caller to know whether
    the image shows a leaf, flower, or whole habit.
    """
    try:
        import requests
    except ImportError as exc:
        raise ImportError(
            "The 'requests' package is required for live PlantNet API calls. "
            "Install it with: pip install requests"
        ) from exc

    with open(image_path, "rb") as image_file:
        files = [("images", (os.path.basename(image_path), image_file, "image/jpeg"))]
        params = {"api-key": api_key}
        data = {"organs": ["auto"]}

        try:
            response = requests.post(
                PLANTNET_API_URL,
                params=params,
                files=files,
                data=data,
                timeout=10,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"PlantNet API request failed: {exc}") from exc

    if response.status_code != 200:
        raise RuntimeError(f"PlantNet API returned HTTP {response.status_code}: {response.text}")

    payload = response.json()
    results = payload.get("results", [])

    if not results:
        return _empty_result()

    top = results[0]
    species = top.get("species", {})
    species_name = species.get("scientificNameWithoutAuthor", "unknown")
    common_names = species.get("commonNames", [])
    common_name = common_names[0] if common_names else "unknown"
    family = species.get("family", {}).get("scientificNameWithoutAuthor", "unknown")
    confidence = float(top.get("score", 0.0))

    return SpeciesResult(
        species_name=species_name,
        common_name=common_name,
        confidence=confidence,
        family=family,
    )


def identify_species(
    image_path: str,
    api_key: str | None = None,
    mock: bool = False,
) -> SpeciesResult:
    """Identify the plant species in an image.

    Parameters
    ----------
    image_path:
        Path to the plant photo (JPEG or PNG).  Ignored in mock mode.
    api_key:
        PlantNet API key.  Falls back to the ``PLANTNET_API_KEY`` environment
        variable if not supplied.  Ignored in mock mode.
    mock:
        When ``True``, return a deterministic fixed result without any HTTP
        call.  Intended for tests and CI pipelines.

    Returns
    -------
    SpeciesResult
        Top species candidate.  ``is_low_confidence`` is set automatically
        when ``confidence < 0.4``.

    Raises
    ------
    ValueError
        If ``mock=False`` and no API key can be resolved.
    RuntimeError
        On HTTP error (contains the status code) or network failure.
    """
    if mock:
        return _mock_result()

    resolved_key = api_key or os.environ.get("PLANTNET_API_KEY")
    if not resolved_key:
        raise ValueError(
            "A PlantNet API key is required for live identification. "
            "Pass it as ``api_key`` or set the PLANTNET_API_KEY environment variable."
        )

    return _call_plantnet_api(image_path, resolved_key)
