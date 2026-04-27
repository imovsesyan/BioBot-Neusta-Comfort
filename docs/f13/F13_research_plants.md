# F12 / F13 Research — Zones, Thermal Protection Slots, Plant Monitoring & Species Prediction

**Date**: 2026-04-27
**Project**: BioBot-Neusta-Comfort
**Status**: Final research — no production code

---

## Executive Summary

For F12 (zones + thermal protection slots) and F13 (plant species ID + health + irrigation), the recommended posture is:

1. **Plant species ID**: integrate **PlantNet API** (Pl@ntNet) as the single photo-based identifier. It is open-data, has a free tier sufficient for prototyping, exposes a simple HTTPS API consumable from Python with `requests`, and aligns with the project's open-source preference.
2. **Plant health from photos**: use a **pretrained PlantVillage CNN from Hugging Face Hub** (e.g., `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification`) for offline disease classification on leaf images. Treat output as advisory, not diagnostic.
3. **Plant health from sensors**: re-use existing humidex risk levels (F10), Aquacheck soil moisture, and Meteo France inputs — combine via a deterministic **species-aware rule table** (analogous to F11's profile×risk matrix), not a new ML model. This sidesteps the F9/F10 leakage concern flagged in CLAUDE.md.
4. **Irrigation suggestions**: a **rule-based combination of (humidex_risk_level × soil_moisture_band × species_water_need_class)** is sufficient. Defer ET₀-based agronomic models (Penman-Monteith / FAO-56) until an explicit horticultural use case exists.
5. **F12 zones**: define zones as **both spatial (sensor-anchored) and temporal (time-of-day windows)**, with thermal protection slots derived from rolling humidex risk on the existing 15-min Neusta / 1h Meteo cadence. No new model — windowed thresholds over F10 outputs are enough.

The unifying architectural principle, inherited from F11: **rules-as-data + photos-as-files**, with ML and APIs introduced only where a rule table cannot reasonably encode the answer (i.e., pixel-level species/disease recognition).

---

## Section 1 — Plant Species Identification APIs

### 1.1 Candidates Compared

#### PlantNet API (Pl@ntNet)
- **What it is**: an open-science plant identification web service backed by a CNN classifier and a community-curated reference image database. Operated by INRIA / CIRAD / INRAE / IRD (French public research consortium).
- **Endpoint**: `https://my-api.plantnet.org/v2/identify/{project}` (e.g., `all`, `weurope`, `useful`).
- **Inputs**: 1–5 images per request, plus `organs` hints (`leaf`, `flower`, `fruit`, `bark`, `habit`, `auto`).
- **Output**: ranked list of species candidates each with a `score` (confidence 0–1) and taxonomic metadata (family, genus, common names). No health/disease information.
- **Free tier**: 500 identifications/day with a registered API key.
- **Python integration**: no official SDK; standard `requests`-style HTTPS multipart upload. Trivial to wrap.
- **Offline?**: No — requires internet.
- **License posture**: API is open access for non-commercial use; commercial use requires contacting Pl@ntNet.
- **Source**: `https://my.plantnet.org/`, `https://my.plantnet.org/doc/openapi`.

#### Plant.id API (Kindwise / FlorAI)
- **What it is**: a commercial plant ID + plant health API. Two separate endpoints: `/identification` (species) and `/health_assessment` (disease, pest, deficiency).
- **Inputs**: 1–5 images (base64 or URL) + optional location/datetime.
- **Output**: ranked candidates with `probability`, plant taxonomy, edibility, watering. The health endpoint returns disease classes with probabilities and suggested treatments.
- **Free tier**: trial only (e.g., 100–500 free credits); production use is paid (~€0.01–0.10 per call).
- **Python integration**: no first-party SDK; HTTPS JSON.
- **Offline?**: No.
- **License posture**: proprietary commercial.
- **Source**: `https://web.plant.id/plant-identification-api/`.

#### iNaturalist API
- **What it is**: a community-science platform with ML-based "computer vision suggestions" trained on iNaturalist observations (all taxa, not only plants).
- **Endpoint**: `https://api.inaturalist.org/v1/computervision/score_image`.
- **Inputs**: image upload.
- **Output**: ranked taxon suggestions with `combined_score` (visual + geographic prior). No health info.
- **Free tier**: ~10k requests/day historically.
- **Python integration**: `pyinaturalist` (community library).
- **Offline?**: No.
- **Caveat**: taxonomically broad — less precise for cultivated ornamentals than PlantNet's plant-only model.
- **Source**: `https://api.inaturalist.org/v1/docs/`.

#### Google Cloud Vision API
- **What it is**: a generic object/label detection API. Returns labels like "Monstera deliciosa" but is **not** a dedicated plant taxonomy classifier.
- **Free tier**: 1,000 units/month, then $1.50 per 1,000 units.
- **Python integration**: official `google-cloud-vision` SDK.
- **Offline?**: No.
- **Caveat**: results are at label level (often genus or common name), not curated taxonomy. Not recommended as a primary plant ID source.

#### Pl@ntNet Python SDK
- **Status**: no official SDK exists. Community wrappers on PyPI have uneven maintenance.
- **Recommendation**: write a thin internal `requests` wrapper instead of depending on a community SDK.

### 1.2 Comparison Matrix

| Criteria | PlantNet | Plant.id | iNaturalist | Google Vision |
|---|---|---|---|---|
| Single-photo species ID | Yes | Yes | Yes | Partial (generic labels) |
| Returns health status | No | Yes (separate endpoint) | No | No |
| Free tier | 500/day | Trial only, then paid | ~10k/day | 1,000/month |
| Open-source / open-science | Yes (research consortium) | No | Yes (data) / No (CV) | No |
| Python SDK quality | None (use `requests`) | None (use `requests`) | `pyinaturalist` (community) | Official SDK |
| Confidence scoring | Yes (0–1 score) | Yes (probability) | Yes (combined_score) | Yes (label confidence) |
| Offline capable | No | No | No | No |
| Plant-specific taxonomy | Yes | Yes | All taxa | No |
| Best for ornamental / cultivated plants | Strong | Strong | Weaker | Weak |

### 1.3 Recommendation — Section 1

**Integrate PlantNet API as the single species-ID source.**

Justification:
- The project has an open-source preference (inherited from F11). PlantNet is the only candidate backed by a public research consortium.
- 500/day free is far above any realistic prototype load.
- HTTPS + `requests` is trivial to wrap; the absence of an official SDK is manageable.
- Confidence scores allow a fallback rule: if top score < 0.4, use `unknown_species` with a conservative default rule — directly compatible with the rules-as-data approach.
- Health assessment is deliberately separated (Section 2) — using PlantNet for ID and a Hugging Face model for health keeps each capability independently replaceable.

**Flag**: Pl@ntNet's commercial-use policy must be reviewed before any non-research deployment. Document the API key and quota in the README.

---

## Section 2 — Plant Health Assessment

### 2.1 Photo-based health (PlantVillage and successors)

- **PlantVillage dataset**: ~54,000 leaf images across 14 crops and 38 disease/healthy classes. Canonical benchmark for plant disease classification (Mohanty, Hughes & Salathé, *Frontiers in Plant Science*, 2016).
- **Pretrained models on Hugging Face Hub**:
  - `linkanjarad/mobilenet_v2_1.0_224-plant-disease-identification` — MobileNetV2 fine-tuned on PlantVillage.
  - `wambugu71/crop_leaf_diseases_vit` — Vision Transformer variant.
  - Various community CNN/ViT checkpoints under tags `plant-disease`, `plantvillage`.
- **Strengths**: free, runnable offline via `transformers` + `torch`, well-documented dataset, supports batch inference.
- **Limitations**:
  - PlantVillage is **crop-centric** (tomato, potato, apple, grape, corn, etc.). Performance on **ornamental/houseplants** is unvalidated and likely poor.
  - In-the-wild photos of whole plants degrade accuracy substantially (30–50% drop vs lab/segmented images, per published studies).
  - Disease classes are crop-specific; the system cannot diagnose generic stress (wilting, etiolation) in unseen species.
- **PlantDoc dataset** (Singh et al., 2020): in-the-wild supplement to PlantVillage (~2,500 images). Models fine-tuned on PlantDoc generalize better to user photos.
- **BioCLIP** (Imageomics Institute, 2024): zero-shot taxonomy classifier across tree-of-life — more about species than disease; worth tracking for future use.

### 2.2 Sensor-based health inference (using existing BioBot data)

The four data sources in `data/processed/` provide:
- **Aquacheck 15-min**: soil moisture (volumetric water content) — directly relevant to plant water stress.
- **Meteo France 1h**: outdoor temperature, humidity, humidex, vivabilite_score_meteo.
- **IoT 15-min**: indoor temperature/humidity/air quality.
- **Neusta 15-min**: ambient + livability features.

Inferable plant-stress indicators **without a new ML model**:

| Indicator | Derivation from existing data | Reference |
|---|---|---|
| **Drought stress risk** | `aquacheck.soil_moisture` below species-specific threshold | FAO Irrigation Paper 56 (Allen et al., 1998) |
| **Heat stress risk** | F10 `humidex_risk_level` ∈ {high_risk, dangerous, critical} sustained ≥ 2h | Wahid et al., *Env. Exp. Botany* (2007) |
| **Cold stress risk** | Temperature below species-specific minimum (5–10°C tropicals, 0°C frost) | RHS hardiness zones |
| **Humidity stress** | Indoor RH < 30% (most tropicals) or > 80% (succulents/cacti) sustained | Pennisi & van Iersel, *HortScience* (2012) |
| **Vapor Pressure Deficit (VPD)** | `VPD = SVP × (1 − RH/100)`; high VPD → transpiration stress | Grossiord et al., *New Phytologist* (2020) |

VPD is a **deterministic formula**, not a model — a scientifically defensible plant-stress proxy that complements humidex (which is human-centric). VPD is the standard horticultural variable for transpiration.

### 2.3 Combination strategy

A two-channel design:

- **Channel A — photo (event-driven, occasional)**: user uploads a photo → species ID via PlantNet → if leaf is the dominant organ, run the PlantVillage-derived disease model → output `(species_candidate, top_disease_candidate, confidence)`.
- **Channel B — sensor (continuous, automated)**: rolling window over `data/processed/` → derive species-aware stress flags from VPD, soil moisture, humidex risk → output `(stress_type, severity, since_when)`.

Recommendations key off **both channels together**, with photo optional. If only sensor data is available, the system still functions (degraded mode). Require both channels to agree before flagging high-severity alerts; allow either alone for low-severity advisories.

### 2.4 Recommendation — Section 2

- **Photos**: use a Hugging Face PlantVillage-derived model as a **secondary advisory signal only**. Surface it with explicit "training-data limited to crop species; use as guidance" caveats.
- **Sensors**: do **not** train a new health model. Compute VPD, drought-stress flags from Aquacheck, and heat-stress flags from existing humidex risk levels. Combine deterministically.

---

## Section 3 — Irrigation Suggestions

### 3.1 Rule-based approach (recommended)

A simple rule stack drawn from FAO-56 simplifications and university extension irrigation guides:

1. **Soil moisture trigger**: if `aquacheck.soil_moisture < lower_band_for_species`, recommend irrigation.
2. **Heat amplifier**: if `humidex_risk_level ∈ {high_risk, dangerous, critical}`, advance the irrigation window earlier in the day (before sun peak).
3. **Rain skip**: if Meteo France shows precipitation > 2–5 mm in next 24h, suppress the recommendation.
4. **Time-of-day filter**: never irrigate between ~10:00 and ~17:00 during heat events (evaporation loss + leaf scorch). Prefer 06:00–09:00 or after 19:00.
5. **Species water-need class**: a categorical attribute (`xeric` / `mesic` / `hydric`) gates the soil moisture threshold. Three classes are sufficient.

### 3.2 Model-based approach (defer)

The agronomic gold standard is **FAO Penman-Monteith ET₀** scaled by crop coefficient `Kc`. Penman-Monteith requires net radiation and wind speed — unavailable for a typical indoor BioBot deployment. The reduced **Hargreaves equation** (temperature only) is feasible if needed later.

Python libraries for future reference:
- `pyeto` — FAO-56 Penman-Monteith and Hargreaves. Lightweight, MIT.
- `pyfao56` — full FAO-56 reference implementation.

### 3.3 Mapping to existing F10 risk levels

| Humidex risk | Soil moisture | Suggested action |
|---|---|---|
| livable | normal | none |
| livable | dry | irrigate at next morning window |
| discomfort | normal | monitor; consider light irrigation if mesic/hydric |
| discomfort | dry | irrigate today, morning |
| high_risk | normal | check soil; mist if high VPD |
| high_risk | dry | irrigate now (early morning) + shade |
| dangerous | any | irrigate at coolest available window; move to shade if movable |
| critical | any | emergency: shade + deep soak at coolest window |

Expanded with 3-class species water need: 5 humidex × 3 moisture × 3 species = **45 base cells** × diurnal time modifier — comparable to F11's recommendation matrix, authorable manually from FAO and extension references.

### 3.4 Recommendation — Section 3

Use a **3-input rule table**: `(humidex_risk_level, soil_moisture_band, species_water_need_class) → irrigation_action`, with a time-of-day filter post-applied. Defer ET₀-based modeling until precision agronomy is explicitly required.

---

## Section 4 — F12: Zones and Thermal Protection Slots

### 4.1 Identifying favorable vs dangerous time periods

F10 already produces per-row `humidex_risk_level` at 15-min cadence (1h for Meteo). Constructing time windows is a **rolling-window classification over an existing label**, not a new modeling task:

- **Dangerous slot**: any contiguous run of ≥ 2h where `humidex_risk_level ∈ {high_risk, dangerous, critical}`.
- **Favorable slot**: any contiguous run of ≥ 2h where `humidex_risk_level == livable` AND temperature within 18–24°C AND VPD within a healthy plant band.
- **Transition slot**: the 30–60 min before a dangerous slot opens — the "do something now" actionable window.

Generated as `reports/tables/f12_uc1_zones_slots.csv` with columns `(zone_id, slot_type, start_ts, end_ts, dominant_risk_level, source_sensor)`.

### 4.2 Standard frameworks for protective time windows

- **Agronomy**: "irrigate at the cooler ends of the day" (06:00–09:00 / after 19:00) — codified in extension publications.
- **ASHRAE 55 adaptive comfort model** (de Dear & Brager, 2002): defines comfort bands as a function of running mean outdoor temperature — can drive ventilation timing recommendations.
- **Heat-Health Early-Warning Systems (HHEWS)** (Météo-France / WHO Europe): vigilance levels by sustained humidex/temperature thresholds over rolling windows — a direct analogue to thermal protection slots.
- **Chronobiology**: not directly applicable to plant or environmental protection scheduling.

The simplest defensible framework: **rolling-window classification of F10 risk levels with a minimum-duration filter**, labeled with HHEWS vigilance categories (vert / jaune / orange / rouge). Auditable, traceable to public health practice, zero new modeling debt.

### 4.3 Spatial vs temporal zones

**Spatial zones — sensor-anchored**

Each IoT / Aquacheck / Neusta sensor occupies a physical location. Since the four sources are not spatially merged (CLAUDE.md constraint), zones are defined **per source**, with a small manual `zones.yaml` config mapping sensor IDs to location labels:

```yaml
zone_id: living_room_north
sources:
  - { source: iot, sensor_id: ABC123 }
  - { source: aquacheck, sensor_id: XYZ789 }
location_kind: indoor | outdoor | mixed
exposure: north | south | east | west | none
```

**Temporal zones — time-of-day windows**

Canonical diurnal windows: `night 22–06`, `morning 06–10`, `midday 10–15`, `afternoon 15–19`, `evening 19–22`. Aggregate F10 risk distribution per window. These windows align with HHEWS communications and irrigation timing best practice.

**Combined**: per `(spatial_zone, temporal_window, date)` cell, summarize humidex risk, mean VPD, mean soil moisture. Produces a small daily zone-status table, ideal for visualization or for driving the F11 recommendation engine.

### 4.4 Recommendation — Section 4

- **Define zones as `(spatial_zone, temporal_window)` 2D index**, anchored to existing sensor IDs and a small `zones.yaml` mapping. Do not attempt to spatially merge the four data sources.
- **Generate thermal protection slots** as rolling-window aggregations of F10 humidex risk levels, with 2-hour minimum duration and explicit `transition_slot` markers 30–60 min before high-risk slots open.
- **Re-use HHEWS vert/jaune/orange/rouge vocabulary** for slot vigilance labels.
- **No new model**. F12 is a windowing and labeling layer over F10 outputs.

---

## Section 5 — Integration Complexity Assessment

### 5.1 Architecture

The existing pipeline is script-driven: scripts orchestrate, package modules host logic, no microservices. F12 and F13 follow the same pattern.

### 5.2 New `src/biobot/` subpackages

```
src/biobot/
├── plants/
│   ├── __init__.py
│   ├── species_identifier.py    # thin PlantNet API wrapper (HTTP + caching)
│   ├── disease_classifier.py    # HF transformers wrapper (offline)
│   ├── species_catalog.py       # species → water_need_class, hardiness, etc.
│   └── health_rules.py          # sensor-channel stress rules (VPD, drought, heat)
├── irrigation/
│   ├── __init__.py
│   └── rules.py                 # (humidex × moisture × species) → action
└── zones/
    ├── __init__.py
    ├── spatial.py               # zones.yaml loader + sensor→zone mapping
    ├── temporal.py              # diurnal window definitions + slot extraction
    └── protection_slots.py      # rolling-window slot generator over F10 outputs
```

Existing F11 `recommendations/rules_recommender.py` should be **extended, not duplicated** — plant care recommendations are the same shape as comfort recommendations (rule lookup over profile × risk). The "profile" for a plant is `(species, water_need_class, location_zone)`.

### 5.3 Photo storage layout

```
data/
└── plants/
    ├── photos/
    │   ├── raw/                         # original uploads (gitignored)
    │   │   └── 2026/04/27/<uuid>.jpg
    │   ├── processed/                   # resized/normalized for ML (gitignored)
    │   └── examples/                    # small committed demo set (≤10 photos, ≤5 MB)
    ├── plant_inventory.csv              # plant_id, zone_id, species, planted_at, notes
    └── identification_log.csv           # plant_id, photo_path, top_species, score, ts
```

- Raw photos are gitignored (privacy + repo size).
- A small curated `examples/` set can be committed for tests and demos.
- Photos should be downsized to ≤ 1024 px on the long edge before API calls (PlantNet recommendation; reduces upload time).

### 5.4 Microservice vs script

**Recommendation: integrate as scripts + package modules. Do not introduce a microservice.**

Justification:
- Existing pipeline is script-driven, not service-driven. A microservice would fork the architecture.
- PlantNet calls are infrequent (event-driven, user-initiated) — synchronous script invocation is sufficient.
- The disease classifier runs offline via `transformers` — no service boundary needed.
- If the project later moves toward an interactive frontend, the `species_identifier.py` and `disease_classifier.py` modules can be promoted to a thin FastAPI service without rewriting core logic.

### 5.5 Pipeline integration

```
data/processed/*.csv.gz           data/plants/photos/raw/*.jpg
        │                                   │
        │                    ┌──────────────┴──────────────┐
        │                    ▼                             ▼
        │            species_identifier            disease_classifier
        │              (PlantNet API)              (HF offline model)
        │                    │                             │
        │                    └──────────────┬──────────────┘
        │                                   ▼
        │                     data/plants/identification_log.csv
        │                                   │
        ▼                                   │
zones/protection_slots ◄──── F10 humidex    │
        │                                   │
        ▼                                   │
reports/tables/f12_uc1_zones_slots.csv      │
                                            │
        ┌───────────────────────────────────┘
        ▼
irrigation/rules + plants/health_rules
        │
        ▼
reports/tables/f13_uc1_irrigation_advice.csv
reports/tables/f13_uc1_plant_health_advice.csv
```

### 5.6 New dependencies

- `requests` — for PlantNet API calls.
- `Pillow` — for photo resizing/normalization.
- `transformers` + `torch` — for disease classifier (already in `requirements-advanced.txt`).
- `pyyaml` — for `zones.yaml`.
- Optional: `pyeto` if ET₀ irrigation modeling is introduced later.

### 5.7 Compliance / risk flags

- **GDPR**: photos may capture people or identifiable interiors. Raw photo store must be gitignored; any deployed component handling user uploads needs GDPR consent + retention policy.
- **PlantNet ToS**: research-only API use is free; commercial deployment requires explicit licensing.
- **Non-diagnostic disclaimer**: disease classifier output must be presented as advisory only, especially given PlantVillage training-data limitations on ornamentals.

---

## Prioritized Build Plan

1. **F12-UC1 — Zone definition (spatial + temporal index)**. Build first — it is purely a structural layer over already-computed F10 outputs and a small manual `zones.yaml`. It unblocks every downstream feature (protection slots, irrigation advice, plant health rules) by providing a stable join key. Zero new model risk.

2. **F12-UC2 — Thermal protection slots (rolling-window labeling over F10)**. Second — a thin computation on top of zones plus existing humidex risk levels. Adds no model, inherits F10's auditability. The most defensible "new feature" deliverable.

3. **Irrigation suggestions — rule table on `(humidex_risk × soil_moisture × species_water_need)`**. Third — runs on existing sensor data **without** any photo capability. Highest-value sensor-only deliverable; proves the F13 architecture before adding API and ML dependencies.

4. **F13-UC1 — Plant species identification via PlantNet API**. Fourth — introduces the first external network dependency and the first photo storage path. Doing it after the rule layers means a failed or rate-limited API call still leaves a working sensor-driven system.

5. **F13-UC1 — Plant health from photo (Hugging Face PlantVillage-derived model)**. Last — the most experimental component (training-data limited to crops, in-the-wild accuracy degraded). Best layered on once the deterministic foundations (zones, slots, irrigation rules, species ID) are stable. Treat as an advisory enhancement, not a load-bearing capability.

Each step delivers standalone value. The order minimizes the blast radius of the riskiest components (external API + ML model) by deferring them until the deterministic backbone is in place — directly mirroring F11's "rules first, ML/LLM later" sequencing.

---

## Sources

- Joly, A. et al. *Pl@ntNet: A citizen science platform to learn from plant images*. `https://my.plantnet.org/doc/openapi`
- Mohanty, S. P., Hughes, D. P., & Salathé, M. (2016). *Using deep learning for image-based plant disease detection*. Frontiers in Plant Science.
- Singh, D. et al. (2020). *PlantDoc: A dataset for visual plant disease detection*. CoDS-COMAD.
- Hugging Face Hub plant-disease models: `https://huggingface.co/models?search=plant+disease`
- Allen, R. G. et al. (1998). *Crop evapotranspiration*. FAO Irrigation and Drainage Paper 56.
- Wahid, A. et al. (2007). *Heat tolerance in plants: An overview*. Environmental and Experimental Botany.
- Grossiord, C. et al. (2020). *Plant responses to rising vapor pressure deficit*. New Phytologist.
- Pennisi, S. V. & van Iersel, M. W. (2012). *Quantified light integral and growth characteristics of foliage plants*. HortScience.
- de Dear, R. & Brager, G. S. (2002). *Thermal comfort in naturally ventilated buildings*. Energy and Buildings.
- Plant.id API: `https://web.plant.id/plant-identification-api/`
- iNaturalist API: `https://api.inaturalist.org/v1/docs/`
- Google Cloud Vision API: `https://cloud.google.com/vision/docs`
- Météo-France Vigilance Canicule: `https://vigilance.meteofrance.fr/`
- WHO Europe HHEWS: `https://www.who.int/europe/`
- BioCLIP (Imageomics Institute, 2024): `https://imageomics.github.io/bioclip/`
- `pyeto` (FAO-56): `https://github.com/Evapotranspiration/ETo`
