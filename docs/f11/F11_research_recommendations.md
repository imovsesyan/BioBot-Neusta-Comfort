# F11-UC1 / F11-UC5 Research Report — Personalized Comfort Recommendation System

**Date**: 2026-04-27
**Domain**: Comfort & Wellness Tech (cross-domain with Bot Frameworks for delivery layer)
**Requested By**: BioBot-Neusta-Comfort project (F11 personalization phase)
**Status**: Final (research-only; no production code)

---

## Executive Summary

For a thermal-comfort recommendation system built on top of F9/F10 outputs with **no real users**, **~4,315 rows**, and a **rule-derived target leakage risk** in the underlying scores, the recommended design is:

1. **Profile**: a small categorical profile — `activity_level` (3 bins mapped to MET ranges), `clothing_level` (3 bins mapped to CLO bands), and `vulnerability_flag` (healthy adult / elderly / child / chronic-condition). This is richer than binary but still tractable for synthetic generation.
2. **Strategy**: a **rule-based recommendation engine** as the primary system, structured as a `(humidex_risk_level × profile_bucket) → action_set` lookup, optionally augmented with an **LLM rephrasing layer** for natural-language delivery. Lightweight ML and content-based filtering should be deferred until real interaction data exists.
3. **Scientific guardrail**: do **not** train any new model on `vivabilite_binary_mean` or on F10-UC4 outputs — these inherit the formula-derived leakage already flagged in F9. Use humidex thresholds and raw environmental features as recommendation inputs, not the rule-derived labels themselves, to avoid stacking rule outputs on rule outputs.

The bulk of the value at this stage is in **traceable, auditable rules**. ML and LLM layers are explicit *future* additions, not the foundation.

---

## Research Question

What is the right architecture for F11-UC1 (personalized comfort recommendations) and F11-UC5 (adapting recommendations to user profiles), given:

- No real users, only synthetic profiles.
- ~4,315 rows of environmental data.
- Existing F9 livability scores with target-leakage risk (R² ≈ 0.997).
- Existing F10 humidex risk classifications derived from rule thresholds (livable / discomfort / high_risk / dangerous / critical).
- Four upstream sources (IoT, Aquacheck, Neusta, Meteo France) that are NOT merged.

---

## 1. Profile Design Findings

### 1.1 Activity encoding: binary vs MET

**Binary (active=1, sedentary=0)**
- Pros: trivially fits ~4,315-row dataset; easy to explain to stakeholders; low risk of overfitting; aligns with simple synthetic-profile generation.
- Cons: collapses important physiological variation. Sleeping, sitting, walking, and cycling are all relevant for thermal comfort but cannot be distinguished. Loses the ability to compute personalized PMV/PPD-style indices.

**MET (Metabolic Equivalent of Task) scale**
- The MET scale is the established standard for metabolic heat production in thermal comfort research (ASHRAE 55, ISO 7730). 1 MET ≈ 58.2 W/m² of body surface — a seated, resting adult. Reference values:
  - 0.7–0.9 MET: sleeping
  - 1.0–1.2 MET: seated, quiet
  - 1.4–1.6 MET: standing, relaxed
  - 1.8–2.0 MET: walking slowly
  - 2.5–3.0 MET: light housework
  - 3.0–4.0+ MET: active walking, manual work
- Pros: scientifically standard; required input for any PMV/PPD or adaptive-comfort calculation; supports continuous personalization.
- Cons: continuous MET values introduce spurious precision when no real user is providing them. With only synthetic profiles, the dataset cannot meaningfully justify decimal MET resolution.

**Recommendation**: a **3-bin categorical encoding** mapped to MET ranges:
- `sedentary` → MET ≈ 1.0 (resting / desk work)
- `light` → MET ≈ 1.5 (standing / light movement)
- `active` → MET ≈ 2.5+ (walking / housework / outdoor activity)

This preserves enough physiological signal to differentiate rules ("active people in `discomfort` already need cooling, sedentary do not yet") without inventing precision the synthetic profiles cannot support.

### 1.2 Clothing encoding: binary vs CLO

**Binary (light=0, heavy=1)**
- Pros: easy to combine with binary activity; compact rule tables.
- Cons: cannot represent the seasonal continuum (summer dress ≠ T-shirt + jeans ≠ business attire).

**CLO index (Clothing Insulation Unit)**
- 1 CLO = 0.155 m²·K/W ≈ a typical business suit. Standard reference values (ASHRAE 55, ISO 9920):
  - 0.3 CLO: shorts + T-shirt (summer minimum)
  - 0.5 CLO: light summer clothing (short sleeves + light trousers)
  - 0.7 CLO: typical indoor casual
  - 1.0 CLO: business suit / winter indoor
  - 1.5+ CLO: heavy winter outdoor
- Pros: scientifically standard; required for PMV; differentiates summer vs winter rules.
- Cons: same precision concern as MET — continuous values are not meaningful from synthetic profiles.

**Recommendation**: **3-bin categorical encoding** mapped to CLO bands:
- `light` → CLO ≈ 0.4 (summer)
- `medium` → CLO ≈ 0.7 (transitional)
- `heavy` → CLO ≈ 1.0+ (winter)

### 1.3 Age group / vulnerability flag

Strongly recommend including this. Heat-related health outcomes are dominated by vulnerability — WHO and CDC heat-health guidance, the EuroHEAT project, and the French `Plan Canicule` all stratify advice by vulnerability category. Heat illness incidence in the elderly (65+) is roughly 3–5× that of healthy adults at the same humidex; infants and small children have reduced thermoregulation capacity and dehydrate faster.

**Recommendation**: a `vulnerability` categorical field with values:
- `healthy_adult` (default)
- `elderly` (65+)
- `child` (under ~12)
- `chronic_condition` (cardiovascular, respiratory, diabetes, pregnancy)

This drives meaningful recommendation differences (e.g., elderly should act at humidex 30, healthy adults at 35–40), which is exactly the personalization the use case asks for.

### 1.4 Minimal viable profile

| Field | Type | Values |
|---|---|---|
| `activity_level` | categorical | `sedentary`, `light`, `active` |
| `clothing_level` | categorical | `light`, `medium`, `heavy` |
| `vulnerability` | categorical | `healthy_adult`, `elderly`, `child`, `chronic_condition` |

Total combinations: 3 × 3 × 4 = **36 profile buckets**. Crossed with 5 humidex risk levels → 180 cells in a recommendation matrix. This is small enough to author manually with domain references, large enough to demonstrate meaningful personalization.

---

## 2. Recommendation Strategy Comparison

### Option A — Rule-based recommendations per (risk × profile) combo

- **What it is**: a deterministic lookup table mapping `(humidex_risk_level, activity_level, clothing_level, vulnerability)` → an ordered list of action items (hydrate, reduce activity, ventilate, seek cooler space, contact emergency, etc.).
- **Pros**: fully traceable, auditable, no ML required, no training data needed, aligns with how WHO / EuroHEAT / CDC heat-health guidance is structured. Perfectly suited to the "no real users" constraint. Easy to localize into French.
- **Cons**: maintenance burden grows with profile dimensions; rigid; cannot learn from feedback (until feedback exists).
- **Best for**: the current project state. Use this as the foundation.
- **Source**: WHO Public Health Advice on Preventing Health Effects of Heat (2011, updated 2018); Santé publique France `Plan Canicule`; ASHRAE 55 (2023) guidance tables.

### Option B — Content-based filtering using profile similarity

- **What it is**: define a small library of "advice items" with feature vectors (e.g., `requires_movement`, `cooling_intensity`, `time_to_act`); for each user profile, recommend the items closest in cosine distance to the profile's needs vector.
- **Pros**: more flexible than rules; can scale if the advice library grows.
- **Cons**: with no real users, the similarity metric is being designed against synthetic data — it adds complexity without adding signal. Effectively reduces to a less-readable version of the rule table.
- **Best for**: deferred (post-real-users phase).
- **Source**: Aggarwal, "Recommender Systems: The Textbook" (Springer, 2016), ch. 4 on content-based methods.

### Option C — Lightweight ML (decision tree / logistic regression)

- **What it is**: train a small classifier on synthetic `(profile + environmental features) → label` data, where the label is hand-authored "right action".
- **Pros**: in principle generalizes; explainable if a tree is shallow.
- **Cons**: critical issue — the *labels* would be hand-authored from rules, so the ML would just be approximating a rule table with extra steps. Risk of fitting noise in synthetic profile generation. Adds a model artifact that obscures what a rule table makes explicit.
- **Best for**: not now. Only meaningful once real labels (e.g., user feedback "this advice helped / did not help") exist.
- **Source**: Cold-start recommender literature consistently warns that ML over synthetic-only ground truth is at best lossy compression of the synthesis logic. See e.g., Schein et al., "Methods and Metrics for Cold-Start Recommendations" (SIGIR 2002).

### Option D — LLM-generated recommendations from profile + risk

- **What it is**: feed `(profile, current humidex risk, environmental snapshot, locale)` into an LLM with a constrained prompt; LLM outputs natural-language advice.
- **Pros**: produces fluent, contextual French/English text; handles long-tail combinations; great for delivery via chat/voice.
- **Cons**:
  - Lack of determinism — same input can yield different advice across runs, which is problematic for any health-adjacent system.
  - Hallucination risk: an LLM may invent thresholds or suggest unsafe actions (e.g., "drink salt water").
  - Compliance: medical-adjacent advice in the EU may trigger MDR (Medical Device Regulation) Class I obligations if the system is presented as health guidance. Even for a research prototype, this should be flagged.
  - Cost / latency for a free-running LLM call per recommendation.
- **Best for**: a **rephrasing layer** on top of rule outputs, not a primary recommender. Have the rule engine produce a structured action list, then the LLM phrases it.
- **Source**: Singhal et al., "Large language models encode clinical knowledge" (Nature, 2023) and follow-ups document the same hallucination/specificity concerns for any health-adjacent LLM output.

### Comparison matrix

| Criteria | A. Rules | B. Content-based | C. Lightweight ML | D. LLM |
|---|---|---|---|---|
| Works with no users | Yes | Partial (synthetic only) | Partial | Yes |
| Works with ~4,315 rows | Yes | Yes | Marginal | Yes |
| Auditable / explainable | Excellent | Good | Good (tree) / Fair (LR) | Poor |
| Handles target-leakage risk in F9/F10 | Best (decoupled) | Medium | Worst (inherits) | Medium |
| Determinism | Excellent | Excellent | Excellent | Poor without temp=0 + caching |
| Compliance posture | Easiest to defend | Easiest | Defensible | Riskiest (medical-adjacent) |
| Time to first useful output | Hours | Days | Days | Hours |
| Fit for current project | **High** | Low | Low (now) | Medium (rephrasing only) |

---

## 3. Personalization Without Users — Cold-Start Strategies

The cold-start problem in recommender systems is well-studied. Approaches relevant here:

### 3.1 Knowledge-based recommenders
When user history does not exist, the literature (Aggarwal 2016, ch. 5; Burke 2000) recommends *knowledge-based* systems built from explicit domain models. Heat-health guidance (WHO, CDC, Santé publique France) is exactly such a domain model. This maps directly onto Option A above.

### 3.2 Persona / archetype generation
Common in HCI and digital-health research (e.g., the EU MyHealthMyData project, several DTx prototypes). Construct a small set of *archetypal* synthetic profiles representing target users:
- "Sedentary office worker, healthy adult"
- "Elderly resident with cardiovascular condition"
- "Active outdoor worker"
- "Young child"
- "Pregnant adult"

For each archetype, instantiate the categorical fields above. ~5–10 archetypes is typical and far more useful than randomly sampled synthetic profiles for stakeholder demos and validation.

### 3.3 Stratified synthetic profile sampling
For broader coverage, sample profiles from a stratified distribution informed by demographic priors (e.g., INSEE for France: roughly 20% of the population is 65+, 18% under 15, etc.). This produces a synthetic user pool whose distribution matches the target deployment region, useful for evaluating recommendation coverage.

### 3.4 Bootstrap from existing health-adjacent datasets
Public datasets that can inform synthetic profile distributions:
- **NHANES** (National Health and Nutrition Examination Survey, US): age × activity × chronic-condition crosstabs.
- **SHARE** (Survey of Health, Ageing and Retirement in Europe): elderly health states across EU.
- **EuroHEAT** project deliverables: heat vulnerability frameworks.

These should *inform priors*, not be merged into the BioBot dataset (which would create yet another scientific-merge concern).

### 3.5 A/B-testable rule variants
Even without real users, design the rule table such that future A/B testing is cheap: keep `(profile, risk) → action` as data, not code, so variant rule sets can be swapped.

---

## 4. Scientific Constraints Audit

### 4.1 Is it safe to use F10 humidex risk levels as recommendation inputs?

**Short answer**: yes, but with a caveat.

The F10 humidex risk levels (`livable`, `discomfort`, `high_risk`, `dangerous`, `critical`) come from **fixed thresholds on the humidex value itself**, not from a learned model. Humidex is a deterministic formula of temperature and humidity (Masterton & Richardson, 1979). Thresholds are expert-derived public-health bands (Environment Canada, Santé publique France). Using these as recommendation inputs is therefore **scientifically defensible** — the recommendation is keyed off humidex, with the risk-level label being a human-readable bucket.

The F10-UC4 *classifier* is a different matter. F10-UC4 trains a model to predict the rule-derived label from features that include the very variables used to compute humidex. This makes F10-UC4 a tautology (R² will be artificially high for the same reason F9 is suspect). **Do not chain recommendations off the F10-UC4 model output.** Use the rule labels directly from the humidex value.

### 4.2 Does building on `vivabilite_binary_mean` introduce leakage risk?

**Yes — and the leakage is already flagged in CLAUDE.md**. The R² ≈ 0.997 in F9 strongly suggests `vivabilite_binary_mean` is itself a deterministic function of the same environmental variables (likely a humidex/temperature/humidity threshold encoded by the data provider). Training a recommender on top of `vivabilite_binary_mean` would either:

1. Silently re-derive humidex thresholds (no new information), or
2. Inherit any biases or formula errors in the original derivation.

**Recommendation**: do **not** use `vivabilite_binary_mean` as a recommendation driver. Use the underlying environmental signals (temperature, humidity, humidex) and the public-health humidex thresholds. If `vivabilite_binary_mean` later turns out to be an independently labeled signal (pending confirmation with the project owner per CLAUDE.md), this stance can be revisited.

### 4.3 Other scientific risks from using F9/F10 as recommendation drivers

1. **Stacking rule outputs**: F10-UC4 already classifies rule labels. Building F11 on F10-UC4 predictions would be a third layer of rule-on-rule — recommendations would no longer be traceable to physical measurements.
2. **Unmerged data sources**: per CLAUDE.md, IoT, Aquacheck, Neusta, and Meteo France are not spatially/temporally aligned. Any recommendation that cites "the temperature outside" must be explicit about which source it comes from. Mixing unaligned sources in a single recommendation is a silent correctness hazard.
3. **Synthetic-profile generalization**: any evaluation of F11 on synthetic profiles is *not* evidence the system works on real users. Reports should use language like "rule coverage demonstrated across 36 synthetic profile buckets" rather than "validated on users".
4. **Medical-adjacent compliance**: any output that resembles medical advice (especially for `chronic_condition` or `elderly` vulnerability) should carry an explicit non-medical-advice disclaimer. If the project ever moves toward CE marking or clinical use, MDR Class I (or higher) obligations apply. This is project-owner / legal territory, but the system architecture should make it easy to swap or gate the medical-adjacent rules later.
5. **Dataset size for evaluation**: ~4,315 rows is small enough that any per-bucket evaluation will have wide confidence intervals. F11 evaluation should be qualitative (do the rules make sense?) rather than statistical.

---

## 5. RECOMMENDATION

### Profile encoding

Use a **3-field categorical profile**:

```
activity_level   ∈ {sedentary, light, active}        # mapped internally to MET ranges
clothing_level   ∈ {light, medium, heavy}            # mapped internally to CLO bands
vulnerability    ∈ {healthy_adult, elderly, child, chronic_condition}
```

**Justification**: this is the smallest profile that (a) preserves the variables that scientifically drive thermal comfort (metabolism, insulation, vulnerability), (b) avoids inventing precision the synthetic profiles cannot support, and (c) yields a manageable 36-bucket cross with humidex risk levels. Continuous MET / CLO encodings should be deferred until real wearable or self-report inputs exist; binary encodings discard too much signal to support meaningful personalization.

### Recommendation strategy

**Primary**: a **rule-based recommendation engine** structured as a `(humidex_risk_level, activity_level, clothing_level, vulnerability) → ordered_action_list` lookup, authored from WHO / Santé publique France / ASHRAE 55 references.

**Secondary (optional)**: an **LLM rephrasing layer** that takes the rule-engine's structured action list and renders it as natural-language advice in French/English. The LLM does not generate the *content* of the advice — only its phrasing. Temperature 0, prompt-cached, with a hard deny-list for medical-adjacent inventions.

**Defer**: content-based filtering and lightweight ML until real user feedback exists. Without real labels, both approaches are lossy compressions of the rule table.

**Avoid**: training any new F11 model on `vivabilite_binary_mean` or on F10-UC4 outputs. Drive recommendations from humidex thresholds and raw environmental measurements only.

### Justification summary

- **Matches data reality**: no real users → rule-based + synthetic profiles is the standard cold-start answer in recommender literature.
- **Sidesteps F9/F10 leakage**: keying off humidex thresholds (deterministic public-health bands) instead of the leaked `vivabilite_binary_mean` keeps F11 scientifically defensible.
- **Compliance-friendly**: deterministic, auditable rules are far easier to defend than ML/LLM outputs in a medical-adjacent context.
- **Cheap to iterate**: rules-as-data make A/B testing trivial when real users arrive.
- **Future-proof**: the profile schema, action library, and rule table become the labeled training data for a future ML system once real interaction data exists.

---

## Next Steps (for the engineering team — no code in this report)

1. **Author the action library**: a flat list of advice items (e.g., `hydrate_500ml`, `move_to_shade`, `reduce_activity`, `seek_cooled_space`, `contact_emergency_15`), each with a French and English label. Source from WHO / Santé publique France / CDC heat-health pages.
2. **Author the rule table**: 5 humidex levels × 36 profile buckets = 180 cells, each pointing to an ordered subset of the action library. Use elderly + critical humidex as the most aggressive cell, healthy-adult + livable as the no-action cell.
3. **Author 5–10 archetypal personas** for stakeholder demos.
4. **Decide on compliance positioning**: agree internally whether this is "informational, non-medical" or aspirationally regulated, and add a corresponding disclaimer.
5. **Plan evaluation**: qualitative review of the rule table against published guidance + spot-check generated advice for ~10 archetypes × 5 risk levels.
6. **Confirm `vivabilite_binary_mean` semantics with the data provider** (already an open question in CLAUDE.md). The answer determines whether F9/F10 outputs can be used more directly in a later F11 iteration.

---

## Sources

- WHO. *Public health advice on preventing health effects of heat — New and updated information for different audiences*. World Health Organization Regional Office for Europe, 2011 (2018 update). https://www.who.int/europe/publications/i/item/9789289002059
- Santé publique France. *Plan national canicule*. https://www.santepubliquefrance.fr/
- ASHRAE Standard 55-2023. *Thermal Environmental Conditions for Human Occupancy*. https://www.ashrae.org/technical-resources/standards-and-guidelines
- ISO 7730:2005. *Ergonomics of the thermal environment — Analytical determination and interpretation of thermal comfort using calculation of the PMV and PPD indices*. https://www.iso.org/standard/39155.html
- ISO 9920:2007. *Ergonomics of the thermal environment — Estimation of thermal insulation and water vapour resistance of a clothing ensemble*.
- Masterton, J. M., & Richardson, F. A. (1979). *Humidex: A method of quantifying human discomfort due to excessive heat and humidity*. Environment Canada.
- Aggarwal, C. C. (2016). *Recommender Systems: The Textbook*. Springer. Chapters 4 (content-based) and 5 (knowledge-based).
- Burke, R. (2000). *Knowledge-based recommender systems*. Encyclopedia of Library and Information Systems.
- Schein, A. I., Popescul, A., Ungar, L. H., & Pennock, D. M. (2002). *Methods and Metrics for Cold-Start Recommendations*. SIGIR.
- Singhal et al. (2023). *Large language models encode clinical knowledge*. Nature.
- EuroHEAT project deliverables. WHO Europe.
- CDC Heat & Health resources. https://www.cdc.gov/heat-health/
- INSEE demographic statistics (France). https://www.insee.fr/
- European Medical Device Regulation (MDR) 2017/745. EUR-Lex.
