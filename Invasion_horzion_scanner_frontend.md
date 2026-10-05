# Invasion horizon scanner

Which species not yet recorded in a country are most likely to become invasive there?

Simulated demo data. Species names and values are made up.

Country

List length (1 to 35,000)

**–**Species in list

**–**Mean invasion risk

**–**Mean uncertainty

**–**Assessed as Major or Massive

Taxonomic group

EICAT category

What do the EICAT categories mean?

EICAT rates the documented harm a species has done to native species where it is already established, not its predicted impact here.

Minimum risk

0%

Maximum uncertainty

40%

### Uncertainty vs invasion risk

### Invasion risk vs EICAT score

Exports the filtered list in its current sort order.

No species match these filters.

Select a column heading to sort. Select a row to see how each feature pushed the prediction up or down.

## What this shows

For a chosen country, the model scores every species that has no recorded alien or native presence there and ranks them by the probability that they would establish as an invasive alien species. Each score comes with an uncertainty estimate and, where one exists, the species' EICAT impact category.

## Data used

- **SINAS** alien and native species records per country. Established alien records are the positive labels; native records feed the species' realised climate niche.
- **GloBI** species interactions, plus genus-, family- and order-level interactions, sorted into resource, enemy, mutualism and competition links.
- **GBIF** occurrence-based presence per species and country, and the taxonomy (genus up to kingdom) used to link species.
- **Country climate**: latitude, longitude, mean temperature, temperature seasonality, frost days, precipitation and aridity index, scaled to 0–1.
- **BACI trade data** (HS17, 2019), grouped into five ecological channels: flora and timber, aquatic fauna, terrestrial pets, agricultural hitchhikers and microbial risk.

## How the model works

- A two-layer heterogeneous GraphSAGE network learns a 64-dimension embedding per species from the interaction and taxonomy graph.
- An autoencoder compresses the seven climate variables of each country into 32 dimensions.
- A five-stream classifier combines: graph topology, Mahalanobis climate-niche distance, per-axis climate suitability, trade pathways and biotic opportunity. A detached climate gate suppresses scores for species that are climatically unsuited.
- Biotic opportunity is the share of a species' helpful partners (food, hosts, pollinators) and harmful partners (predators, parasites, competitors) already present in the target country.
- Trade exposure is routed through the countries where the species is confirmed present.
- Training uses known invasions as positives against sampled negatives, with 5-fold cross-validation. The production model is then trained on all positives.
- Species already recorded in the country are excluded. The workflow can also check candidates against live GBIF occurrence counts.

## Feature contributions

The detail panel uses Integrated Gradients on the model's pre-sigmoid score, against an average-country baseline. It shows 21 contributions: seven climate axes, five trade channels for imports and five for exports, positive and negative biotic opportunity, and the two relational blocks (species embedding, country latent). They add up to the difference between the predicted and baseline score. Red bars raise risk; blue bars lower it.

## How uncertainty is calculated

The ± value is the uncertainty of each prediction, estimated with Monte Carlo dropout. The classifier and the trade encoder keep their dropout layers switched on, and each species–country pair is scored 30 separate times. Every pass drops a different random set of connections, so the passes give slightly different probabilities.

- **Invasion risk** is the mean probability over the 30 passes.
- **Uncertainty (±)** is the standard deviation of those 30 probabilities. A wide spread means the model's answer depends on which connections were active, so the prediction is less certain. It is shown in percentage points, and the charts and the maximum-uncertainty filter use it.
- **Standard error** is the standard deviation divided by √30. It describes how precisely the mean was estimated from 30 passes, so it shrinks as you add passes. Both values are in the CSV export (`uncertainty_sd` and `uncertainty_se_30_passes`).

This captures the model's own uncertainty. It does not cover gaps or errors in the input data, such as missing records or unrecorded interactions.

## EICAT impact categories

EICAT (Environmental Impact Classification for Alien Taxa) is the IUCN standard for rating how much harm an alien species has caused to native species and ecosystems. The category is based on documented impacts, mostly from places where the species is already established. It shows what the species has done elsewhere. It does not predict impact in the selected country. A species receives the highest category supported by any of its impact records.

Impacts are recorded through twelve mechanisms: competition, predation, hybridisation, disease transmission, parasitism, poisoning or toxicity, bio-fouling, grazing or browsing, impacts on ecosystem chemistry, physical structure or fire, and indirect impacts through other species.

The supplied notebook does not produce EICAT categories. In the demo they stand in for a lookup joined on GBIF ID. Most species have never been assessed, so many rows show NE or DD.

## Plugging in the real model

All data goes through two functions that currently simulate results. To use a live backend, host this file with your platform and call `HorizonScanner.configure`. The page listens for no inputs from the host, but fires a `horizonscanner:select` event when a species is opened.

```
HorizonScanner.configure({ apiBase: "/api/horizon" });

GET {apiBase}/scan?country=Belgium&n=500
 -> { rows: [{ gbif_id, taxon_name, group, probability,
                 uncertainty, uncertainty_se, eicat }] }
    uncertainty: SD of the probability over the MC-dropout passes
    uncertainty_se: SD / sqrt(n_passes)  (optional)
    group: plant | invertebrate | fish | herp | bird | mammal | fungi
    eicat: MC | MN | MO | MR | MV | DD | NE   (rows sorted by probability)

GET {apiBase}/explain?gbif_id=123&country=Belgium
 -> { baseline_probability, probability,
      contributions: [{ block, name, value }] }   // value in logit units
    block: climate | trade | biotic | relational

document.addEventListener("horizonscanner:select", e => e.detail)
```

Live calls only work when the page is served from a domain allowed to reach your API; the demo hosted on claude.ai cannot make them.