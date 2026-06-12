# Ground measurements — canonical raw inputs

This directory holds the daily ground-truth GHI CSVs that the SuSSE warehouse
ingests as training data. The files here are the **only** authoritative source
for ground GHI; the warehouse tables `ground_measurements_raw` and
`ground_measurements` are derived from them by `GroundIngestJob` +
`curate_ground` (see [`src/susse/warehouse_ops/population/`](../../src/susse/warehouse_ops/population/)).

The folder is checked into git so any contributor can run NB 01's ingest demo
end-to-end immediately after `git clone`.

## Subfolders

| Subfolder | Source | Coverage |
|---|---|---|
| `MAK_physics_dept/` | Makerere University Physics Department station archive (Uganda) | 3 station files: `kampala.csv`, `lira.csv`, `tororo.csv`. ~5,000 rows each. |
| `ministry_energy_ug/` | Uganda Ministry of Energy and Mineral Development | 2 station files: `soroti.csv`, `wadelai.csv`. ~700 rows each. |

> **CrossBoundary Energy data is *not* checked in.** The 6 country files
> (egypt / ghana / kenya / madagascar / nigeria / somalia) are covered by a
> non-disclosure agreement and may not be distributed publicly. They have
> already been ingested into the BigQuery warehouse and the model bundle
> the portal serves; the originals are retained on Google Drive for
> internal use. The directory is also listed in `.gitignore` so dropping a
> fresh copy into `data/ground_measurements/CBE_Data/` does not
> accidentally re-introduce it.

## Schema

Every CSV in every subfolder shares the same schema — handled by a single
`StandardCsvAdapter` in [`adapters.py`](../../src/susse/warehouse_ops/population/adapters.py):

| Column | Type | Notes |
|---|---|---|
| `datetime` | ISO date string | Daily-resolution; one row per (date, location). |
| `ghi` | float | **Daily total GHI in Wh/m²/day** (raw values; curation converts to kWh/m²/day during warehouse ingest). |
| `location` | string | Station identifier. Makerere / Ministry-of-Energy files use the station name (`kampala`, `soroti`, …); the (private) CBE country files use `<country>_location<N>`. |
| `latitude` | float | Decimal degrees. |
| `longitude` | float | Decimal degrees. |

## Adding a new ground source

If the new provider's CSV matches this schema, drop the file under a new
subfolder here and run the existing `StandardCsvAdapter` (NB 01, Pattern 3).
If the layout differs, subclass `GroundSourceAdapter` and implement
`parse(file_path) -> DataFrame` returning the canonical schema above.
The downstream curation + MERGE pipeline is unchanged — the adapter is the
only thing that knows about the partner's column layout.

## What does NOT live here

- **External validation datasets** (e.g. Katongole 2023) live under
  `data/external_references/`, not here. This directory is reserved for
  ground truth that we curate ourselves and feed into training.
- **Generated artifacts** (training-snapshot parquets, fitted-model bundles)
  live under `data/training_snapshots/` and `data/bundles/`, both gitignored.
