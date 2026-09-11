# Movie recommender backend

Collaborative-filtering recommender on MovieLens, served over FastAPI.
Build order and rationale: [`../docs/design/simple-build-plan.md`](../docs/design/simple-build-plan.md).

## Quickstart

```bash
cd api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
jupyter lab notebooks/01_explore.ipynb
```

## Status

- [x] **Weekend 1** — data download + exploration
- [ ] **Weekend 2** — `train.py`: item-item cosine, centered + shrunk, top-50 neighbors
- [ ] **Weekend 3** — `evaluate.py`: Recall@20 vs popularity baseline ← *the one that matters*
- [ ] **Weekend 4** — `main.py` + `streamlit_app.py`
- [ ] **Weekend 5** — *optional* Gemini natural-language filters

## Dataset facts (from weekend 1, reproduce with the notebook)

| | |
|---|---|
| Users | 610 |
| Movies with ≥1 rating | 9,724 (of 9,742 in catalog) |
| **Serveable catalog** (has `tmdbId` and ≥1 rating) | **9,716** |
| Ratings | 100,836, scale 0.5–5.0 |
| Matrix density | 1.70% |
| Global mean rating | 3.502 |
| Movies with <5 ratings | 62% |
| Date range | 1996-03-29 → 2018-09-24 |

## Note on the download

`files.grouplens.org` has had an **expired TLS certificate since 2026-08-28**, so
`download_data.py` tries the official zip, reports the failure, and falls back to
a mirror whose files are verified against pinned SHA-256 hashes and the published
row counts. Retry the official host later — it will work again once GroupLens
renews, with no change to the script.

The fallback verifies content rather than disabling TLS verification. If you hit
this sort of expired-cert problem elsewhere, resist `verify=False` /
`ssl._create_unverified_context()`: pin the hash instead. It's a stronger check,
and it's the answer you want to have given when someone asks.

The notebook is committed **with its outputs**, so it doubles as a reference for
what your own run should produce. Clear them with
`jupyter nbconvert --clear-output --inplace notebooks/01_explore.ipynb` if the
diffs get noisy.
