# Movie recommender backend

Collaborative-filtering recommender on MovieLens, served over FastAPI.
Build order and rationale: [`../docs/design/simple-build-plan.md`](../docs/design/simple-build-plan.md).

## Quickstart

```bash
cd api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py
jupyter lab notebooks/01_explore.ipynb    # weekend 1
python train.py                            # weekend 2 -- builds artifacts/
pytest tests/ -q
jupyter lab notebooks/02_inspect_model.ipynb
```

## Status

- [x] **Weekend 1** — data download + exploration
- [x] **Weekend 2** — `train.py`: item-item cosine, centered + shrunk, top-50 neighbors
- [ ] **Weekend 3** — `evaluate.py`: Recall@20 vs popularity baseline ← *the one that matters*
- [ ] **Weekend 4** — `main.py` + `streamlit_app.py`
- [ ] **Weekend 5** — *optional* Gemini natural-language filters

## Dataset facts (from weekend 1, reproduce with the notebook)

| | |
|---|---|
| Users | 610 |
| Movies with ≥1 rating | 9,724 (of 9,742 in catalog) |
| **Serveable catalog** (has `tmdbId` and ≥1 rating) | **9,716** |
| ↳ after de-duplicating `tmdbId` (weekend 2) | **9,715** |
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

---

## Weekend 2 — the model

`python train.py` turns `ratings.csv` into `artifacts/model.npz` in about 1.5 seconds.

> Written up ahead of finishing it: the numbers below are from a verified run, but
> the four `TODO(you)` blocks in `train.py` are still yours to fill (see the last
> section). Edit this into your own words once you have — it is the draft of the
> answer you will give when someone asks how the model works.

### What I solved

Given 100,836 ratings from 610 people, rank all 9,724 movies for someone who has rated
five of them — without matrix factorisation, and without a separate code path for new
users.

The answer is item-item collaborative filtering: precompute how similar every pair of
movies is, and a user's scores over the whole catalog become a single sparse
matrix-vector product, `S.dot(r)`. A brand-new user with 3 ratings and a MovieLens user
with 500 go through identical code. Three things make the similarities honest:

| | |
|---|---|
| **Centre each rating** on a shrunk per-user mean | Raw MovieLens ratings are 0% negative, so a 1★ still *recommends* similar films. After centring ~46% are negative, which is what lets a dislike push away. |
| **Shrink** each similarity by `n_co / (n_co + λ)` | 35% of movies have exactly one rating. Two obscure films sharing one viewer otherwise score a near-perfect cosine. |
| **Keep the top 50** neighbours per movie | The full similarity matrix is 26.3M nonzeros; the truncated one is 1.9 MB. |

The centring mean is `(Σr + m·3.5016) / (n + m)` with `m = 5`. For a MovieLens user
(minimum 20 ratings) that sits a median of 0.018 from their plain mean, so training is
unaffected — but a new user who rates three films 5★ has a plain mean of 5.0, which
would centre their entire vector to zero and return nothing. One formula, used at train,
eval and serve time, so there is no train/serve mismatch to explain.

### Tools

- **`scipy.sparse`** — CSR to build the ratings matrix, CSC to slice movie columns.
  Centring works by replacing `.data` in place, which keeps the sparsity structure
  (including ratings that centre to exactly 0.0) byte-for-byte identical to `R`.
- **Blocked matrix products.** The full item-item matrix is 26,325,068 nonzeros — 27.8%
  dense, ~211 MB — and the co-rating counts need a second one the same size. So it is
  never built: 512 movie columns at a time are multiplied, shrunk, truncated and
  discarded. Peak memory is ~40 MB for the slabs. Scaling to the 25M-rating dataset
  needs no structural change, just more blocks.
- **`np.argpartition`** — top-50 of 9,724 without sorting all of them, then `argsort` on
  just the 50 to order them.
- **pandas `groupby`** for the centring means and the catalog statistics.
- **pytest** — a 3×3 matrix worked out on paper, plus property tests (blocked result
  equals brute force; block size never changes the model; shrinkage is monotone in
  evidence and never flips a sign).

### What I learned

- **Adjusted cosine is just cosine after you fix the units.** Centring is not
  preprocessing hygiene, it is the step that gives "dislike" a direction.
- **Shrinkage is a prior, and it is the same move three times** — the shrunk user mean,
  the shrunk similarity, and the shrunk popularity mean in the catalog all say "with
  little evidence, fall back toward the default". That generalises; the specific formula
  does not.
- **Co-rating counts must come from the sparsity pattern, not the values.** 240 ratings
  centre to exactly 0.0 under plain user-mean centring. SciPy stores them, so `C.nnz`
  looks right, but the natural-looking `B = (C != 0)` drops all 240 from the evidence
  counts. Subtle, silent, and only visible if you go looking.
- **Divide-by-zero hides well in sparse code.** A movie whose centred column is all
  zeros has norm 0; without a guard the NaN spreads through every dot product and you
  ship an artifact whose neighbour lists are merely "a bit odd".
- **Top-k truncation makes the similarity matrix asymmetric.** Expected, not a bug — but
  it means "similar to" is directional, and the `/similar` endpoint inherits that.
- **Shrinkage does not evict the long tail from the lists.** Single-rating films still
  occupy 34.9% of all neighbour slots, because fifty slots have to hold something. What
  changed is that they hold **0.0%** of the slots belonging to films with 20+ ratings,
  and sit at similarity ~0.09. They can never outrank real evidence where real evidence
  exists, which is the outcome that actually matters.

### What I'd improve in weekend 3

- **λ=10 and k=50 are untuned placeholders.** Picking them by eyeballing neighbour lists
  is overfitting to the three films I happened to check. Weekend 3 produces Recall@20;
  tune against that.
- **Nothing here proves the model works.** The neighbour lists are plausible, but a
  recommender that loses to most-popular produces equally plausible lists. That is the
  whole reason weekend 3 exists.
- **Item cold-start is unsolved by construction.** 18 catalog movies have no ratings and
  no column in `R`, so CF cannot reach them, ever. Fixing it means content features
  (genre, year) — a hybrid model, not a tweak.
- **Popularity leaks in locally.** *The Godfather* ranks 5th among Star Wars IV's
  neighbours. With 610 users, "watched by people who rate a lot of films" is a real
  signal the model partly picks up.

### Artifacts (`artifacts/`, gitignored — regenerate with `python train.py`)

`model.npz`:

| Key | Shape / type | |
|---|---|---|
| `movie_ids` | int32 (9724,) | column index → MovieLens `movieId` |
| `neighbor_idx` | int32 (9724, 50) | column indices, descending by similarity |
| `neighbor_sim` | float32 (9724, 50) | shrunk cosine similarities |
| `global_mean` | float64 | 3.5016 — weekend 3/4 must centre with this exact value |
| `lam`, `topk`, `center_m` | scalars | the parameters this artifact was built with |
| `n_users`, `n_ratings`, `built_at` | scalars | provenance |

The parameters live inside the artifact so that weekend 3's sweep can leave several
`.npz` files on disk and the file itself still says which is which.

`catalog.csv` — 9,715 rows, one per serveable movie:
`index, movieId, tmdbId, title, year, genres, n_ratings, mean_rating, shrunk_mean`

- `index` points into the model arrays. The **model** covers all 9,724 rated movies (the
  8 without a `tmdbId` still contribute co-rating evidence); the **catalog** is what the
  API can return. Filter at serve time, not at train time.
- `shrunk_mean` = `(n·r̄ + 20·3.5016)/(n + 20)` is weekend 3's popularity baseline,
  precomputed. Ranking by raw mean returns films with a single 5.0 vote — a baseline you
  deliberately weakened proves nothing when your model beats it.
- `year` is nullable: 13 titles carry no `(YYYY)` suffix.
- One `tmdbId` (4912) maps to two movieIds; the one with more ratings wins.

### Filling in the TODOs

`train.py` ships with four blocks — the centring mean, the centring itself,
column normalisation, and the shrinkage line. Each has the formula in the comment above
it and a check immediately below that fails with a specific diagnosis. Work top to
bottom, running `python train.py` after each.

The gate for the weekend: **Toy Story's neighbours must be mostly Animation/Children.**
Expected output with the defaults:

```
0.359  Toy Story 2 (1999)          0.660  The Empire Strikes Back     0.534  Aliens (1986)
0.299  Aladdin (1992)              0.612  Return of the Jedi          0.326  Blade Runner (1982)
0.267  Toy Story 3 (2010)          0.427  Raiders of the Lost Ark     0.306  Dr. Strangelove
```

If they aren't, the causes in order of likelihood: centring skipped, `n_co` built from
the centred matrix instead of the rating pattern, or rows normalised instead of columns.
