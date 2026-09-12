# Simple build plan (start here)

The companion doc [`ai-recommender.md`](ai-recommender.md) is the full design —
what I'd build if I already knew the answers. **This is the version to actually
build first.** Same project, ~20% of the parts.

Read the big doc for §2.1 (why cold-start shapes everything), §2.2 (dataset
choice), and §6 (evaluation). Ignore the rest until something forces you back.

---

## The one thing that matters

A resume reviewer cannot see your architecture. They see one bullet, and if you
get an interview, they ask *"how does it work and how do you know it works?"*

So the deliverable is **one model you can explain on a whiteboard, and one number
proving it beats a dumb baseline.** Everything else — the second model, the
database, the LLM, the deployment — is decoration that costs weeks and answers
no interview question. Build the number first.

---

## What to cut, and why

| Big design | Simplified | Why this is safe to cut |
|---|---|---|
| Two models (kNN + ALS/fold-in) | **One: item-item cosine** | Item-item *is* collaborative filtering. You get personalization from it too (see below) with no matrix factorization, no λ-regularized solve, no `implicit` dependency. Cuts the hardest-to-explain part. |
| SQLite + session ids + 3 ratings endpoints | **No database at all** | Streamlit holds the user's ratings in `st.session_state` and posts them in the request body. API stays stateless. Deletes a whole layer, and the free-tier durability trade-off disappears with it. |
| React integration (phase 7) | **Streamlit only** | Your React app keeps working, untouched. Wiring a new backend into it teaches you CORS, not ML. Do it later if you want, or never. |
| Gemini parse + explain, breaker, cache | **One endpoint, last, optional** | Genuinely a bonus. It is also the only part that can be added in an afternoon *after* everything works. |
| Artifact versioning, `strategy` field, `/meta/metrics`, rate limiting, structured logs, CORS allowlist, deployment | **All of it** | Every line here exists to serve users you don't have. |
| Parquet catalog | **CSV** | Drops the `pyarrow` dependency. The file is 9.7k rows. |
| Leave-one-out split + Recall + NDCG + coverage + popularity metrics | **Recall@20 + popularity baseline** | Keep the split (chronological, per user) — that part is not optional. Add NDCG and coverage later if you want more to talk about. |

Kept, non-negotiable, because these *are* the project:

- MovieLens `ml-latest-small`, downloaded by a script.
- Mean-centering before similarity, and shrinkage by co-rating count.
- A chronological leave-one-out split.
- **Recall@20 compared against a most-popular baseline.**

---

## The whole system

Seven files. Roughly 400 lines of Python total.

```
api/
├── download_data.py    # ~30 lines  urlretrieve + unzip MovieLens
├── train.py            # ~180 lines ratings.csv → model.npz + catalog.csv
├── evaluate.py         # ~60 lines  prints Recall@20: model vs popularity
├── recommend.py        # ~50 lines  the model, as plain functions
├── main.py             # ~60 lines  FastAPI, 3 endpoints
└── requirements.txt
streamlit_app.py        # ~80 lines  the UI
```

Dependencies, complete: `pandas numpy scipy scikit-learn fastapi uvicorn
streamlit requests`. Add `google-genai` only at phase 5.

### Personalization without matrix factorization

This is the simplification that saves the most time. Build the item-item
similarity matrix `S` once (top-50 neighbors per movie, sparse). Then a user's
scores over the whole catalog are **one sparse matrix-vector product**:

```python
# r: sparse vector of the user's ratings, centered so dislikes push away
r = user_ratings - shrunk_user_mean(r)   # see the note below
scores = S.dot(r)               # that's the entire recommender
scores[already_rated] = -np.inf
top = np.argsort(-scores)[:20]
```

No ALS, no fold-in, no linear solve, no cold-start special case — a new user with
3 ratings and a MovieLens user with 500 go through the identical code path.

> ⚠️ **The centering is not cosmetic.** With raw 1–5 ratings every term is
> positive, so rating a movie 1★ still *recommends things like it* — just a bit
> less. Subtracting a midpoint is what makes a dislike push the recommendation
> away. This is the bug you will otherwise spend an evening on, and it's a good
> thing to be able to explain.

> 📌 **Amended in weekend 2.** This section originally said `r = user_ratings - 3.0`.
> A fixed midpoint works, but training centers on each user's *own* mean (people
> rate on different scales — see notebook 01 §2), so serving on a constant leaves
> a train/serve mismatch to explain away. Both now use a mean shrunk toward the
> global average:
>
> ```
> mu_u = (Σ ratings + m · 3.5016) / (n + m)        m = 5
> ```
>
> For a MovieLens user this lands a median of 0.018 from their plain mean, so
> training is unchanged. It earns its keep at serve time: a new user who rates
> three films 5★ has a plain mean of 5.0, which would center their whole vector
> to zero and return nothing. One formula in `train.py`, `evaluate.py` and
> `recommend.py`. Details in [`api/README.md`](../../api/README.md#weekend-2--the-model).

### Three endpoints

```http
GET  /movies/{tmdb_id}/similar?k=20    # neighbor lookup, no user needed
POST /recommend                        # {"ratings":[{"tmdb_id":550,"rating":5}]}
GET  /health
```

`POST /recommend` takes the ratings in the body and returns recommendations.
Stateless. That's why there's no database.

---

## Order to build it, ~5 weekends

Each weekend ends with something that runs. If you stop after any one, what you
have still works and still demos.

| # | Build | Study alongside it |
|---|---|---|
| **1** | `download_data.py`, then a notebook: load `ratings.csv`/`movies.csv`/`links.csv`, plot the rating distribution, count how many movies have a usable `tmdbId`. | pandas groupby/merge; sparse matrices (`scipy.sparse.csr_matrix`) — what CSR is and why a 610×9724 dense matrix is wasteful |
| **2** | `train.py`: build `S`, mean-center, shrink, keep top-50, save `model.npz`. Eyeball it — neighbors of *Toy Story* should be animated films. **If they aren't, stop and fix it before moving on.** | cosine similarity by hand on paper; why centering and shrinkage exist (the "two obscure movies, same 2 raters" failure) |
| **3** | `evaluate.py`: chronological leave-one-out, Recall@20, model vs most-popular vs random. **This is the weekend that matters.** | train/test leakage — specifically why random-splitting ratings across time inflates every number; what Recall@k measures and what it misses |
| **4** | `main.py` + `streamlit_app.py`: rate 5 movies with sliders, see 20 recommendations with posters. | FastAPI path/body params, pydantic models, `st.session_state` |
| **5** | *Optional bonus.* One Gemini call: "slow sci-fi from the 70s" → JSON filters, applied before ranking. Key in an env var, never in the repo. | structured/JSON output from an LLM; validating model output against a schema instead of trusting it |

Weekend 3 is the whole project. Weekends 1–2 produce a model; weekend 3 produces
*evidence*. A recommender that can't beat most-popular is the most common silent
failure in a first ML project, and you cannot tell by looking at the
recommendations — they look plausible either way.

---

## The resume bullet

Write it after weekend 3, with your real numbers:

> **Movie recommender** — Built a collaborative-filtering recommender on the
> MovieLens 100K dataset (Python, scikit-learn, FastAPI, Streamlit). Item-item
> cosine similarity with mean-centering and co-rating shrinkage; achieved
> **Recall@20 of 0.XX vs. 0.YY for a most-popular baseline** under a
> chronological leave-one-out split. Served via a REST API with an interactive
> Streamlit UI.

What makes that bullet work is `0.XX vs 0.YY` and "chronological". Naming the
baseline and the split says you know how recommenders get evaluated — and how
they get evaluated *wrong*. Without the comparison it's an unfalsifiable claim
and a reviewer reads it as one.

Be ready for these, because they're what gets asked:

- *Why item-item and not user-user?* Items are more stable than users, there are
  fewer of them here than you'd have users at scale, and the similarity matrix
  can be precomputed.
- *What's the cold-start problem and how did you handle it?* New user: needs a
  few ratings before scores mean anything; the `/similar` endpoint needs none.
  New movie: no ratings, so CF can't reach it at all — that's a real limitation
  of your model, and saying so is better than claiming you solved it.
- *Why does your model beat popularity by so little?* 610 users is a small,
  sparse matrix. Honest answer, and it shows you know what the numbers depend on.
- *What would you do next?* Hybrid in genre/year content features to handle new
  movies; move to ml-25m; the items in the big doc's §7.

---

## Two things not to do

**Don't skip evaluation to get to the UI.** The UI is more fun and more visible,
and a Streamlit app with no metric behind it is a project a reviewer can't
distinguish from a tutorial. The metric is the part that's yours.

**Don't reach for a neural model to make it look impressive.** A well-evaluated
kNN beats a PyTorch two-tower model you can't explain, in every interview. Add
matrix factorization later as a *second* model compared against this one on the
same split — at that point it's a strong follow-up bullet, because you'll have
something to compare it to.
