# 🎬 Movie Finder → AI Movie Recommender

**Goal:** take a working-but-boring TMDB browser — endless grids of posters that
show everyone the same thing — and turn it into a recommender that learns what
*you* like. A React frontend gets a Python ML backend: a collaborative-filtering
model trained on real ratings, served over FastAPI, with an evaluation number
behind it rather than a vibe.

## What it is

Movie Finder started as a React app that browses TMDB — trending, top-rated,
genres, search, trailers, cast. That part works. What it can't do is
*personalize*: every visitor sees the identical popularity-ordered list.

This project adds the missing half. A FastAPI service trained on the
[MovieLens 100K](https://grouplens.org/datasets/movielens/) dataset (610 users,
100,836 real ratings) computes item-to-item similarity from actual rating
behaviour — "people who liked *Heat* also liked…" — and turns a handful of
ratings from a new visitor into a ranked list of 20 recommendations.

Built solo over ~5 weekends, one milestone at a time, each one demoable on its own.

## Tech stack

| Layer | Tools |
|---|---|
| **Frontend** | React 17, React Router 5, React Bootstrap 5, react-slick, react-player |
| **Backend** | Python 3.11, FastAPI, Uvicorn, Pydantic |
| **ML / data** | pandas, NumPy, SciPy (sparse matrices), scikit-learn |
| **Demo UI** | Streamlit |
| **Data** | MovieLens `ml-latest-small`, TMDB API (posters, metadata, trailers) |
| **LLM** *(bonus)* | Google Gemini — natural-language filters, not the recommender |
| **Tooling** | Jupyter Lab, Create React App, Git |

## Features

Honest status — this is built in public, in order.

### Working now

- **Browse & discover** — trending (day/week), now playing, top rated, popular
- **Browse by genre**, with pagination
- **TV series** section
- **Search** across titles
- **Movie detail** — synopsis, cast, star rating, YouTube trailer, similar titles
- **Responsive UI** with carousels and skeleton loading states
- **Reproducible dataset pipeline** — one script downloads and *verifies* MovieLens
  (SHA-256 pinned, row counts checked against published statistics)
- **Exploratory analysis notebook** — sparsity, rating distribution, the long
  tail, TMDB id coverage

### In progress

| Feature | Milestone | Status |
|---|---|---|
| Item-item collaborative filtering (`/movies/{id}/similar`) | Week 2 | ⏳ |
| **Offline evaluation — Recall@20 vs. a popularity baseline** | Week 3 | ⏳ |
| Personalized `POST /recommend` — rate 5 movies, get 20 back | Week 3 | ⏳ |
| FastAPI service + Streamlit demo UI | Week 4 | ⏳ |
| Natural-language search via Gemini ("slow sci-fi from the 70s") | Week 5 | ⏳ *bonus* |

Week 3 is the one that matters: a recommender without an offline metric is a pile
of vibes. Design and rationale live in
[`docs/design/simple-build-plan.md`](docs/design/simple-build-plan.md), with the
fuller architecture in [`docs/design/ai-recommender.md`](docs/design/ai-recommender.md).

---

## Getting started

### Prerequisites

- **Node.js** 16+ and npm
- **Python** 3.9+
- A free [TMDB API key](https://www.themoviedb.org/settings/api) (frontend)
- *(Week 5 only)* a free [Google AI Studio key](https://aistudio.google.com/apikey)

### 1. Clone

```bash
git clone https://github.com/dth2701/Movie-Finder.git
cd Movie-Finder
```

### 2. Frontend

```bash
npm install
npm start
```

Opens <http://localhost:3000>.

> ⚠️ The TMDB key is currently hardcoded in `src/sevice/API.js`. Replace it with
> your own — and see [Week 1 → what I'd improve](#what-id-improve-next-week-2)
> for why that's a mistake I'm fixing rather than one to copy.

### 3. Backend — data & notebook

```bash
cd api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python download_data.py          # downloads + verifies MovieLens (~2.5 MB)
jupyter lab notebooks/01_explore.ipynb
```

`download_data.py` is idempotent — re-running skips the download. Use `--force`
to re-fetch. Details in [`api/README.md`](api/README.md).

### 4. API & demo UI *(from week 4)*

```bash
uvicorn main:app --reload --port 8000   # docs at localhost:8000/docs
streamlit run ../streamlit_app.py       # UI at localhost:8501
```

---

# Week 1 — data foundation

**Shipped:** a reproducible dataset pipeline and an exploratory analysis that
decided the model's design.

### What problem I solved

I had an app with **zero ratings in it**. Nobody logs in, and the stars on each
card just display TMDB's public average. Collaborative filtering needs a
user × item ratings matrix, so before writing any model I needed real rating
data — and I needed to actually understand it, because I'd already learned that
guessing at the shape of your data means discovering it later as a bug.

Three concrete problems, and what I did:

**1. Getting trustworthy data without scraping.** The brief allowed scraping or
downloading. I chose MovieLens: real explicit ratings with timestamps, a
research licence, and — the deciding factor — a `links.csv` that maps every
movie to a TMDB id, so it joins straight onto the frontend I already had.
Scraping IMDb would have meant building a crawler and a rate limiter before
writing a line of ML, and TMDB doesn't expose per-user ratings anyway.

**2. The official download was broken.** `files.grouplens.org` has had an
**expired TLS certificate since 2026-08-28**, so the download just failed. I
verified it was genuinely server-side rather than my machine — checked the
served certificate's expiry directly, and confirmed `curl` failed identically to
Python.

The obvious fix that every Stack Overflow answer suggests is
`verify=False` / `ssl._create_unverified_context()`. **I didn't do that**, because
it means accepting any bytes anyone hands you. Instead I found the dataset
mirrored, confirmed two independent copies were byte-identical, pinned their
SHA-256 hashes in the script, and verified the extracted row counts against
MovieLens' published statistics. My script now tries the official host first,
explains the failure if it fails, and falls back to the verified mirror — so it
starts using the official source again the moment GroupLens renews, with no
code change.

Pinning the hash verifies the *content*, which is strictly stronger than
trusting the *transport*. That reframing was the most useful thing I learned all
week.

**3. Not knowing what the data actually looked like.** I wrote a notebook to
answer four questions before modelling anything:

| Question | Answer |
|---|---|
| How sparse is the matrix? | **1.70%** filled — 610 × 9,724, only 100,836 ratings |
| What's the rating distribution? | 0.5–5.0, skewed high (mean **3.502**), and every user has their own scale |
| How many movies map to TMDB? | **9,716 serveable** — 8 missing a `tmdbId`, plus 1 duplicate |
| Is "average rating" a usable ranking? | **No** — the top 10 by raw mean are all films with a single 5.0 vote |

### Tools I used

- **`pandas`** — loading and joining the three CSVs, `groupby` aggregations
- **`scipy.sparse`** (CSR) — storing the ratings matrix; **29× smaller** than
  dense here, and the gap widens with dataset size
- **`matplotlib`** — rating distribution, per-user means, log-log long-tail plots
- **`hashlib` + `urllib`** — checksum-verified downloads
- **Jupyter Lab** — exploration
- **`jupyter nbconvert --execute`** — running the notebook top-to-bottom in CI
  style, to prove it works from a clean kernel rather than from lucky cell order

### What I learned

**Centering ratings isn't cosmetic — I measured it.** This was the week's real
insight. I checked what fraction of the rating signal is negative:

```
raw ratings        →   0.0% negative
minus 3.0          →  18.9% negative
minus each user's mean →  45.7% negative
```

With raw 1–5 values, *every* term in a similarity sum is positive — so rating a
film 1★ still nudges the model **toward** similar films, just less strongly. The
model literally cannot represent dislike. Subtracting each user's own mean is
what fixes it. I'd read "remember to mean-center" in tutorials and skipped past
it; seeing `0.0% negative` in my own output is what made it stick.

**Sparse matrices are a habit, not an optimization.** Dense would have been
47 MB here — survivable. The same arithmetic on MovieLens 25M is tens of GB. I
learned what CSR actually stores (`data`, `indices`, `indptr`) instead of
treating it as a magic wrapper.

**62% of movies have fewer than 5 ratings.** That's why similarity needs
*shrinkage*: two obscure films rated by the same two people score a perfect 1.0
similarity on evidence of two people. The long-tail plot made an abstract
warning concrete.

**Popularity is a genuinely strong baseline, and an easy one to cheat.** Ranking
by raw mean returns nonsense, so it's tempting to use *that* as the baseline my
model beats. Beating a baseline you deliberately weakened proves nothing. The
honest version shrinks the mean toward the global average — and it returns real,
recognizable films, which is exactly what makes it hard to beat.

**Verify, don't assume.** When the download failed, my first instinct was "my
Python is misconfigured." Checking the actual certificate took two minutes and
pointed at a completely different fix. Same lesson in the notebook: the design
doc *predicted* some movies would lack a TMDB id; measuring gave me 8 missing
**and 1 duplicate** — meaning `tmdb_id → movieId` isn't one-to-one, which I'd
otherwise have found as a duplicated row in an API response.

**A limitation named is better than a limitation hidden.** 18 movies have no
ratings at all, so collaborative filtering **cannot recommend them** — ever.
That's the item cold-start problem. It's inherent to the method, not a bug, and
I'd rather be able to explain it than pretend I solved it.

### What I'd improve next (week 2)

**Build `train.py`** — item-item cosine similarity, carrying the six decisions
the notebook settled:

1. Sparse CSR throughout
2. Mean-center before computing similarity *(the 0.0%-negative finding)*
3. Shrink similarity by co-rating count, `sim × n_co/(n_co + λ)` *(the 62% finding)*
4. Popularity baseline = **shrunk** mean, not raw
5. Keep top-50 neighbours per movie — never materialize the full 9,724² matrix
6. Tie-break duplicate `tmdbId`s toward the `movieId` with more ratings

**Sanity-check by eye before trusting any metric.** The neighbours of *Toy Story*
should be animated children's films. If they aren't, something is wrong and no
amount of Recall@20 will tell me what.

**Things I know are wrong and am deliberately deferring:**

- **The TMDB key is hardcoded in `src/sevice/API.js` and in git history.** Any
  key in a Create React App bundle ships to every visitor — `REACT_APP_*`
  included, since those are inlined at build time. It needs rotating, and the
  real fix is proxying TMDB through FastAPI. The lesson is already applied to
  the backend: the Gemini key will live *only* server-side, in an environment
  variable, and `.env` is gitignored.
- `package.json`'s `homepage` still points at the repo I learned from, so
  `npm run deploy` would publish to the wrong place.
- The notebook is committed with its outputs (helpful as a reference for what a
  correct run looks like, noisy in diffs). If it gets annoying:
  `jupyter nbconvert --clear-output --inplace notebooks/01_explore.ipynb`.

---

## Project structure

```
Movie-Finder/
├── src/                          # React frontend
│   ├── component/                # layout, home, carousels, cards
│   ├── Pages/                    # routed pages
│   └── sevice/                   # TMDB API client + config
├── api/                          # Python backend
│   ├── download_data.py          # ✅ week 1 — verified dataset fetch
│   ├── notebooks/01_explore.ipynb# ✅ week 1 — exploratory analysis
│   ├── train.py                  # ⏳ week 2
│   ├── evaluate.py               # ⏳ week 3
│   └── main.py                   # ⏳ week 4 — FastAPI
├── streamlit_app.py              # ⏳ week 4
└── docs/design/
    ├── simple-build-plan.md      # the plan I'm following
    └── ai-recommender.md         # fuller architecture + trade-offs
```

## Credits

- Ratings data: [GroupLens MovieLens](https://grouplens.org/datasets/movielens/) —
  Harper & Konstan, 2015, *The MovieLens Datasets: History and Context*, ACM TiiS 5(4)
- Movie metadata, posters, trailers: [TMDB](https://www.themoviedb.org/) (this
  product uses the TMDB API but is not endorsed or certified by TMDB)
