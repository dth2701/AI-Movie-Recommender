"""Weekend 2: build the item-item similarity model.

Usage:
    python train.py                             # build with the defaults below
    python train.py --lam 20 --topk 100         # for the weekend-3 parameter sweep
    python train.py --inspect "Blade Runner"    # print one movie's neighbours and exit

Reads  api/data/raw/ml-latest-small/{ratings,movies,links}.csv
Writes api/artifacts/model.npz   -- the model: top-K neighbours per movie
       api/artifacts/catalog.csv -- what the API can serve, plus the popularity baseline

Both are gitignored. They are reproducible from this script, which is the point.

------------------------------------------------------------------------------
WHAT THIS SCRIPT COMPUTES, IN ONE PARAGRAPH
------------------------------------------------------------------------------
Two movies are similar if the same people liked both. Formally: put every user's
ratings in a big sparse matrix R (users x movies), subtract each user's own mean
so that "dislike" becomes a negative number, normalise each movie's column to
unit length, and the cosine similarity between movies i and j is just the dot
product of their two columns. Then multiply by n_co/(n_co + lambda) so that a
similarity computed from two co-raters counts for less than one computed from
two hundred. Keep the top 50 per movie. That's the whole model.

------------------------------------------------------------------------------
THE THREE IDEAS, AND WHY EACH ONE IS NOT OPTIONAL
------------------------------------------------------------------------------
1. CENTERING. Raw MovieLens ratings run 0.5-5.0 and 0% of them are negative, so
   every term in a cosine sum is positive: rating a film 1 star still recommends
   films like it, just slightly less. Subtracting a per-user mean makes roughly
   half the values negative, which is what lets a dislike push the model away.
   Notebook 01 cell 9 measures exactly this.

   We subtract a SHRUNK user mean, not the plain one:

       mu_u = (sum of u's ratings + m * global_mean) / (n_u + m)     m = 5

   For a MovieLens user (all have >= 20 ratings) this differs from their plain
   mean by a median of 0.018 -- training is unchanged. It matters at serving
   time: a new user who rates three films 5 stars has a plain mean of 5.0, so
   plain centering zeroes out their entire vector and the API returns nothing.
   Shrinking toward the global mean fixes that with one extra term, and it means
   train.py, evaluate.py and recommend.py all center identically. No train/serve
   mismatch to explain away in an interview.

2. SHRINKAGE. 35% of the movies here have exactly one rating and another 13%
   have two. Two obscure films rated by the same two people get a cosine of 1.0
   -- perfect twins, on evidence of two people. Multiplying by
   n_co/(n_co + lambda) pulls low-evidence pairs toward zero. Notebook 02's
   lambda ablation shows this failing and then not failing.

3. TOP-K TRUNCATION. The full item-item matrix for this dataset has 26.3 million
   nonzeros (27.8% dense, ~211 MB sparse) and you need a second matrix the same
   size for the co-rating counts. So we never build it: we take 512 movie columns
   at a time, shrink and truncate inside the loop, and keep only (9724, 50). Peak
   memory ~40 MB, runtime ~1.5 s. It is also the answer to "what would you change
   to run this on the 25-million-rating dataset?" -- nothing structural, just more
   blocks.

   Truncation makes the similarity matrix ASYMMETRIC: j can be in i's top 50
   while i is not in j's. That is expected and fine.

------------------------------------------------------------------------------
FOUR THINGS FOR YOU TO WRITE
------------------------------------------------------------------------------
Search this file for "TODO(you)". Each one is 1-3 lines, has the formula in the
comment above it, and is followed by a check that fails loudly and tells you what
went wrong. Fill them in top to bottom and run the script after each.
"""

import argparse
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

HERE = Path(__file__).parent
RAW_DIR = HERE / "data" / "raw" / "ml-latest-small"
ARTIFACT_DIR = HERE / "artifacts"

# Defaults. Do NOT tune LAMBDA/TOPK this weekend by looking at neighbour lists --
# you have no metric yet, so "tuning" means overfitting to the three movies you
# happened to check. Weekend 3 gives you Recall@20; tune against that.
LAMBDA = 10.0  # shrinkage strength: a pair needs ~lambda co-raters to count fully
TOPK = 50  # neighbours kept per movie
CENTER_M = 5.0  # prior strength for the shrunk user mean
BLOCK = 512  # movie columns processed per block
POP_PRIOR = 20.0  # prior for catalog.shrunk_mean (notebook 01 cell 15)


# =============================================================================
# 1. Load
# =============================================================================


def load_data():
    """Read the three CSVs. Fail with a useful message if they aren't there."""
    if not RAW_DIR.exists():
        sys.exit(f"no dataset at {RAW_DIR}\nrun:  python download_data.py")
    ratings = pd.read_csv(RAW_DIR / "ratings.csv")
    movies = pd.read_csv(RAW_DIR / "movies.csv")
    links = pd.read_csv(RAW_DIR / "links.csv")
    return ratings, movies, links


def build_index(ratings):
    """Map userId/movieId onto contiguous 0..n-1 matrix positions.

    Notebook 01 used `.astype("category").cat.codes`, which does the same thing
    implicitly. We do it explicitly because `movie_ids` has to be SAVED -- it is
    the only thing that lets weekend 4 turn "column 4821" back into a real movie.

    Sorted order is a deliberate choice: it makes the artifact reproducible
    regardless of the row order of ratings.csv.
    """
    movie_ids = np.sort(ratings.movieId.unique()).astype(np.int64)
    user_ids = np.sort(ratings.userId.unique()).astype(np.int64)
    col_of = pd.Series(np.arange(len(movie_ids)), index=movie_ids)
    row_of = pd.Series(np.arange(len(user_ids)), index=user_ids)
    return movie_ids, user_ids, row_of, col_of


def build_matrices(ratings, n_users, n_items, row_of, col_of):
    """Build R (ratings) and B (binary) as CSR, sharing the same structure.

    B is the co-rating evidence: B[u, i] = 1 means "user u rated movie i", and
    B.T @ B counts how many users rated both of any two movies. Build it from the
    same (row, col) index arrays as R, never from a test on the centered values.

    Why that specifically: some ratings center to exactly 0.0 (240 of the 100,836
    do under plain user-mean centering; with the m=5 shrunk mean it happens not
    to hit any, but m is a flag you can change). SciPy keeps those as explicit
    stored zeros, so `C.nnz` still looks right -- but the natural-looking
    `B = (C != 0)` evaluates them as False and silently drops all 240 from the
    co-rating counts. The sparsity pattern means "rated", not "rated nonzero";
    keep it independent of the values and the trap never comes up.
    """
    rows = row_of[ratings.userId].to_numpy()
    cols = col_of[ratings.movieId].to_numpy()
    shape = (n_users, n_items)
    R = csr_matrix((ratings.rating.to_numpy(np.float32), (rows, cols)), shape=shape)
    B = csr_matrix((np.ones(len(ratings), np.float32), (rows, cols)), shape=shape)
    return R, B


# =============================================================================
# 2. Centering
# =============================================================================


def shrunk_user_mean(sums, counts, m, global_mean):
    """Per-user mean, pulled toward the global mean by a prior of `m` ratings.

        mu_u = (sum_of_u's_ratings + m * global_mean) / (n_u + m)

    `sums` and `counts` are arrays, one entry per user, already in matrix-row
    order. Return an array of the same shape.

    With m = 0 this is the plain user mean. With m -> infinity everyone gets the
    global mean. m = 5 is "pretend every user also rated five films at exactly
    the global average" -- negligible for a user with 200 ratings, decisive for
    one with three.
    """
    # TODO(you) #1 -- one line, the formula is three lines above.
    return (sums + m * global_mean) / (counts + m)


def _check_shrunk_mean():
    """Two cases: a heavy user should barely move, a new user should move a lot."""
    heavy = shrunk_user_mean(np.array([700.0]), np.array([200.0]), CENTER_M, 3.5)
    if not abs(heavy[0] - 3.5) < 0.05:
        sys.exit(
            f"TODO #1 wrong: a user with 200 ratings averaging 3.5 came out at "
            f"{heavy[0]:.3f}. A heavy user's shrunk mean should stay near their "
            f"own mean -- check you divided by (n + m), not (n * m) or just n."
        )
    new_user = shrunk_user_mean(np.array([15.0]), np.array([3.0]), CENTER_M, 3.5)
    if not 3.5 < new_user[0] < 4.6:
        sys.exit(
            f"TODO #1 wrong: three 5-star ratings gave a mean of {new_user[0]:.3f}. "
            f"It should sit between the global mean (3.5) and their raw mean (5.0) "
            f"-- that gap is the whole reason this function exists."
        )


def center_ratings(R, mu):
    """Subtract each user's shrunk mean from their observed ratings.

    Return a new CSR matrix with R's EXACT sparsity structure (same nnz, same
    indices, same indptr) and centered values.

    The trick is that CSR stores values row by row in `R.data`, and
    `np.diff(R.indptr)` tells you how many values belong to each row -- which is
    exactly what you printed in notebook 01 when you reconstructed row 0. So:

        C = R.copy()
        row_of_value = <an array the same length as R.data, giving the row each
                        value belongs to -- np.repeat is the tool>
        C.data = R.data - mu[row_of_value]

    Note what we do NOT do: subtract mu from the whole dense matrix. The zeros in
    a sparse ratings matrix mean "not rated", not "rated zero". Centering them
    would invent ~9,560 opinions per user that nobody expressed -- which is also
    why we replace .data in place instead of densifying anything.
    """
    # TODO(you) #2 -- three lines, sketched above.
    C = R.copy()
    row_of_value = np.repeat(np.arange(R.shape[0]), np.diff(R.indptr))
    C.data = R.data - mu[row_of_value]
    return C


def _check_centering(R, C):
    if C.shape != R.shape or C.nnz != R.nnz:
        sys.exit(
            f"TODO #2 wrong: centered matrix is {C.shape} with {C.nnz:,} stored "
            f"values, expected {R.shape} with {R.nnz:,}. The structure must be "
            f"preserved exactly: copy R and replace .data. Anything that filters "
            f"on the centered values (.eliminate_zeros(), boolean masks) throws "
            f"away the ratings that happen to center to 0.0."
        )
    pct_negative = 100.0 * (C.data < 0).mean()
    if not 35.0 < pct_negative < 65.0:
        sys.exit(
            f"TODO #2 wrong: only {pct_negative:.1f}% of centered ratings are "
            f"negative. Notebook 01 cell 9 measured ~50% after centering (and 0% "
            f"before). If you see ~0%, the subtraction didn't happen; if you see "
            f"something lopsided, you probably indexed mu by the wrong axis."
        )


def normalize_columns(C):
    """L2-normalise each MOVIE column, so a dot product is a cosine.

    Return (Cn, norms) where Cn is sparse and `norms` is the (n_items,) array of
    pre-normalisation column lengths.

        norms = sqrt(sum of squares down each column of C)
        norms[norms == 0] = 1.0        <-- the guard; see below
        Cn    = C scaled so column i is divided by norms[i]

    Useful: `C.multiply(C).sum(axis=0)` gives the column sums of squares as a
    numpy matrix -- `np.asarray(...).ravel()` turns it into a flat array. And
    `C.multiply(row_vector)` broadcasts across columns, so multiplying by
    1.0/norms scales each column.

    THE GUARD IS NOT DEFENSIVE PROGRAMMING, IT IS A REAL BUG. A movie whose every
    centered rating is exactly 0 has a column of length 0. Dividing by it gives
    NaN, NaN spreads through every dot product it touches, and you end up with a
    silently corrupt .npz file whose neighbour lists look merely "a bit odd".
    Setting the norm to 1.0 leaves that column as zeros, which is the honest
    answer: we know nothing about that movie.
    """
    # TODO(you) #3 -- three lines, sketched above. Return (Cn, norms).
    norms = np.sqrt(np.asarray(C.multiply(C).sum(axis=0)).ravel())
    norms[norms == 0] = 1.0
    return C.multiply(1.0 / norms).tocsc().astype(np.float32), norms


def _check_normalisation(Cn, norms):
    if np.isnan(Cn.data).any():
        sys.exit(
            "TODO #3 wrong: NaN in the normalised matrix. That is the zero-norm "
            "column dividing by zero -- add the norms[norms == 0] = 1.0 guard."
        )
    lengths = np.sqrt(np.asarray(Cn.multiply(Cn).sum(axis=0)).ravel())
    nonzero = lengths > 0
    if not np.allclose(lengths[nonzero], 1.0, atol=1e-5):
        worst = np.abs(lengths[nonzero] - 1.0).max()
        sys.exit(
            f"TODO #3 wrong: columns are not unit length (worst error {worst:.4f}). "
            f"If they're wildly off you may have normalised ROWS (users) instead "
            f"of COLUMNS (movies) -- axis=0 sums down columns."
        )
    n_zero = int((norms == 0).sum())
    if n_zero:
        print(f"    note: {n_zero} movie(s) had a zero-length column (guarded)")


# =============================================================================
# 3. Similarity
# =============================================================================


def apply_shrinkage(sim, n_co, lam):
    """Scale each similarity down by how much evidence it rests on.

        sim *= n_co / (n_co + lambda)

    n_co[a, b] is how many users rated BOTH movies. With lambda = 10: a pair with
    2 co-raters keeps 2/12 = 17% of its similarity, a pair with 100 keeps 91%.
    Same shape as `sim`. Return the scaled array.

    This is the same shrink-toward-a-prior move as shrunk_user_mean above and as
    the shrunk_mean in the catalog. Three uses of one idea: when you have little
    evidence, fall back toward the default.
    """
    # TODO(you) #4 -- one line.
    return sim * (n_co / (n_co + lam))


def top_neighbors(Cn, B, lam, topk, block):
    """Blocked item-item cosine -> (neighbor_idx, neighbor_sim), both (n_items, topk).

    Everything here except the TODO is mechanical. The interesting part is that
    the full similarity matrix never exists: each iteration builds a
    (block, n_items) slab, shrinks it, drops self-similarity, keeps the top
    `topk`, and throws the slab away.
    """
    n_items = Cn.shape[1]
    kth = min(topk, n_items - 1)
    Cn = Cn.tocsc()
    Bc = B.tocsc()

    neighbor_idx = np.zeros((n_items, topk), dtype=np.int32)
    neighbor_sim = np.zeros((n_items, topk), dtype=np.float32)

    for start in range(0, n_items, block):
        end = min(start + block, n_items)

        # (block x n_items): cosine between each movie in this block and every movie
        sim = (Cn[:, start:end].T @ Cn).toarray().astype(np.float32)
        n_co = (Bc[:, start:end].T @ Bc).toarray().astype(np.float32)

        sim = apply_shrinkage(sim, n_co, lam)  # <-- TODO(you) #4

        # A movie is always its own best match. Remove it before taking the top k.
        sim[np.arange(end - start), np.arange(start, end)] = -np.inf

        # argpartition finds the top k in O(n) without sorting all 9,724 columns;
        # then we sort just those k so the neighbour lists come out descending.
        part = np.argpartition(-sim, kth, axis=1)[:, :topk]
        vals = np.take_along_axis(sim, part, axis=1)
        order = np.argsort(-vals, axis=1)
        neighbor_idx[start:end] = np.take_along_axis(part, order, axis=1)
        neighbor_sim[start:end] = np.take_along_axis(vals, order, axis=1)

    return neighbor_idx, neighbor_sim


def _check_neighbors(neighbor_idx, neighbor_sim):
    if np.isnan(neighbor_sim).any():
        sys.exit("neighbour similarities contain NaN -- something upstream divided by zero")
    if np.isinf(neighbor_sim).any():
        sys.exit(
            "neighbour similarities contain +/-inf. If a movie has fewer than topk "
            "candidates the -inf self-marker can survive into the results."
        )
    drops = np.diff(neighbor_sim, axis=1)
    if (drops > 1e-6).any():
        sys.exit("neighbour lists are not sorted descending -- check the argsort step")
    self_hit = (neighbor_idx == np.arange(len(neighbor_idx))[:, None]).any()
    if self_hit:
        sys.exit(
            "a movie appears in its own neighbour list -- the self-exclusion line "
            "is wrong (remember the block is offset: row r of the slab is movie "
            "start + r)"
        )
    if float(neighbor_sim.max()) > 1.0 + 1e-5:
        sys.exit(
            f"max similarity is {neighbor_sim.max():.3f}, which is impossible for a "
            f"shrunk cosine (<= 1.0). Did shrinkage get applied as n_co/lambda, or "
            f"were the columns not normalised?"
        )


# =============================================================================
# 4. Catalog
# =============================================================================


def build_catalog(ratings, movies, links, movie_ids, global_mean, prior=POP_PRIOR):
    """Rows the API can actually serve: has a tmdbId AND at least one rating.

    `index` points into the model arrays. The MODEL covers all 9,724 rated movies
    -- the 8 without a tmdbId still contribute co-rating evidence to everything
    else, so there is no reason to drop them from training. The CATALOG is the
    9,716 we can return to a client. Filter at serve time, not at train time.

    `shrunk_mean` is weekend 3's popularity baseline, precomputed here:

        (n * mean + prior * global_mean) / (n + prior)

    Notebook 01 cell 13 showed why the raw mean is useless for ranking -- it
    returns films with a single 5.0 vote. A baseline you deliberately weakened
    proves nothing when your model beats it.
    """
    stats = ratings.groupby("movieId").rating.agg(n_ratings="size", mean_rating="mean")

    cat = pd.DataFrame({"index": np.arange(len(movie_ids), dtype=np.int32), "movieId": movie_ids})
    cat = cat.merge(links[["movieId", "tmdbId"]], on="movieId", how="left")
    cat = cat.merge(movies[["movieId", "title", "genres"]], on="movieId", how="left")
    cat = cat.merge(stats, left_on="movieId", right_index=True, how="left")

    cat["shrunk_mean"] = (cat.n_ratings * cat.mean_rating + prior * global_mean) / (
        cat.n_ratings + prior
    )

    # 13 titles have no "(YYYY)" suffix ("Moonlight", "The OA", ...). Nullable
    # integer column, not a crash and not a fake 0.
    cat["year"] = cat.title.str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")

    cat = cat[cat.tmdbId.notna()].copy()
    cat["tmdbId"] = cat.tmdbId.astype(np.int64)

    # One tmdbId (4912, "Confessions of a Dangerous Mind") maps to two movieIds.
    # Tie-break on rating count, as decided in notebook 01 cell 18: keep 6003
    # (15 ratings) over 144606 (1). Decide it here, once, rather than discovering
    # it as a duplicate row in an API response.
    before = len(cat)
    cat = cat.sort_values(["tmdbId", "n_ratings"], ascending=[True, False])
    cat = cat.drop_duplicates("tmdbId", keep="first")
    if before != len(cat):
        print(f"    dropped {before - len(cat)} duplicate tmdbId row(s), kept the higher-rated")

    cat = cat.sort_values("index").reset_index(drop=True)
    return cat[
        [
            "index",
            "movieId",
            "tmdbId",
            "title",
            "year",
            "genres",
            "n_ratings",
            "mean_rating",
            "shrunk_mean",
        ]
    ]


# =============================================================================
# 5. Save / inspect / report
# =============================================================================


def save(path, movie_ids, neighbor_idx, neighbor_sim, global_mean, args, n_users, n_ratings):
    """Write model.npz, params included.

    Saving the params inside the artifact is not bookkeeping for its own sake:
    weekend 3's sweep will leave several .npz files on disk and the file itself
    has to be able to say which is which. Same reason global_mean is in here --
    evaluate.py and the API must center with the exact value used at training
    time, not recompute something slightly different.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        movie_ids=movie_ids.astype(np.int32),
        neighbor_idx=neighbor_idx,
        neighbor_sim=neighbor_sim,
        global_mean=np.float64(global_mean),
        lam=np.float64(args.lam),
        topk=np.int32(args.topk),
        center_m=np.float64(args.center_m),
        n_users=np.int32(n_users),
        n_ratings=np.int32(n_ratings),
        built_at=np.array(datetime.now(timezone.utc).isoformat(timespec="seconds")),
    )


def print_neighbors(title_query, movie_ids, neighbor_idx, neighbor_sim, movies, ratings, n=10):
    """Print one movie's neighbours with genres and rating counts. The eyeball test."""
    meta = movies.set_index("movieId")
    counts = ratings.groupby("movieId").size()
    matches = movies[movies.title.str.contains(title_query, case=False, regex=False)]
    matches = matches[matches.movieId.isin(movie_ids)]
    if matches.empty:
        print(f"  no rated movie matching {title_query!r}")
        return
    movie_id = int(matches.iloc[0].movieId)
    pos = int(np.searchsorted(movie_ids, movie_id))

    print(f"\n  {meta.loc[movie_id, 'title']}  [{meta.loc[movie_id, 'genres']}]  n={counts[movie_id]}")
    for j, sim in list(zip(neighbor_idx[pos], neighbor_sim[pos]))[:n]:
        mid = int(movie_ids[j])
        print(
            f"    {sim:.3f}  n={counts.get(mid, 0):>4}  "
            f"{meta.loc[mid, 'title'][:46]:<46} {meta.loc[mid, 'genres'][:38]}"
        )


def peak_mb():
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / 1e6 if sys.platform == "darwin" else rss / 1e3  # macOS bytes, Linux KB


# =============================================================================
# main
# =============================================================================


def main():
    p = argparse.ArgumentParser(description="Build the item-item similarity model.")
    p.add_argument("--lam", type=float, default=LAMBDA, help=f"shrinkage strength (default {LAMBDA})")
    p.add_argument("--topk", type=int, default=TOPK, help=f"neighbours per movie (default {TOPK})")
    p.add_argument("--center-m", type=float, default=CENTER_M, dest="center_m",
                   help=f"prior strength for the shrunk user mean (default {CENTER_M})")
    p.add_argument("--block", type=int, default=BLOCK, help=f"movie columns per block (default {BLOCK})")
    p.add_argument("--out", type=Path, default=ARTIFACT_DIR, help="artifact directory")
    p.add_argument("--inspect", metavar="TITLE", help="print an existing model's neighbours and exit")
    args = p.parse_args()

    ratings, movies, links = load_data()
    model_path = args.out / "model.npz"

    if args.inspect:
        if not model_path.exists():
            sys.exit(f"no model at {model_path} -- run `python train.py` first")
        d = np.load(model_path, allow_pickle=False)
        print_neighbors(args.inspect, d["movie_ids"], d["neighbor_idx"], d["neighbor_sim"],
                        movies, ratings, n=20)
        return

    t0 = time.time()
    movie_ids, user_ids, row_of, col_of = build_index(ratings)
    n_users, n_items = len(user_ids), len(movie_ids)
    global_mean = float(ratings.rating.mean())
    print(f"loaded: {len(ratings):,} ratings, {n_users:,} users, {n_items:,} rated movies")
    print(f"        global mean {global_mean:.4f}, density {len(ratings) / (n_users * n_items):.2%}")

    R, B = build_matrices(ratings, n_users, n_items, row_of, col_of)

    print("centering:")
    _check_shrunk_mean()
    agg = ratings.groupby("userId").rating.agg(["sum", "count"]).reindex(user_ids)
    mu = shrunk_user_mean(agg["sum"].to_numpy(np.float64), agg["count"].to_numpy(np.float64),
                          args.center_m, global_mean)
    plain = agg["sum"].to_numpy(np.float64) / agg["count"].to_numpy(np.float64)
    print(f"    [#1 ok] user means span {mu.min():.2f}..{mu.max():.2f} around a global "
          f"{global_mean:.3f}; the m={args.center_m:g} prior moved them a median of "
          f"{np.median(np.abs(mu - plain)):.3f}")

    C = center_ratings(R, mu.astype(np.float32))
    _check_centering(R, C)
    print(f"    [#2 ok] {100.0 * (C.data < 0).mean():.1f}% of centered ratings are negative "
          f"(0% before centering -- this is what lets the model model dislike)")

    Cn, norms = normalize_columns(C)
    _check_normalisation(Cn, norms)
    print(f"    [#3 ok] {n_items:,} movie columns scaled to unit length -- a dot product "
          f"between two of them is now a cosine")

    print(f"similarity: {n_items:,} movies in blocks of {args.block}, lambda={args.lam:g}, k={args.topk}")
    t_sim = time.time()
    neighbor_idx, neighbor_sim = top_neighbors(Cn, B, args.lam, args.topk, args.block)
    _check_neighbors(neighbor_idx, neighbor_sim)
    print(f"    [#4 ok] built in {time.time() - t_sim:.1f}s, "
          f"strongest similarity {neighbor_sim.max():.3f}")

    unreachable = int((neighbor_sim[:, 0] <= 0).sum())
    print(f"    movies with no positive neighbour: {unreachable} "
          f"({unreachable / n_items:.1%} -- these can never be recommended)")

    print("catalog:")
    catalog = build_catalog(ratings, movies, links, movie_ids, global_mean)

    args.out.mkdir(parents=True, exist_ok=True)
    save(model_path, movie_ids, neighbor_idx, neighbor_sim, global_mean, args, n_users, len(ratings))
    catalog_path = args.out / "catalog.csv"
    catalog.to_csv(catalog_path, index=False)

    print(f"\nwrote {model_path}   ({model_path.stat().st_size / 1e6:.1f} MB, "
          f"neighbours {neighbor_idx.shape})")
    print(f"wrote {catalog_path} ({len(catalog):,} serveable movies of {n_items:,} rated)")
    print(f"total {time.time() - t0:.1f}s, peak memory {peak_mb():.0f} MB")

    print("\nSANITY CHECK -- the gate for this weekend.")
    print("If Toy Story's neighbours are not mostly Animation/Children, STOP and fix it.")
    print("Usual causes, in order: centering skipped, n_co built from the centered")
    print("matrix instead of the binary one, or rows normalised instead of columns.")
    for title in ("Toy Story (1995)", "Star Wars: Episode IV", "Alien (1979)"):
        print_neighbors(title, movie_ids, neighbor_idx, neighbor_sim, movies, ratings, n=8)
    print('\nMore: python train.py --inspect "Blade Runner"')
    print("Then: jupyter lab notebooks/02_inspect_model.ipynb")


if __name__ == "__main__":
    try:
        main()
    except NotImplementedError as todo:
        sys.exit(
            f"\n{todo}\n\n"
            "Open train.py, find that TODO(you) block and write the line(s) sketched in\n"
            "the comment above it. Run `python train.py` again -- the check immediately\n"
            "after it will tell you if the formula came out wrong, and how."
        )
