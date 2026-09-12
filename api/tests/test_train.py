"""Tests for the weekend-2 model.

Run from api/:   pytest tests/ -q

The first test is the build plan's "cosine similarity by hand on paper" study
item, made permanent. Everything below it is a property test that holds for any
input -- those catch the bugs that produce plausible-looking-but-wrong neighbour
lists, which is the failure mode you cannot spot by eyeballing.
"""

import numpy as np
import pandas as pd
import pytest
from scipy.sparse import csr_matrix

from train import (
    apply_shrinkage,
    build_catalog,
    build_matrices,
    build_index,
    center_ratings,
    normalize_columns,
    shrunk_user_mean,
    top_neighbors,
)

# =============================================================================
# The toy case -- do this one on paper first
# =============================================================================
#
#   R (3 users x 3 movies, "." = not rated)      mu (each user's own mean)
#
#            M0   M1   M2                          u0: (5+3+4)/3 = 4
#     u0      5    3    4                          u1: (3+5)/2   = 4
#     u1      3    5    .                          u2: (4+2)/2   = 3
#     u2      4    .    2
#
#   Work out, with pencil:
#     1. C, the centered matrix (subtract each row's mu from that row's ratings).
#        One cell lands on exactly 0.0. Notice which, and keep it -- u0 DID rate
#        M2, they just rated it at their own average.
#     2. Each column's L2 norm.
#     3. cos(Mi, Mj) = (dot product of the two centered columns, over the users
#        who rated BOTH) / (norm_i * norm_j).
#     4. n_co(Mi, Mj) = how many users rated both. The (M1, M2) pair is the
#        interesting one: u0 rated both, so n_co = 1, even though u0's centered
#        M2 value is 0.0. This is why the binary matrix is built from the
#        sparsity pattern and not from `C != 0` -- get that wrong and this pair's
#        evidence count silently drops to 0.
#     5. shrunk = cos * n_co / (n_co + LAMBDA), with LAMBDA = 1.0 below.
#
#   Then fill in EXPECTED_SHRUNK and run the test. If your paper and the code
#   disagree, one of them is wrong, and you want to find out which now rather
#   than in week 3 when a bad Recall@20 has three possible explanations.

TOY_LAMBDA = 1.0

# TODO(you): replace None with your three hand-derived numbers, e.g.
#   EXPECTED_SHRUNK = {("M0", "M1"): -0.5443, ("M0", "M2"): ..., ("M1", "M2"): ...}
# Until then this test reports as skipped rather than failing the suite.
EXPECTED_SHRUNK = None


def toy_matrices():
    """Return (R, B) for the 3x3 matrix drawn above."""
    rows = [0, 0, 0, 1, 1, 2, 2]
    cols = [0, 1, 2, 0, 1, 0, 2]
    vals = [5.0, 3.0, 4.0, 3.0, 5.0, 4.0, 2.0]
    R = csr_matrix((np.array(vals, np.float32), (rows, cols)), shape=(3, 3))
    B = csr_matrix((np.ones(len(vals), np.float32), (rows, cols)), shape=(3, 3))
    return R, B


def toy_shrunk_similarity():
    """Run the real pipeline on the toy matrix, with plain user means (m = 0)."""
    R, B = toy_matrices()
    mu = np.array([4.0, 4.0, 3.0], np.float32)  # each user's own mean
    C = center_ratings(R, mu)
    Cn, _ = normalize_columns(C)
    sim = (Cn.T @ Cn).toarray().astype(np.float64)
    n_co = (B.T @ B).toarray().astype(np.float64)
    return apply_shrinkage(sim, n_co, TOY_LAMBDA)


def test_toy_similarity_matches_hand_derivation():
    if EXPECTED_SHRUNK is None:
        pytest.skip("TODO(you): fill in EXPECTED_SHRUNK from your paper derivation")
    S = toy_shrunk_similarity()
    name_to_col = {"M0": 0, "M1": 1, "M2": 2}
    for (a, b), expected in EXPECTED_SHRUNK.items():
        got = S[name_to_col[a], name_to_col[b]]
        assert got == pytest.approx(expected, abs=1e-3), (
            f"shrunk similarity({a}, {b}): code says {got:+.4f}, your paper says "
            f"{expected:+.4f}"
        )


# =============================================================================
# Centering
# =============================================================================


def test_centering_preserves_sparsity_structure():
    """The cell that centers to exactly 0.0 must survive as a stored value."""
    R, _ = toy_matrices()
    C = center_ratings(R, np.array([4.0, 4.0, 3.0], np.float32))
    assert C.nnz == R.nnz == 7
    assert (C.indices == R.indices).all()
    assert (C.indptr == R.indptr).all()
    assert (C.data == 0.0).sum() == 1, "u0's rating of M2 should center to exactly 0.0"


def test_co_rating_counts_survive_a_zero_centered_rating():
    """The (C != 0) trap, in three movies.

    u0 rated both M1 and M2, so they are co-rated once -- regardless of the fact
    that u0's centered M2 value is 0.0. Build the binary matrix from the rating
    pattern and this holds automatically.
    """
    R, B = toy_matrices()
    n_co = (B.T @ B).toarray()
    assert n_co[1, 2] == 1

    C = center_ratings(R, np.array([4.0, 4.0, 3.0], np.float32))
    from_values = (C != 0).astype(np.float32)
    assert (from_values.T @ from_values).toarray()[1, 2] == 0, (
        "if this ever stops being 0, the trap is gone and this test can go"
    )


def test_centering_makes_dislike_negative():
    R, _ = toy_matrices()
    assert (R.data < 0).sum() == 0, "raw ratings are all positive -- that is the problem"
    C = center_ratings(R, np.array([4.0, 4.0, 3.0], np.float32))
    assert (C.data < 0).sum() > 0


def test_shrunk_user_mean_interpolates_between_raw_and_global():
    global_mean = 3.5
    heavy = shrunk_user_mean(np.array([700.0]), np.array([200.0]), 5.0, global_mean)
    new_user = shrunk_user_mean(np.array([15.0]), np.array([3.0]), 5.0, global_mean)
    assert heavy[0] == pytest.approx(3.5, abs=0.05)  # 200 ratings: barely moves
    assert global_mean < new_user[0] < 5.0  # 3 x 5-star: pulled well off 5.0
    # m = 0 must reduce to the plain mean exactly
    plain = shrunk_user_mean(np.array([15.0]), np.array([3.0]), 0.0, global_mean)
    assert plain[0] == pytest.approx(5.0)


# =============================================================================
# Normalisation
# =============================================================================


def test_normalized_columns_have_unit_length():
    R, _ = toy_matrices()
    C = center_ratings(R, np.array([4.0, 4.0, 3.0], np.float32))
    Cn, norms = normalize_columns(C)
    lengths = np.sqrt(np.asarray(Cn.multiply(Cn).sum(axis=0)).ravel())
    assert np.allclose(lengths, 1.0, atol=1e-5)
    assert (norms > 0).all()


def test_zero_norm_column_yields_zeros_not_nan():
    """A movie rated only by users who rated it at their own mean."""
    R = csr_matrix((np.array([4.0, 4.0], np.float32), ([0, 1], [0, 0])), shape=(2, 2))
    C = center_ratings(R, np.array([4.0, 4.0], np.float32))
    Cn, norms = normalize_columns(C)
    assert not np.isnan(Cn.toarray()).any(), "missing the norms[norms == 0] = 1.0 guard"
    assert norms[0] == 1.0
    assert (Cn.toarray()[:, 0] == 0).all()


# =============================================================================
# Shrinkage
# =============================================================================


def test_shrinkage_never_increases_a_similarity():
    sim = np.array([[0.9, -0.4], [0.2, 1.0]])
    n_co = np.array([[2.0, 50.0], [1.0, 300.0]])
    out = apply_shrinkage(sim.copy(), n_co, 10.0)
    assert (np.abs(out) <= np.abs(sim) + 1e-9).all()


def test_shrinkage_is_monotone_in_evidence():
    """More co-raters must keep more of the similarity, and sign must not flip."""
    sim = np.full((1, 4), 0.8)
    n_co = np.array([[1.0, 5.0, 50.0, 500.0]])
    out = apply_shrinkage(sim.copy(), n_co, 10.0).ravel()
    assert (np.diff(out) > 0).all()
    assert out[0] < 0.15 and out[-1] > 0.75
    neg = apply_shrinkage(np.array([[-0.8]]), np.array([[2.0]]), 10.0)
    assert neg[0, 0] < 0, "shrinkage scales toward zero, it must not flip the sign"


def test_shrinkage_with_lambda_zero_is_a_no_op():
    sim = np.array([[0.3, -0.7]])
    assert np.allclose(apply_shrinkage(sim.copy(), np.array([[2.0, 9.0]]), 0.0), sim)


# =============================================================================
# Neighbour selection
# =============================================================================


def random_model(n_users=40, n_items=25, density=0.3, seed=0):
    rng = np.random.default_rng(seed)
    mask = rng.random((n_users, n_items)) < density
    rows, cols = np.nonzero(mask)
    vals = rng.integers(1, 11, size=len(rows)) / 2.0
    R = csr_matrix((vals.astype(np.float32), (rows, cols)), shape=(n_users, n_items))
    B = csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(n_users, n_items))
    counts = np.asarray(B.sum(axis=1)).ravel()
    sums = np.asarray(R.sum(axis=1)).ravel()
    mu = shrunk_user_mean(sums, np.maximum(counts, 1), 5.0, 3.5).astype(np.float32)
    Cn, _ = normalize_columns(center_ratings(R, mu))
    return Cn, B


def test_neighbours_exclude_self_and_are_sorted():
    Cn, B = random_model()
    idx, sim = top_neighbors(Cn, B, lam=10.0, topk=5, block=7)
    assert not np.isnan(sim).any() and not np.isinf(sim).any()
    assert not (idx == np.arange(len(idx))[:, None]).any()
    assert (np.diff(sim, axis=1) <= 1e-6).all()


def test_blocked_result_matches_brute_force():
    """The blocked loop must give the same answer as the obvious dense version."""
    Cn, B = random_model()
    lam, topk = 10.0, 5
    idx, sim = top_neighbors(Cn, B, lam=lam, topk=topk, block=7)

    S = (Cn.T @ Cn).toarray().astype(np.float32)
    S = apply_shrinkage(S, (B.T @ B).toarray().astype(np.float32), lam)
    np.fill_diagonal(S, -np.inf)
    expected = -np.sort(-S, axis=1)[:, :topk]

    assert np.allclose(sim, expected, atol=1e-5)
    assert np.allclose(np.take_along_axis(S, idx, axis=1), sim, atol=1e-5)


@pytest.mark.parametrize("block", [1, 3, 7, 25, 1000])
def test_block_size_does_not_change_the_model(block):
    """Catches off-by-one errors in the block offset -- the self-exclusion line
    has to know that row r of a slab is really movie `start + r`."""
    Cn, B = random_model()
    ref_idx, ref_sim = top_neighbors(Cn, B, lam=10.0, topk=5, block=25)
    idx, sim = top_neighbors(Cn, B, lam=10.0, topk=5, block=block)
    assert np.allclose(sim, ref_sim, atol=1e-6)
    assert (idx == ref_idx).all()


# =============================================================================
# Catalog
# =============================================================================


def catalog_fixture():
    ratings = pd.DataFrame(
        {
            "userId": [1, 1, 1, 2, 2, 3],
            "movieId": [10, 20, 30, 10, 20, 10],
            "rating": [5.0, 4.0, 3.0, 4.0, 2.0, 5.0],
        }
    )
    movies = pd.DataFrame(
        {
            "movieId": [10, 20, 30],
            "title": ["Popular One (1999)", "Rare One (1999)", "Moonlight"],
            "genres": ["Drama", "Drama", "Drama"],
        }
    )
    # 10 and 20 both point at tmdb 777; 10 has more ratings and must win.
    links = pd.DataFrame({"movieId": [10, 20, 30], "tmdbId": [777.0, 777.0, 888.0]})
    movie_ids = np.array([10, 20, 30])
    return build_catalog(ratings, movies, links, movie_ids, global_mean=3.5, prior=20.0)


def test_duplicate_tmdb_id_keeps_the_more_rated_movie():
    cat = catalog_fixture()
    assert cat.tmdbId.duplicated().sum() == 0
    assert 10 in set(cat.movieId), "movieId 10 has 3 ratings and should be kept"
    assert 20 not in set(cat.movieId), "movieId 20 has 2 ratings and should be dropped"


def test_catalog_index_points_into_the_model_arrays():
    cat = catalog_fixture()
    assert cat["index"].is_monotonic_increasing
    assert set(cat["index"]) <= {0, 1, 2}
    assert cat.loc[cat.movieId == 30, "index"].iloc[0] == 2


def test_missing_year_is_null_not_a_crash():
    cat = catalog_fixture()
    assert pd.isna(cat.loc[cat.movieId == 30, "year"].iloc[0])
    assert cat.loc[cat.movieId == 10, "year"].iloc[0] == 1999


def test_shrunk_mean_pulls_low_count_movies_toward_the_global_mean():
    """The popularity baseline. A movie with one 5.0 must not outrank a good one."""
    cat = catalog_fixture().set_index("movieId")
    assert cat.loc[30, "mean_rating"] == 3.0
    assert cat.loc[30, "shrunk_mean"] == pytest.approx((1 * 3.0 + 20 * 3.5) / 21, abs=1e-6)
    assert cat.loc[30, "shrunk_mean"] > cat.loc[30, "mean_rating"]
