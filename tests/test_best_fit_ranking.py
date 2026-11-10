import numpy as np

from surrDAMH.post_processing import rank_best_fit_candidates


def test_rank_best_fit_candidates_supports_all_modes():
    misfits = np.array([0.5, 0.3, 0.1])
    posteriors = np.array([10.0, 20.0, 5.0])
    likelihoods = np.array([5.0, 1.0, 2.0])

    assert rank_best_fit_candidates(misfits, posteriors, likelihoods, mode="l2").tolist() == [2, 1, 0]
    assert rank_best_fit_candidates(misfits, posteriors, likelihoods, mode="posterior").tolist() == [1, 0, 2]
    assert rank_best_fit_candidates(misfits, posteriors, likelihoods, mode="likelihood").tolist() == [0, 2, 1]


def test_rank_best_fit_candidates_rejects_unknown_mode():
    try:
        rank_best_fit_candidates(np.array([1.0]), np.array([1.0]), np.array([1.0]), mode="unknown")
    except ValueError as exc:
        assert "Unsupported best-fit ranking mode" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown mode")
