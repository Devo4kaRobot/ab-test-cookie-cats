import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abtest import bayes  # noqa: E402
from abtest import stats as S  # noqa: E402

# Cookie Cats, retention_7: 8 502 из 44 700 против 8 279 из 45 489
XA, NA, XB, NB = 8502, 44700, 8279, 45489


def test_prior_from_mean_and_posterior():
    a, b = bayes.prior_from_mean(0.19, 100)
    assert (a, b) == pytest.approx((19, 81))
    post = bayes.posterior(30, 100, (a, b))
    assert post.mean() == pytest.approx(49 / 200)
    for bad in [(1.2, 100), (0.19, 0)]:
        try:
            bayes.prior_from_mean(*bad)
        except ValueError:
            continue
        raise AssertionError(f"prior_from_mean{bad} должна падать с ValueError")


def test_symmetry_when_groups_swapped():
    r = bayes.compare_proportions(4000, 20000, 4300, 20000)
    s = bayes.compare_proportions(4300, 20000, 4000, 20000)
    assert r["diff"] == pytest.approx(-s["diff"], abs=2e-4)
    assert r["prob_b_greater"] == pytest.approx(s["prob_b_less"], abs=0.005)
    assert r["loss_choose_a"] == pytest.approx(s["loss_choose_b"], abs=2e-4)
    assert r["ci"][0] == pytest.approx(-s["ci"][1], abs=5e-4)


def test_identical_groups_are_a_coin_flip():
    r = bayes.compare_proportions(1000, 5000, 1000, 5000)
    assert r["prob_b_less"] == pytest.approx(0.5, abs=0.01)
    assert r["prob_b_less"] + r["prob_b_greater"] == pytest.approx(1.0)
    assert r["loss_choose_a"] == pytest.approx(r["loss_choose_b"], rel=0.02)


def test_prob_b_less_matches_one_sided_p_for_large_n():
    r = bayes.compare_proportions(XA, NA, XB, NB)
    w = S.two_proportion_test(XA, NA, XB, NB)
    p_one_sided = stats.norm.cdf(w["z"])  # H1: «B хуже A»
    assert r["prob_b_less"] == pytest.approx(1 - p_one_sided, abs=0.001)
    assert r["prob_b_less"] > 0.99


def test_credible_interval_close_to_wald():
    r = bayes.compare_proportions(XA, NA, XB, NB)
    w = S.two_proportion_test(XA, NA, XB, NB)
    assert r["ci"][0] == pytest.approx(w["ci"][0], abs=3e-4)
    assert r["ci"][1] == pytest.approx(w["ci"][1], abs=3e-4)
    assert r["diff"] == pytest.approx(w["diff"], abs=1e-4)


def test_expected_loss_properties():
    r = bayes.compare_proportions(XA, NA, XB, NB)
    # оба ≥ 0, разность потерь = разность средних: E[max(d,0)] − E[max(−d,0)] = E[d]
    assert r["loss_choose_a"] >= 0 and r["loss_choose_b"] >= 0
    assert r["loss_choose_a"] - r["loss_choose_b"] == pytest.approx(r["diff"], abs=1e-12)
    # B заметно хуже: выбирать B дорого, выбирать A почти бесплатно
    assert r["loss_choose_b"] > 100 * r["loss_choose_a"]
    assert r["loss_choose_b"] >= -r["diff"]
    # с ростом выборки потеря правильного выбора стремится к нулю
    small = bayes.compare_proportions(190, 1000, 182, 1000)
    big = bayes.compare_proportions(19000, 100000, 18200, 100000)
    assert big["loss_choose_a"] < small["loss_choose_a"]


def test_threshold_probabilities_monotone_and_sum_to_one():
    prev_below = 1.0
    for t in [0.0, 0.0025, 0.005, 0.01]:
        r = bayes.compare_proportions(XA, NA, XB, NB, threshold=t)
        assert r["prob_below"] + r["prob_rope"] + r["prob_above"] == pytest.approx(1.0)
        assert r["prob_below"] <= prev_below
        prev_below = r["prob_below"]
    assert bayes.compare_proportions(XA, NA, XB, NB, threshold=0)["prob_below"] == pytest.approx(
        bayes.compare_proportions(XA, NA, XB, NB)["prob_b_less"])


def test_prior_barely_matters_for_large_n_but_matters_for_small():
    weak = bayes.prior_from_mean(0.19, 100)
    big_u = bayes.compare_proportions(XA, NA, XB, NB)
    big_w = bayes.compare_proportions(XA, NA, XB, NB, prior=weak)
    assert big_w["diff"] == pytest.approx(big_u["diff"], abs=5e-5)
    small_u = bayes.compare_proportions(10, 40, 4, 40)
    small_w = bayes.compare_proportions(10, 40, 4, 40, prior=weak)
    assert abs(small_w["diff"] - small_u["diff"]) > 0.01


def test_reproducible_with_seed():
    r1 = bayes.compare_proportions(XA, NA, XB, NB, seed=7)
    r2 = bayes.compare_proportions(XA, NA, XB, NB, seed=7)
    assert r1["ci"] == r2["ci"] and r1["prob_b_less"] == r2["prob_b_less"]


def test_per_installs():
    out = bayes.per_installs(-0.0082, (-0.0133, -0.0031))
    assert out["players"] == pytest.approx(-820)
    assert out["ci"] == pytest.approx((-1330, -310))
    assert bayes.per_installs(0.01, installs=1000)["players"] == pytest.approx(10)
    assert "ci" not in bayes.per_installs(0.01)
