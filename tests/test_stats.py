import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abtest import stats as S  # noqa: E402


def test_srm_balanced_and_skewed():
    assert S.srm_test([5000, 5000])["p_value"] > 0.99
    assert S.srm_test([5000, 5400])["p_value"] < 0.001
    # Cookie Cats: 44 700 против 45 489
    assert S.srm_test([44700, 45489])["p_value"] == pytest.approx(0.0086, abs=0.0005)


def test_two_proportion_matches_manual():
    r = S.two_proportion_test(100, 1000, 130, 1000)
    p = 230 / 2000
    z = 0.03 / np.sqrt(p * (1 - p) * 2 / 1000)
    assert r["z"] == pytest.approx(z)
    assert r["ci"][0] < 0.03 < r["ci"][1]


def test_bootstrap_proportion_close_to_wald():
    w = S.two_proportion_test(4000, 20000, 4300, 20000)
    b = S.bootstrap_proportion_diff(4000, 20000, 4300, 20000, n_boot=20000)
    assert b["ci"][0] == pytest.approx(w["ci"][0], abs=0.002)
    assert b["ci"][1] == pytest.approx(w["ci"][1], abs=0.002)


def test_mann_whitney_prob_superiority():
    a = np.arange(100); b = a + 1000
    assert S.mann_whitney(a, b)["prob_superiority"] == 1.0
    assert S.mann_whitney(a, a)["prob_superiority"] == pytest.approx(0.5)


def test_holm_monotone_and_known_case():
    out = S.holm([0.01, 0.04, 0.03])
    assert [o["p_holm"] for o in out] == pytest.approx([0.03, 0.06, 0.06])
    assert [o["reject"] for o in out] == [True, False, False]


def test_power_sample_size_consistency():
    n_a, _ = S.sample_size_proportion(0.19, 0.01)
    assert S.power_proportion(0.19, 0.01, n_a) == pytest.approx(0.80, abs=0.01)
    # MDE при n из Cookie Cats — около 0.73 п.п.
    assert S.mde_proportion(0.19, 45000) == pytest.approx(0.0073, abs=0.0002)


def test_bootstrap_stat_diff_shape_and_center():
    rng = np.random.default_rng(1)
    a = rng.normal(0, 1, 3000); b = rng.normal(0.5, 1, 3000)
    r = S.bootstrap_stat_diff(a, b, np.mean, n_boot=400)
    assert r["ci"][0] < 0.5 < r["ci"][1] and len(r["samples"]) == 400


def test_cuped_reduces_variance_keeps_mean():
    rng = np.random.default_rng(2)
    x = rng.normal(10, 3, 5000); y = 2 + 0.8 * x + rng.normal(0, 2, 5000)
    y2, theta = S.cuped(y, x)
    assert theta == pytest.approx(0.8, abs=0.05)
    assert y2.mean() == pytest.approx(y.mean())
    assert y2.var() < 0.5 * y.var()


def test_poststratified_diff_one_stratum_and_imbalance():
    rng = np.random.default_rng(0)
    a, b = rng.binomial(1, 0.3, 4000), rng.binomial(1, 0.3, 4000)
    one = S.poststratified_diff(a, np.zeros_like(a), b, np.zeros_like(b))
    assert one["diff"] == pytest.approx(b.mean() - a.mean())
    # группы с разным составом: в B больше «слабого» страта, эффекта внутри стратов нет
    s_a = np.r_[np.zeros(3000), np.ones(1000)]
    s_b = np.r_[np.zeros(3500), np.ones(500)]
    y_a = np.r_[rng.binomial(1, 0.1, 3000), rng.binomial(1, 0.6, 1000)]
    y_b = np.r_[rng.binomial(1, 0.1, 3500), rng.binomial(1, 0.6, 500)]
    raw = y_b.mean() - y_a.mean()
    adj = S.poststratified_diff(y_a, s_a, y_b, s_b)
    assert raw < -0.04                      # прямое сравнение видит «эффект» из-за состава
    assert abs(adj["diff"]) < 0.03          # поправка его убирает
    assert adj["ci"][0] < adj["diff"] < adj["ci"][1]
