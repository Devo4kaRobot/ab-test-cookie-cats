import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from abtest import sequential as Q  # noqa: E402


def test_obf_constant_k5_matches_tables():
    b = Q.boundaries(5, alpha=0.05, kind="obf")
    assert b["C"] == pytest.approx(2.040, abs=0.005)        # табличное значение 2,040 (Jennison & Turnbull)
    assert b["z"][-1] == pytest.approx(2.040, abs=0.005)
    assert b["z"][0] == pytest.approx(2.040 * np.sqrt(5), abs=0.01)


def test_pocock_constant_k5_matches_tables():
    b = Q.boundaries(5, alpha=0.05, kind="pocock")
    assert b["C"] == pytest.approx(2.413, abs=0.005)        # табличное значение 2,413
    assert np.allclose(b["z"], b["C"])                       # граница плоская


def test_obf_boundaries_decrease_and_end_above_fixed():
    for K in (2, 5, 20):
        z = Q.boundaries(K, kind="obf")["z"]
        assert np.all(np.diff(z) < 0)
        assert z[-1] > 1.96                                  # плата за промежуточные проверки


def test_single_look_is_ordinary_z_test():
    for kind in Q.KINDS:
        b = Q.boundaries(1, alpha=0.05, kind=kind)
        assert b["z"][0] == pytest.approx(1.959964, abs=1e-5)
        assert b["p_nominal"][0] == pytest.approx(0.05)


def test_type_one_error_close_to_alpha_on_fresh_null_paths():
    rng = np.random.default_rng(123)                        # другой seed, чем при калибровке
    K, n = 5, 100_000
    z = np.cumsum(rng.standard_normal((n, K)), axis=1) / np.sqrt(np.arange(1, K + 1))
    for kind in Q.KINDS:
        fpr = (Q.first_crossing(z, Q.boundaries(K, kind=kind)) > 0).mean()
        assert fpr == pytest.approx(0.05, abs=0.004)
    naive = (np.abs(z) > 1.96).any(axis=1).mean()           # подглядывание без поправки
    assert naive > 0.12


def test_first_crossing_and_sequential_test():
    c = np.array([4.0, 3.0, 2.0])
    z = np.array([[1.0, 3.5, 0.0],     # пересечение на 2-й проверке
                  [1.0, -2.5, -2.1],   # только на 3-й, в минус
                  [4.5, 0.0, 0.0],     # сразу на 1-й
                  [1.0, 1.0, 1.0]])    # не пересёк
    assert Q.first_crossing(z, c).tolist() == [2, 3, 1, 0]

    r = Q.sequential_test(z[1], c)
    assert r["stopped"] and r["look"] == 3 and r["direction"] == -1 and r["final"]
    assert r["boundary"] == pytest.approx(2.0)

    r = Q.sequential_test([1.0, 1.0], c)                     # прошли 2 из 3 проверок — продолжаем
    assert not r["stopped"] and r["look"] is None and not r["final"]


def test_input_validation():
    with pytest.raises(ValueError):
        Q.boundaries(5, kind="haybittle")
    with pytest.raises(ValueError):
        Q.sequential_test([1.0, 2.0, 3.0], np.array([3.0, 2.0]))
    with pytest.raises(ValueError):
        Q.first_crossing(np.zeros((2, 4)), np.ones(3))
