"""Байесовское сравнение двух долей (модель Бета-Биномиальная) на numpy/scipy.

Конверсия p каждой группы получает априорное Beta(a, b); после x успехов из n апостериорное
распределение — Beta(a + x, b + n − x). Всё, что зависит от обеих групп (разность, вероятность
«B хуже A», ожидаемые потери), считается методом Монте-Карло по выборкам из двух апостериорных.
Разность везде B − A, как в abtest.stats.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats

UNIFORM = (1.0, 1.0)


def prior_from_mean(mean: float, strength: float) -> tuple[float, float]:
    """Априорное Beta с заданным средним и «силой» — числом условных игроков, которое оно стоит
    (a + b = strength). Например, prior_from_mean(0.19, 100) — как будто заранее видели 100 игроков
    с удержанием 19%."""
    if not 0 < mean < 1 or strength <= 0:
        raise ValueError("mean должно быть в (0, 1), strength > 0")
    return mean * strength, (1 - mean) * strength


def posterior(x: int, n: int, prior: Sequence[float] = UNIFORM):
    """Апостериорное распределение доли: scipy-объект Beta(a + x, b + n − x) (pdf, ppf, mean …)."""
    a, b = prior
    return stats.beta(a + x, b + n - x)


def compare_proportions(x_a: int, n_a: int, x_b: int, n_b: int, prior: Sequence[float] = UNIFORM,
                        prior_b: Sequence[float] | None = None, threshold: float | None = None,
                        alpha: float = 0.05, n_draws: int = 200_000, seed: int = 0) -> dict:
    """Сравнение долей B и A по апостериорным распределениям.

    prior — Beta(a, b) для обеих групп (по умолчанию равномерное); prior_b — отдельное для B, если нужно.
    threshold — порог практической значимости в долях (0.0025 = 0,25 п.п.): дополнительно считаются
    P(Δ < −threshold), P(|Δ| ≤ threshold) и P(Δ > threshold).

    Возвращает:
      p_a, p_b — апостериорные средние; diff — среднее Δ = p_B − p_A; ci — равнохвостый
      (1 − alpha)-интервал Δ; rel_lift, rel_ci — то же для p_B / p_A − 1;
      prob_b_less, prob_b_greater — P(p_B < p_A), P(p_B > p_A);
      loss_choose_a = E[max(p_B − p_A, 0)] — сколько в среднем теряем, выбрав A, если лучше B;
      loss_choose_b = E[max(p_A − p_B, 0)] — то же при выборе B (всё в долях, не в п.п.);
      samples — выборка Δ (для графиков).
    """
    prior_b = prior if prior_b is None else prior_b
    rng = np.random.default_rng(seed)
    pa = rng.beta(prior[0] + x_a, prior[1] + n_a - x_a, n_draws)
    pb = rng.beta(prior_b[0] + x_b, prior_b[1] + n_b - x_b, n_draws)
    d = pb - pa
    rel = pb / pa - 1
    q = [100 * alpha / 2, 100 * (1 - alpha / 2)]
    lo, hi = np.percentile(d, q)
    rlo, rhi = np.percentile(rel, q)
    out = {
        "p_a": float(posterior(x_a, n_a, prior).mean()), "p_b": float(posterior(x_b, n_b, prior_b).mean()),
        "diff": float(d.mean()), "ci": (float(lo), float(hi)),
        "rel_lift": float(rel.mean()), "rel_ci": (float(rlo), float(rhi)),
        "prob_b_less": float((d < 0).mean()), "prob_b_greater": float((d > 0).mean()),
        "loss_choose_a": float(np.maximum(d, 0).mean()), "loss_choose_b": float(np.maximum(-d, 0).mean()),
        "samples": d,
    }
    if threshold is not None:
        t = abs(threshold)
        out.update({"threshold": t, "prob_below": float((d < -t).mean()),
                    "prob_rope": float((np.abs(d) <= t).mean()), "prob_above": float((d > t).mean())})
    return out


def per_installs(diff: float, ci: Sequence[float] | None = None, installs: int = 100_000) -> dict:
    """Перевод абсолютной разности долей в игроков на `installs` установок:
    −0,0082 → −820 удержанных игроков на 100 000. Интервал переводится так же."""
    out = {"installs": installs, "players": diff * installs}
    if ci is not None:
        out["ci"] = (ci[0] * installs, ci[1] * installs)
    return out
