"""
The CCG v2 law of contextual drag.

With K drafts in context, T of them correct, model m answers correctly with probability

    logit p_m(T) = c_m + w_m * l(T),    l(T) = log((T + 1/2) / (K - T + 1/2)).

Equivalently, the odds of a correct answer are e^c times the draft odds (T + 1/2)/(K - T + 1/2) to the power w; the 1/2 is the Jeffreys pseudo-count. c_m is the log-odds of a
correct answer when the drafts are tied (l(K/2) = 0); w_m is the model's sensitivity to the draft odds.
Both are fitted by weighted least squares on the log-odds of the calibration rungs (delta-method
weights p(1 - p); probabilities clipped to half a response of the 8N in a rung), and the law then
predicts the other rungs. No parameter comes from another model and no constant is tuned.

The input is a ladder file of ``cards/nodes/cd_ladder.py`` (schema
``contextual_drag.route_model.ladder/1``), so every ladder measured by either version of the card
is scored as it is.

Reference: "Contextual Drag as a Log-Count Vote", Team Princeton-2, November 11, 2026
(aiq-ccg-theory-overleaf, Contextual_Drag_1111.tex).
"""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Dict, Optional, Sequence, Tuple

__all__ = ['ell', 'rung_accuracy', 'fit', 'predict', 'line_baseline']

PSEUDO = 0.5


def ell(T: float, K: int = 4) -> float:
    """
    Log of the draft odds (T + 1/2)/(K - T + 1/2); antisymmetric about the tie.

    Example:
        >>> ell(2), round(ell(0) + ell(4), 12)
        (0.0, 0.0)
        >>> round(math.exp(ell(4)), 6)
        9.0
    """
    return math.log((T + PSEUDO) / (K - T + PSEUDO))


def rung_accuracy(payload: dict, weights: Optional[Dict[str, float]] = None) -> Tuple[Dict[int, float], float]:
    """
    Accuracy at every rung: per problem over orders, then averaged over problems.

    Args:
        payload: a ladder file's contents.
        weights: problem id -> multiplicity, for the problem bootstrap; unset weighs each once.

    Returns:
        ({T: accuracy}, effective number of problems)
    """
    K = int(payload.get('K') or 4)
    ids = [str(p['id']) for p in payload['problems']]
    weights = weights if weights is not None else {i: 1.0 for i in ids}
    cell = defaultdict(lambda: [0, 0])
    for c in payload['contexts']:
        key = (str(c['problem_id']), int(c['T']))
        cell[key][0] += c['n']
        cell[key][1] += c['k']
    acc = {}
    for T in range(K + 1):
        num = den = 0.0
        for i in ids:
            n, k = cell.get((i, T), (0, 0))
            if n and weights.get(i, 0.0):
                num += weights[i] * k / n
                den += weights[i]
        acc[T] = num / den if den else float('nan')
    return acc, sum(weights.get(i, 0.0) for i in ids)


def fit(acc: Dict[int, float], rungs: Sequence[int], n_problems: float, K: int = 4,
        n_orders: int = 8) -> Tuple[float, float]:
    """
    Weighted least squares of logit p on l(T) over the calibration rungs.

    Example:
        >>> c, w = 0.7, 1.3
        >>> acc = {T: 1 / (1 + math.exp(-(c + w * ell(T)))) for T in range(5)}
        >>> [round(x, 6) for x in fit(acc, (0, 1, 2), 20)]
        [0.7, 1.3]
    """
    eps = 0.5 / (n_orders * max(n_problems, 1.0))
    xs, ys, ws = [], [], []
    for T in rungs:
        p = min(max(acc[T], eps), 1 - eps)
        xs.append(ell(T, K))
        ys.append(math.log(p / (1 - p)))
        ws.append(p * (1 - p))
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 0:
        raise ValueError('calibration rungs must include at least two distinct T')
    slope = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    return my - slope * mx, slope


def predict(c: float, w: float, T: float, K: int = 4) -> float:
    return 1 / (1 + math.exp(-(c + w * ell(T, K))))


def line_baseline(acc: Dict[int, float], rungs: Sequence[int], T: int) -> float:
    """No-law baseline: the straight line through the two calibration rungs nearest T, capped to [0, 1]."""
    a, b = sorted(rungs, key=lambda r: (abs(r - T), r))[:2]
    a, b = sorted((a, b))
    slope = (acc[b] - acc[a]) / (b - a)
    return min(1.0, max(0.0, acc[b] + slope * (T - b)))
