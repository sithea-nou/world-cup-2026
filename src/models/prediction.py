import numpy as np


def predict_with_draw_threshold(
    model,
    X: np.ndarray,
    draw_threshold: float = 0.85,
) -> np.ndarray:
    probs = model.predict_proba(X)
    preds = np.argmax(probs, axis=1)

    draw_probs = probs[:, 1]
    max_non_draw = np.maximum(probs[:, 0], probs[:, 2])
    draw_mask = (draw_probs / max_non_draw) > draw_threshold
    preds[draw_mask] = 1

    return preds


def predict_proba_with_draw_boost(
    model,
    X: np.ndarray,
    draw_boost: float = 1.0,
) -> np.ndarray:
    if draw_boost == 1.0:
        return model.predict_proba(X)

    probs = model.predict_proba(X).copy()
    probs[:, 1] *= draw_boost
    probs /= probs.sum(axis=1, keepdims=True)
    return probs
