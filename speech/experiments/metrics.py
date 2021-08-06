from sklearn.metrics import recall_score

import numpy as np
from transformers import EvalPrediction


def unweighted_average_recall(labels, preds) -> float:
    """
    Calculate Unweighted Average Recall Metric

    Args:
        labels (array): Ground Truth
        preds (array): Class predictions

    Returns:
        (float): UAR
    """
    return recall_score(labels, preds, average="macro")


def compute_metrics(p: EvalPrediction, is_regression=False) -> dict:
    """
    Compute evaluation metrics

    Args:
        p (EvalPrediction): Transfomer output
        is_regression (bool, optional):Defaults to False.

    Returns:
        dict: Metrics
    """
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {
            "accuracy": (preds == p.label_ids).astype(np.float32).mean().item(),
            "uar": unweighted_average_recall(p.label_ids, preds),
        }
