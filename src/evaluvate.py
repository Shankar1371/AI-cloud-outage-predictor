import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def pr_auc(y_true, y_prob):
    return float(average_precision_score(y_true, y_prob))


def roc_auc(y_true, y_prob):
   return float(roc_auc_score(y_true, y_prob))


def f1_at(y_true, y_prob, thr=0.5):
    return float(f1_score(y_true, (y_prob>=thr).astype(int)))


def alerts_per_day(timestamps, y_prob, thr=0.5):
    mask = (y_prob >= thr)
    # naive: count alerts per calendar day
    days = np.array([ts.astype('datetime64[D]') for ts in timestamps])
    uniq = np.unique(days)
    return {str(d): int(mask[days==d].sum()) for d in uniq}


def lead_time_minutes(event_times, alert_times, horizon=15):
    """Given arrays of event anchors and alerts, compute minutes lead time distribution.
    This requires aligned indices; you will adapt after window creation."""
    pass