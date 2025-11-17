# baselines/classical_ml.py
import argparse, os, json
from pathlib import Path
import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score


def load_split(pt_path):
    pkg = torch.load(pt_path, map_location="cpu")
    X_m = pkg["X_metrics"].numpy()   # [N, T, Fm]
    X_e = pkg["X_events"].numpy()    # [N, T, Fe]
    y   = pkg["y"].numpy().astype(np.int64)  # [N]
    meta = {"features": pkg.get("features"), "event_features": pkg.get("event_features"), "T": pkg.get("T")}
    return X_m, X_e, y, meta


def _safe_roc_auc(y_true, y_prob):
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _slopes_over_time(arr_2d):
    """
    arr_2d: [N, T] -> slope per row using time index 0..T-1 (least-squares)
    """
    N, T = arr_2d.shape
    t = np.arange(T)
    t_mean = t.mean()
    denom = np.sum((t - t_mean) ** 2) + 1e-9
    x_mean = arr_2d.mean(axis=1, keepdims=True)
    num = np.sum((t - t_mean) * (arr_2d - x_mean), axis=1)
    return (num / denom).astype(np.float32)  # [N]


def _aggregate_windows(X_m, X_e):
    """
    X_m: [N, T, Fm]  metrics
    X_e: [N, T, Fe]  event features (e.g., fail_burst)
    Returns: [N, D] flat feature matrix with mean/std/last/slope per channel.
    """
    def agg_block(X):  # [N, T, F]
        N, T, F = X.shape
        mean = X.mean(axis=1)                     # [N,F]
        std  = X.std(axis=1)                      # [N,F]
        last = X[:, -1, :]                        # [N,F]
        # slope per feature:
        slopes = np.stack([_slopes_over_time(X[:, :, j]) for j in range(F)], axis=1)  # [N,F]
        return np.concatenate([mean, std, last, slopes], axis=1)  # [N, 4F]

    A_m = agg_block(X_m)
    A_e = agg_block(X_e)
    return np.concatenate([A_m, A_e], axis=1).astype(np.float32)


def _evaluate(y_true, y_prob, thr=0.5):
    pr = float(average_precision_score(y_true, y_prob))
    roc = _safe_roc_auc(y_true, y_prob)
    f1 = float(f1_score(y_true, (y_prob >= thr).astype(int)))
    return pr, roc, f1


def run_baselines(data_dir, out_csv, out_json=None, seed=1371):
    data_dir = Path(data_dir)
    # --- load splits
    Xm_tr, Xe_tr, y_tr, meta = load_split(data_dir / "train.pt")
    Xm_va, Xe_va, y_va, _    = load_split(data_dir / "val.pt")
    Xm_te, Xe_te, y_te, _    = load_split(data_dir / "test.pt")

    # --- aggregate windows to classic features
    Xtr = _aggregate_windows(Xm_tr, Xe_tr)
    Xva = _aggregate_windows(Xm_va, Xe_va)
    Xte = _aggregate_windows(Xm_te, Xe_te)

    # --- scale using train only
    scaler = StandardScaler().fit(Xtr)
    Ztr, Zva, Zte = scaler.transform(Xtr), scaler.transform(Xva), scaler.transform(Xte)

    results = []

    # ===== Logistic Regression =====
    lr = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=None,
        random_state=seed,
    )
    lr.fit(Ztr, y_tr)
    p_va = lr.predict_proba(Zva)[:, 1]
    p_te = lr.predict_proba(Zte)[:, 1]
    pr, roc, f1 = _evaluate(y_te, p_te)
    results.append({"model": "LogisticRegression", "pr_auc": pr, "roc_auc": roc, "f1": f1})

    # ===== Random Forest =====
    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(Ztr, y_tr)
    p_te = rf.predict_proba(Zte)[:, 1]
    pr, roc, f1 = _evaluate(y_te, p_te)
    results.append({"model": "RandomForest", "pr_auc": pr, "roc_auc": roc, "f1": f1})

    # --- write CSV
    header = "model,pr_auc,roc_auc,f1\n"
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(header)
        for r in results:
            f.write(f"{r['model']},{r['pr_auc']:.6f},{r['roc_auc']:.6f},{r['f1']:.6f}\n")

    if out_json:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump({"meta": meta, "results": results}, f, indent=2)

    print(f"Wrote {out_csv}")
    if out_json:
        print(f"Wrote {out_json}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with train.pt/val.pt/test.pt")
    ap.add_argument("--out",  required=True, help="CSV path for baseline results")
    ap.add_argument("--json", default=None, help="optional JSON path")
    ap.add_argument("--seed", type=int, default=1371)
    args = ap.parse_args()
    run_baselines(args.data, args.out, args.json, args.seed)


if __name__ == "__main__":
    main()
