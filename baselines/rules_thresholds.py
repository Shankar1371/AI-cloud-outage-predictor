def rule_predict(fail_burst_series, thr):
    return (fail_burst_series > thr).astype(int)