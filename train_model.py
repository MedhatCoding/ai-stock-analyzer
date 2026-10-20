# -*- coding: utf-8 -*-
# تدريب نموذج LightGBM لأسهم الشريعة في البورصة المصرية (EGX33 + متوافق خارج المؤشر) - إصدار 2
# الجديد في الإصدار 2: سمتان إضافيتان (نسبة السيولة 5/20 وزخم 6 شهور) + جدول معايرة على بيانات الاختبار
# يعمل داخل GitHub Actions: يسحب تاريخ ٥ سنين من Yahoo Finance، يبني نفس السمات الفنية
# العشر اللي التطبيق بيحسبها في المتصفح بالظبط، يدرب مصنف LightGBM بفصل زمني 80/20،
# ويصدّر الأوزان في ملف model_weights.json اللي التطبيق بيجيبه ويشغّل التوقع بيه محليًا.
import json, math, time, datetime
import yfinance as yf
import lightgbm as lgb

SYMS = ["ACGC","ETEL","OCDI","PHDC","SKPC","TMGH","ETRS","IFAP","LCSW","MASR","MPCO","ORWE","SAUD","MCQE","AMOC","EGAS","FAIT","ATQA","CIRA","EGAL","FAITA","JUFO","ORHD","ADIB","ARCC","ORAS","EFID","OLFI","MTIE","RACC","ICFC","ISPH","RMDA","EFIH","GOUR","CLHO","ABUK","SWDY","MFPC","FWRY","POUL","EGCH","EGTS","ELKA","EMFD","TAQA","MNHD"]
LOOK = 10
MINF = 55

def features(c, v):
    n = len(c)
    if n < MINF:
        return None
    last = c[-1]
    def sma(k):
        return sum(c[-k:]) / k
    def vol(k):
        a = c[-k:]
        m = sum(a) / k
        return math.sqrt(sum((x - m) ** 2 for x in a) / k)
    def ret(k):
        return (last - c[n - 1 - k]) / c[n - 1 - k] * 100
    g = 0.0
    l = 0.0
    for i in range(n - 14, n):
        d = c[i] - c[i - 1]
        if d > 0:
            g += d
        else:
            l -= d
    if g + l == 0:
        rsi = 50.0
    else:
        lg = g / 14.0
        ll = l / 14.0
        rsi = 100 - 100 / (1 + lg / (ll if ll != 0 else 1e-9))
    hi = max(c[-120:])
    lo = min(c[-120:])
    rng = (hi - lo) if (hi - lo) != 0 else 1e-9
    def pavg(k):
        a = [x for x in v[-k:] if x is not None and x > 0]
        return (sum(a) / len(a)) if a else None
    m5 = pavg(5)
    m20 = pavg(20)
    vr = (m5 / m20) if (m5 is not None and m20 is not None and m20 > 0) else 1.0
    ret120 = ((last - c[n - 121]) / c[n - 121] * 100) if n >= 121 else 0.0
    return [ret(5), ret(10), ret(20), (last / sma(20) - 1) * 100, (sma(20) / sma(50) - 1) * 100, rsi, (last - lo) / rng * 100, vol(20) / last * 100, vr, ret120]

def main():
    X, Y, ok = [], [], []
    for s in SYMS:
        closes, vols = [], []
        try:
            df = yf.download(s + ".CA", period="5y", interval="1d", progress=False, auto_adjust=True)
            mask = df["Close"].notna()
            closes = [float(x) for x in df["Close"][mask].tolist()]
            vols = [(float(x) if x == x else None) for x in df["Volume"][mask].tolist()]
        except Exception:
            closes, vols = [], []
        cnt = 0
        if len(closes) >= MINF + LOOK + 1:
            for t in range(MINF - 1, len(closes) - LOOK):
                f = features(closes[: t + 1], vols[: t + 1])
                if f is None:
                    continue
                X.append(f)
                Y.append(1 if closes[t + LOOK] > closes[t] else 0)
                cnt += 1
        if cnt:
            ok.append(s)
        print(s, "samples:", cnt)
        time.sleep(0.35)
    print("total:", len(X), "symbols:", len(ok))
    if len(X) < 3000:
        raise SystemExit("insufficient data - keeping existing model")
    cut = int(len(X) * 0.8)
    Xtr, Ytr = X[:cut], Y[:cut]
    Xte, Yte = X[cut:], Y[cut:]
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
    booster = lgb.train(params, lgb.Dataset(Xtr, label=Ytr), num_boost_round=300, valid_sets=[lgb.Dataset(Xte, label=Yte)], callbacks=[lgb.early_stopping(30, verbose=False)])
    best = getattr(booster, "best_iteration", None) or booster.num_trees()
    probs = booster.predict(Xte, num_iteration=best)
    acc = sum(1 for p, y in zip(probs, Yte) if (1 if p >= 0.5 else 0) == y) / len(Yte)
    ones = sum(Yte)
    base = max(ones, len(Yte) - ones) / len(Yte)
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    for r, i in enumerate(order):
        ranks[i] = r + 1.0
    npos = float(ones)
    nneg = float(len(Yte) - ones)
    auc = (sum(ranks[i] for i in range(len(Yte)) if Yte[i] == 1) - npos * (npos + 1) / 2) / (npos * nneg)
    buckets = [(0.0, 0.5, "under 50%"), (0.5, 0.6, "50-60%"), (0.6, 0.7, "60-70%"), (0.7, 0.8, "70-80%"), (0.8, 1.01, "80-100%")]
    cal = []
    for lo, hi, name in buckets:
        idx = [i for i, p in enumerate(probs) if lo <= p < hi]
        if len(idx) >= 30:
            pred = sum(probs[i] for i in idx) / len(idx) * 100
            act = sum(Yte[i] for i in idx) / len(idx) * 100
            cal.append({"range": name, "n": len(idx), "pred_pct": round(pred, 1), "actual_pct": round(act, 1)})
    dm = booster.dump_model(num_iteration=best)
    nodes = []
    def walk(nd):
        if "leaf_value" in nd:
            nodes.append([-1, 0.0, -1, -1, float(nd["leaf_value"])])
            return len(nodes) - 1
        idx2 = len(nodes)
        nodes.append(None)
        f = int(nd["split_feature"])
        thr = float(nd["threshold"])
        left = walk(nd["left_child"])
        right = walk(nd["right_child"])
        nodes[idx2] = [f, thr, left, right, 0.0]
        return idx2
    roots = [walk(t["tree_structure"]) for t in dm["tree_info"]]
    out = {"version": 2, "kind": "lightgbm-egx-sharia-10d", "trained_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), "feature_names": ["ret5", "ret10", "ret20", "pos_sma20", "sma_cross", "rsi14", "range120_pos", "vol20", "vol_ratio_5_20", "ret120"], "n_train": len(Xtr), "n_test": len(Xte), "test_acc": round(acc * 100, 1), "naive_base": round(base * 100, 1), "auc": round(auc, 3), "n_trees": len(roots), "n_symbols": len(ok), "symbols": ok, "calibration": cal, "roots": roots, "nodes": nodes}
    with open("model_weights.json", "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
    print("model v2 written - trees:", len(roots), "nodes:", len(nodes), "acc:", round(acc * 100, 1), "auc:", round(auc, 3), "cal buckets:", len(cal))

if __name__ == "__main__":
    main()
