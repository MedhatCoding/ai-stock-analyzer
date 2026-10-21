# -*- coding: utf-8 -*-
# تدريب نموذج LightGBM لأسهم الشريعة في البورصة المصرية - إصدار 6
# الإصدار 6: لوج تشخيصي كامل يتكتب في train_log.txt وبيترفع للريبو تلقائيًا،
# حارس على قسمة AUC، وثبات النسخة على ubuntu-24.04 في الـ workflow.
# صيغة ملف الأوزان (version: 2) زي ما التطبيق في المتصفح متوقعها.
import json, math, time, datetime, traceback
import urllib.request, http.cookiejar
import numpy as np

try:
    import lightgbm as lgb
except Exception:
    lgb = None

LOG = []

def L(s):
    print(s)
    LOG.append(str(s))

SYMS = ["ACGC","ETEL","OCDI","PHDC","SKPC","TMGH","ETRS","IFAP","LCSW","MASR","MPCO","ORWE","SAUD","MCQE","AMOC","EGAS","FAIT","ATQA","CIRA","EGAL","FAITA","JUFO","ORHD","ADIB","ARCC","ORAS","EFID","OLFI","MTIE","RACC","ICFC","ISPH","RMDA","EFIH","GOUR","CLHO","ABUK","SWDY","MFPC","FWRY","POUL","EGCH","EGTS","ELKA","EMFD","TAQA","MNHD"]
LOOK = 10
MINF = 55

def make_opener():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"), ("Accept", "application/json,text/plain,*/*"), ("Accept-Language", "en-US,en;q=0.9")]
    try:
        op.open("https://fc.yahoo.com/", timeout=15)
    except Exception:
        pass
    return op

def fetch(sym, op):
    err = None
    for host in ("query1", "query2"):
        for attempt in range(3):
            try:
                url = "https://" + host + ".finance.yahoo.com/v8/finance/chart/" + sym + ".CA?range=5y&interval=1d"
                with op.open(url, timeout=30) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                res = data["chart"]["result"][0]
                q = res["indicators"]["quote"][0]
                closes_raw = q.get("close") or []
                vols_raw = q.get("volume") or []
                cs, vs = [], []
                for cc, vv in zip(closes_raw, vols_raw):
                    if cc is not None and cc > 0:
                        cs.append(float(cc))
                        vs.append(float(vv) if vv is not None and vv > 0 else None)
                return cs, vs
            except Exception as e:
                err = e
                time.sleep(2 + attempt * 3)
    L(sym + " fetch failed: " + repr(err))
    return [], []

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
    try:
        if lgb is None:
            raise RuntimeError("lightgbm import failed")
        L("start " + datetime.datetime.now(datetime.timezone.utc).isoformat())
        X, Y, ok = [], [], []
        op = make_opener()
        for s in SYMS:
            closes, vols = fetch(s, op)
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
            L(s + " samples: " + str(cnt))
            time.sleep(1.2)
        L("total: " + str(len(X)) + " symbols: " + str(len(ok)))
        if len(X) < 3000:
            L("insufficient data - no model written this run")
            return
        cut = int(len(X) * 0.8)
        Xtr, Ytr = X[:cut], Y[:cut]
        Xte, Yte = X[cut:], Y[cut:]
        params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
        Xtr_np = np.asarray(Xtr, dtype=float)
        Ytr_np = np.asarray(Ytr, dtype=float)
        Xte_np = np.asarray(Xte, dtype=float)
        Yte_np = np.asarray(Yte, dtype=float)
        booster = lgb.train(params, lgb.Dataset(Xtr_np, label=Ytr_np), num_boost_round=300, valid_sets=[lgb.Dataset(Xte_np, label=Yte_np)], callbacks=[lgb.early_stopping(30, verbose=False)])
        best = getattr(booster, "best_iteration", None) or booster.num_trees()
        probs = booster.predict(Xte_np, num_iteration=best)
        acc = sum(1 for p, y in zip(probs, Yte) if (1 if p >= 0.5 else 0) == y) / len(Yte)
        ones = sum(Yte)
        base = max(ones, len(Yte) - ones) / len(Yte)
        order = sorted(range(len(probs)), key=lambda i: probs[i])
        ranks = [0.0] * len(probs)
        for r, i in enumerate(order):
            ranks[i] = r + 1.0
        npos = float(ones)
        nneg = float(len(Yte) - ones)
        if npos > 0 and nneg > 0:
            auc = (sum(ranks[i] for i in range(len(Yte)) if Yte[i] == 1) - npos * (npos + 1) / 2) / (npos * nneg)
        else:
            auc = 0.5
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
        L("model v2 written - trees: " + str(len(roots)) + " nodes: " + str(len(nodes)) + " acc: " + str(round(acc * 100, 1)) + " auc: " + str(round(auc, 3)) + " cal buckets: " + str(len(cal)))
    except Exception:
        L("ERROR: " + traceback.format_exc())
    finally:
        with open("train_log.txt", "w", encoding="utf-8") as fh:
            fh.write(chr(10).join(LOG))

if __name__ == "__main__":
    main()
