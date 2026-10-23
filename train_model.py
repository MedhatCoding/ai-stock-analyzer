# -*- coding: utf-8 -*-
# سهمي — تدريب نموذج LightGBM لأسهم الشريعة في البورصة المصرية - إصدار 7
# الإصدار 7: 16 خاصية (إضافة MACD وبولنجر واتجاه SMA وRSI7 وعائد 60 يوم)،
# وتقييم walk-forward زمني صادق (تدريب على الماضي واختبار على المستقبل في 4 طيات)
# بدل القسمة الثابتة، وثبات صيغة الأوزان المتوقعة في المتصفح (version: 3).
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
MINF = 60
FEATS = ["ret5", "ret10", "ret20", "pos_sma20", "sma_cross", "rsi14", "range120_pos", "vol20", "vol_ratio_5_20", "ret120", "macd_hist", "boll_z", "pos_sma50", "ret60", "rsi7", "sma20_slope"]

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
                ts_raw = res.get("timestamp") or []
                closes_raw = q.get("close") or []
                vols_raw = q.get("volume") or []
                cs, vs, ts = [], [], []
                for i in range(len(closes_raw)):
                    cc = closes_raw[i]
                    vv = vols_raw[i] if i < len(vols_raw) else None
                    if cc is not None and cc > 0:
                        cs.append(float(cc))
                        vs.append(float(vv) if vv is not None and vv > 0 else None)
                        ts.append(int(ts_raw[i]) if i < len(ts_raw) else (ts[-1] + 86400 if ts else 0))
                return cs, vs, ts
            except Exception as e:
                err = e
                time.sleep(2 + attempt * 3)
    L(sym + " fetch failed: " + repr(err))
    return [], [], []

def ema_series(c, k):
    a = 2.0 / (k + 1.0)
    out = [0.0] * len(c)
    out[0] = c[0]
    for i in range(1, len(c)):
        out[i] = a * c[i] + (1.0 - a) * out[i - 1]
    return out

def macd_hist_series(c):
    e12 = ema_series(c, 12)
    e26 = ema_series(c, 26)
    macd = [e12[i] - e26[i] for i in range(len(c))]
    a = 2.0 / 10.0
    sig = [0.0] * len(c)
    sig[0] = macd[0]
    for i in range(1, len(c)):
        sig[i] = a * macd[i] + (1.0 - a) * sig[i - 1]
    return [macd[i] - sig[i] for i in range(len(c))]

def rsi_of(c, t, k):
    g = 0.0
    l = 0.0
    for i in range(t - k + 1, t + 1):
        d = c[i] - c[i - 1]
        if d > 0:
            g += d
        else:
            l -= d
    if g + l == 0:
        return 50.0
    return 100 - 100 / (1 + (g / k) / ((l / k) if l != 0 else 1e-9))

def features(c, v, mh, t):
    n = t + 1
    if n < MINF:
        return None
    last = c[t]
    def sma(end, k):
        return sum(c[end - k + 1:end + 1]) / k
    def ret(k):
        return (last - c[t - k]) / c[t - k] * 100
    def vol(end, k):
        a = c[end - k + 1:end + 1]
        m = sum(a) / k
        return math.sqrt(sum((x - m) ** 2 for x in a) / k)
    rsi14 = rsi_of(c, t, 14)
    rsi7 = rsi_of(c, t, 7)
    hi = max(c[max(0, t - 119):t + 1])
    lo = min(c[max(0, t - 119):t + 1])
    rng = (hi - lo) if (hi - lo) != 0 else 1e-9
    def pavg(end, k):
        a = [x for x in v[max(0, end - k + 1):end + 1] if x is not None and x > 0]
        return (sum(a) / len(a)) if a else None
    m5 = pavg(t, 5)
    m20 = pavg(t, 20)
    vr = (m5 / m20) if (m5 is not None and m20 is not None and m20 > 0) else 1.0
    ret120 = ((last - c[t - 120]) / c[t - 120] * 100) if t >= 120 else 0.0
    s20 = sma(t, 20)
    s50 = sma(t, 50)
    std20 = vol(t, 20)
    boll_z = (last - s20) / (std20 if std20 != 0 else 1e-9)
    s20p = sma(t - 5, 20) if t >= 24 else s20
    slope = (s20 - s20p) / last * 100
    ret60 = ret(60) if t >= 60 else 0.0
    macd_h = (mh[t] / last) * 100 if mh[t] is not None else 0.0
    return [ret(5), ret(10), ret(20), (last / s20 - 1) * 100, (s20 / s50 - 1) * 100, rsi14, (last - lo) / rng * 100, std20 / last * 100, vr, ret120, macd_h, boll_z, (last / s50 - 1) * 100, ret60, rsi7, slope]

def auc_of(probs, ys):
    ones = sum(ys)
    if ones <= 0 or ones >= len(ys):
        return 0.5
    order = sorted(range(len(probs)), key=lambda i: probs[i])
    ranks = [0.0] * len(probs)
    for r, i in enumerate(order):
        ranks[i] = r + 1.0
    npos = float(ones)
    nneg = float(len(ys) - ones)
    return (sum(ranks[i] for i in range(len(ys)) if ys[i] == 1) - npos * (npos + 1) / 2) / (npos * nneg)

def train_booster(Xtr, Ytr, Xval, Yval):
    params = {"objective": "binary", "learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 50, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1, "verbose": -1, "seed": 42}
    dtr = lgb.Dataset(np.asarray(Xtr, dtype=float), label=np.asarray(Ytr, dtype=float))
    dval = lgb.Dataset(np.asarray(Xval, dtype=float), label=np.asarray(Yval, dtype=float))
    booster = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dval], callbacks=[lgb.early_stopping(30, verbose=False)])
    best = getattr(booster, "best_iteration", None) or booster.num_trees()
    return booster, best

def main():
    try:
        if lgb is None:
            raise RuntimeError("lightgbm import failed")
        L("start " + datetime.datetime.now(datetime.timezone.utc).isoformat())
        recs = []
        ok = []
        op = make_opener()
        for s in SYMS:
            closes, vols, ts = fetch(s, op)
            cnt = 0
            if len(closes) >= MINF + LOOK + 1:
                mh = macd_hist_series(closes)
                for t in range(MINF - 1, len(closes) - LOOK):
                    f = features(closes, vols, mh, t)
                    if f is None:
                        continue
                    y = 1 if closes[t + LOOK] > closes[t] else 0
                    recs.append((ts[t] if t < len(ts) else t * 86400, s, f, y))
                    cnt += 1
            if cnt:
                ok.append(s)
            L(s + " samples: " + str(cnt))
            time.sleep(1.2)
        recs.sort(key=lambda r: r[0])
        X = [r[2] for r in recs]
        Y = [r[3] for r in recs]
        L("total: " + str(len(X)) + " symbols: " + str(len(ok)) + " span: " + (datetime.datetime.utcfromtimestamp(recs[0][0]).strftime("%Y-%m-%d") + " -> " + datetime.datetime.utcfromtimestamp(recs[-1][0]).strftime("%Y-%m-%d") if recs else "n/a"))
        if len(X) < 3000:
            L("insufficient data - no model written this run")
            return
        # ─── walk-forward زمني: 4 طيات — اتدرب على الماضي واختبر على المستقبل ───
        NF = 4
        fs = len(X) // (NF + 1)
        aucs = []
        for k in range(1, NF + 1):
            tr_end = fs * k
            te_lo, te_hi = tr_end, min(tr_end + fs, len(X))
            if tr_end < 500 or te_hi - te_lo < 300:
                continue
            val_cut = int(tr_end * 0.9)
            booster, _ = train_booster(X[:val_cut], Y[:val_cut], X[val_cut:tr_end], Y[val_cut:tr_end])
            probs = booster.predict(np.asarray(X[te_lo:te_hi], dtype=float))
            a = auc_of(probs, Y[te_lo:te_hi])
            aucs.append(a)
            L("walk-forward fold " + str(k) + ": auc=" + str(round(a, 3)) + " train_n=" + str(tr_end) + " test_n=" + str(te_hi - te_lo))
        wf_auc = round(sum(aucs) / len(aucs), 3) if aucs else None
        L("walk-forward mean auc: " + str(wf_auc))
        # ─── الموديل النهائي: تدريب على أول 85٪ زمنيًا واختبار صادق على آخر 15٪ ───
        cut = int(len(X) * 0.85)
        Xtr, Ytr = X[:cut], Y[:cut]
        Xte, Yte = X[cut:], Y[cut:]
        val_cut = int(cut * 0.9)
        booster, best = train_booster(Xtr[:val_cut], Ytr[:val_cut], Xtr[val_cut:], Ytr[val_cut:])
        probs = booster.predict(np.asarray(Xte, dtype=float), num_iteration=best)
        acc = sum(1 for p, y in zip(probs, Yte) if (1 if p >= 0.5 else 0) == y) / len(Yte)
        ones = sum(Yte)
        base = max(ones, len(Yte) - ones) / len(Yte)
        auc = auc_of(probs, Yte)
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
        out = {"version": 3, "kind": "lightgbm-egx-sharia-10d", "trained_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"), "feature_names": FEATS, "n_train": len(Xtr), "n_test": len(Xte), "test_acc": round(acc * 100, 1), "naive_base": round(base * 100, 1), "auc": round(auc, 3), "wf_auc": wf_auc, "wf_folds": len(aucs), "n_trees": len(roots), "n_symbols": len(ok), "symbols": ok, "calibration": cal, "roots": roots, "nodes": nodes}
        with open("model_weights.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, separators=(",", ":"))
        L("model v3 written - trees: " + str(len(roots)) + " nodes: " + str(len(nodes)) + " acc: " + str(round(acc * 100, 1)) + " auc: " + str(round(auc, 3)) + " wf_auc: " + str(wf_auc) + " cal buckets: " + str(len(cal)))
    except Exception:
        L("ERROR: " + traceback.format_exc())
    finally:
        with open("train_log.txt", "w", encoding="utf-8") as fh:
            fh.write(chr(10).join(LOG))

if __name__ == "__main__":
    main()
