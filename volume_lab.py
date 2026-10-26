# -*- coding: utf-8 -*-
# volume_lab.py v2 — مختبر الإشارات: باك تست صادق لعوامل إضافية على محرك التوصيات الحالي
# نفس ميزان سجل التوصيات: هدف 3xATR، وقف 2xATR، أفق 14 جلسة، توصية واحدة مفتوحة لكل سهم
# المتغيرات: أ) المحرك الحالي · ب) OBV20 · ج) ضغط شراء · د) وزن OBV · هـ) ب+ج
# الجديد: و) Stochastic<80 · ز) OBV10 أسرع · ح) فوق SMA100 · ط) كمية اليوم>=0.8x المتوسط
#         ي) تقلب ATR%<=5 · ك) ب+و · ل) ب+ح
import json, math, time, datetime, traceback, urllib.request, http.cookiejar

SYMS = "ABUK ACGC ADIB AMOC ARCC ATQA CIEB CIRA CLHO COMI EAST EFID EFIH EGAL EGAS EGCH EGTS ELKA EMFD ETEL ETRS FAIT FAITA FWRY GBCO GOUR HDBK HRHO ICFC IFAP ISPH JUFO LCSW MASR MBSC MCQE MFPC MNHD MPCI MPCO MTIE OCDI OLFI ORAS ORHD ORWE PHDC POUL RACC RMDA SAUD SKPC SWDY TAQA TMGH".split()
HORIZON = 14
MINI = 60
W52 = 250
LOG = []

def L(s):
    print(s)
    LOG.append(str(s))

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
                ts = res.get("timestamp") or []
                oo = q.get("open") or []; hh = q.get("high") or []; ll = q.get("low") or []; cc = q.get("close") or []; vv = q.get("volume") or []
                rows = []
                for i in range(len(cc)):
                    if cc[i] is None or cc[i] <= 0:
                        continue
                    rows.append({
                        "o": float(oo[i]) if (i < len(oo) and oo[i] is not None) else float(cc[i]),
                        "h": float(hh[i]) if (i < len(hh) and hh[i] is not None) else float(cc[i]),
                        "l": float(ll[i]) if (i < len(ll) and ll[i] is not None) else float(cc[i]),
                        "c": float(cc[i]),
                        "v": float(vv[i]) if (i < len(vv) and vv[i] is not None) else 0.0,
                        "d": datetime.datetime.utcfromtimestamp(int(ts[i])).strftime("%Y-%m-%d") if i < len(ts) else None,
                    })
                return rows
            except Exception as e:
                err = e
                time.sleep(2 + attempt * 3)
    L(sym + " fetch failed: " + repr(err))
    return []

def ema_s(c, n):
    k = 2.0 / (n + 1.0)
    out = [c[0]]
    for i in range(1, len(c)):
        out.append(c[i] * k + out[i - 1] * (1 - k))
    return out

def rsi_s(c, n=14):
    out = [None]
    g = 0.0; l = 0.0
    for i in range(1, len(c)):
        d = c[i] - c[i - 1]
        if i <= n:
            g += max(d, 0.0); l += max(-d, 0.0)
            out.append(None)
            if i == n:
                g /= n; l /= n
        else:
            g = (g * (n - 1) + max(d, 0.0)) / n
            l = (l * (n - 1) + max(-d, 0.0)) / n
            out.append(100.0 if l == 0 else 100.0 - 100.0 / (1.0 + g / l))
    return out

def precompute(rows):
    c = [r["c"] for r in rows]
    n = len(rows)
    e12 = ema_s(c, 12); e26 = ema_s(c, 26)
    line = [e12[i] - e26[i] for i in range(n)]
    sig = ema_s(line, 9)
    rsi = rsi_s(c)
    def sma_arr(k):
        out = [None] * n
        s = 0.0
        for i in range(n):
            s += c[i]
            if i >= k:
                s -= c[i - k]
            if i >= k - 1:
                out[i] = s / k
        return out
    s20 = sma_arr(20); s50 = sma_arr(50); s100 = sma_arr(100)
    bu = [None] * n; bl = [None] * n
    s = 0.0; sq = 0.0
    for i in range(n):
        s += c[i]; sq += c[i] * c[i]
        if i >= 20:
            s -= c[i - 20]; sq -= c[i - 20] * c[i - 20]
        if i >= 19:
            m = s / 20.0
            sd = math.sqrt(max(0.0, sq / 20.0 - m * m))
            bu[i] = m + 2 * sd; bl[i] = m - 2 * sd
    tr = [0.0] * n
    for i in range(1, n):
        tr[i] = max(rows[i]["h"] - rows[i]["l"], abs(rows[i]["h"] - rows[i - 1]["c"]), abs(rows[i]["l"] - rows[i - 1]["c"]))
    atr = [None] * n
    for i in range(14, n):
        atr[i] = sum(tr[i - 13:i + 1]) / 14.0
    obv = [0.0] * n
    for i in range(1, n):
        d = c[i] - c[i - 1]
        obv[i] = obv[i - 1] + (rows[i]["v"] if d > 0 else (-rows[i]["v"] if d < 0 else 0.0))
    # Stochastic %K (14) خام
    stk = [None] * n
    for i in range(13, n):
        hh = max(r["h"] for r in rows[i - 13:i + 1])
        ll = min(r["l"] for r in rows[i - 13:i + 1])
        stk[i] = 50.0 if hh == ll else (c[i] - ll) / (hh - ll) * 100.0
    # متوسط كمية 20
    av20 = [None] * n
    vs = 0.0
    for i in range(n):
        vs += rows[i]["v"]
        if i >= 20:
            vs -= rows[i - 20]["v"]
        if i >= 19:
            av20[i] = vs / 20.0
    return {"c": c, "line": line, "sig": sig, "rsi": rsi, "s20": s20, "s50": s50, "s100": s100, "bu": bu, "bl": bl, "atr": atr, "obv": obv, "stk": stk, "av20": av20}

def factors_at(rows, pc, i):
    c = pc["c"]
    obv20 = pc["obv"][i] - pc["obv"][max(0, i - 20)]
    obv10 = pc["obv"][i] - pc["obv"][max(0, i - 10)]
    uv = 0.0; dv = 0.0
    for j in range(max(1, i - 19), i + 1):
        d = c[j] - c[j - 1]
        if d > 0: uv += rows[j]["v"]
        elif d < 0: dv += rows[j]["v"]
    pr20 = (uv / dv) if dv > 0 else None
    av = pc["av20"][i]
    rv = (rows[i]["v"] / av) if (av is not None and av > 0) else None
    a = pc["atr"][i]
    atrpct = (a / c[i] * 100.0) if (a is not None and a > 0 and c[i] > 0) else None
    return {"obv20": obv20, "obv10": obv10, "pr20": pr20, "rv": rv, "stk": pc["stk"][i], "t100": (c[i] > pc["s100"][i]) if pc["s100"][i] is not None else None, "atrpct": atrpct}

def engine_sc(rows, pc, i):
    c = pc["c"]
    p = c[i]
    s20v = pc["s20"][i]; s50v = pc["s50"][i]
    rv = pc["rsi"][i]; mv = pc["line"][i]; sv = pc["sig"][i]
    bU = pc["bu"][i]; bL = pc["bl"][i]
    w0 = max(0, i - W52 + 1)
    hi52 = max(r["h"] for r in rows[w0:i + 1]); lo52 = min(r["l"] for r in rows[w0:i + 1])
    pos = (p - lo52) / ((hi52 - lo52) or 1.0)
    seg = rows[max(0, i - 9):i + 1]
    vUp = sum(r["v"] for r in seg if r["c"] >= r["o"])
    vDn = sum(r["v"] for r in seg if r["c"] < r["o"])
    sc = 0.0
    sc += 1.2 if p > s20v else -1.2
    sc += 1.3 if s20v > s50v else -1.3
    if rv < 30: sc += 2.0
    elif rv > 70: sc -= 2.0
    else: sc += (rv - 50.0) / 25.0
    sc += 1.5 if mv > sv else -1.5
    if p < bL: sc += 1.0
    elif p > bU: sc -= 1.0
    if vUp > vDn * 1.2: sc += 0.8
    elif vDn > vUp * 1.2: sc -= 0.8
    if pos > 0.85: sc += 0.6
    elif pos < 0.2: sc -= 0.4
    a = pc["atr"][i]
    ag = a if (a is not None and a > 0) else p * 0.02
    return sc, ag

def cond(v, i, f):
    base = v["sc"] >= 1.5
    if v["key"] == "A": return base
    if v["key"] == "B": return base and f["obv20"] > 0
    if v["key"] == "C": return base and f["pr20"] is not None and f["pr20"] >= 1.0
    if v["key"] == "D": return (v["sc"] + (1.0 if f["obv20"] > 0 else -1.0)) >= 1.5
    if v["key"] == "E": return base and f["obv20"] > 0 and f["pr20"] is not None and f["pr20"] >= 1.0
    if v["key"] == "F": return base and f["stk"] is not None and f["stk"] < 80
    if v["key"] == "G": return base and f["obv10"] > 0
    if v["key"] == "H": return base and f["t100"] is True
    if v["key"] == "I": return base and f["rv"] is not None and f["rv"] >= 0.8
    if v["key"] == "J": return base and f["atrpct"] is not None and f["atrpct"] <= 5.0
    if v["key"] == "K": return base and f["obv20"] > 0 and f["stk"] is not None and f["stk"] < 80
    if v["key"] == "L": return base and f["obv20"] > 0 and f["t100"] is True
    return base

VARIANTS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L"]

def simulate(rows, pc, key):
    c = pc["c"]
    n = len(rows)
    trades = []
    pos = None
    i = MINI - 1
    while i < n:
        if pos is None:
            if i >= n - 1:
                break
            sc, ag = engine_sc(rows, pc, i)
            f = factors_at(rows, pc, i)
            state = {"key": key, "sc": sc}
            if cond(state, i, f):
                p = c[i]
                pos = {"entry": p, "stop": p - 2 * ag, "target": p + 3 * ag, "i": i}
                pos.update(f)
        else:
            if rows[i]["l"] <= pos["stop"]:
                trades.append(mk(pos, pos["stop"], "stop", i)); pos = None
            elif rows[i]["h"] >= pos["target"]:
                trades.append(mk(pos, pos["target"], "hit", i)); pos = None
            elif i - pos["i"] >= HORIZON:
                trades.append(mk(pos, c[i], "d14", i)); pos = None
        i += 1
    if pos is not None:
        trades.append(mk(pos, c[n - 1], "open", n - 1))
    return trades

def mk(pos, exitp, outcome, j):
    t = {"ret": round((exitp / pos["entry"] - 1.0) * 100.0, 3), "outcome": outcome, "days": j - pos["i"]}
    for k in ("obv20", "obv10", "pr20", "rv", "stk", "t100", "atrpct"):
        t[k] = pos.get(k)
    return t

def metrics(trades):
    closed = [t for t in trades if t["outcome"] != "open"]
    opens = len(trades) - len(closed)
    rets = [t["ret"] for t in closed]
    hits = sum(1 for t in closed if t["outcome"] == "hit")
    stops = sum(1 for t in closed if t["outcome"] == "stop")
    d14 = sum(1 for t in closed if t["outcome"] == "d14")
    gains = sum(r for r in rets if r > 0)
    losses = abs(sum(r for r in rets if r < 0))
    srt = sorted(rets)
    med = srt[len(srt) // 2] if srt else None
    return {
        "n": len(closed), "open": opens,
        "hits": hits, "stops": stops, "timeout": d14,
        "hit_rate": round(hits / len(closed) * 100, 1) if closed else None,
        "stop_rate": round(stops / len(closed) * 100, 1) if closed else None,
        "win_rate": round(sum(1 for r in rets if r > 0) / len(closed) * 100, 1) if closed else None,
        "avg_ret": round(sum(rets) / len(rets), 3) if rets else None,
        "med_ret": med,
        "pf": round(gains / losses, 2) if losses > 0 else None,
        "total": round(sum(rets), 1) if rets else None,
    }

def main():
    try:
        op = make_opener()
        all_trades = {v: [] for v in VARIANTS}
        syms_ok = []
        syms_novol = []
        span = [None, None]
        for s in SYMS:
            rows = fetch(s, op)
            time.sleep(1.5)
            if len(rows) < MINI + HORIZON + 5:
                L(s + ": بيانات غير كافية (" + str(len(rows)) + ")")
                continue
            lastw = rows[-250:]
            volc = sum(1 for r in lastw if r["v"] > 0) / float(len(lastw))
            if volc < 0.8:
                syms_novol.append(s)
                L(s + ": حجم التداول ناقص من المصدر (" + str(round(volc * 100)) + "%) — خارج العينة")
                continue
            pc = precompute(rows)
            for v in VARIANTS:
                tr = simulate(rows, pc, v)
                for t in tr:
                    t["sym"] = s
                    all_trades[v].append(t)
            syms_ok.append(s)
            L(s + ": شموع " + str(len(rows)))
            if rows[0]["d"] and (span[0] is None or rows[0]["d"] < span[0]): span[0] = rows[0]["d"]
            if rows[-1]["d"] and (span[1] is None or rows[-1]["d"] > span[1]): span[1] = rows[-1]["d"]
        base = [t for t in all_trades["A"] if t["outcome"] != "open"]
        ma = metrics(all_trades["A"])
        sel = {}
        conds = {
            "obv20": lambda t: t["obv20"] is not None and t["obv20"] > 0,
            "pressure": lambda t: t["pr20"] is not None and t["pr20"] >= 1.0,
            "stoch80": lambda t: t["stk"] is not None and t["stk"] < 80,
            "sma100": lambda t: t["t100"] is True,
            "relvol": lambda t: t["rv"] is not None and t["rv"] >= 0.8,
            "atrpct5": lambda t: t["atrpct"] is not None and t["atrpct"] <= 5.0,
        }
        for name, fn in conds.items():
            gp = [t for t in base if fn(t)]
            gf = [t for t in base if not fn(t)]
            sel[name] = {"pass": metrics(gp), "fail": metrics(gf)}
        labels = {"A": "المحرك الحالي", "B": "فلتر OBV20", "C": "فلتر ضغط الشراء", "D": "وزن OBV", "E": "ب+ج", "F": "فلتر Stochastic<80", "G": "فلتر OBV10", "H": "فلتر فوق SMA100", "I": "فلتر كمية>=0.8x", "J": "فلتر تقلب<=5%", "K": "ب+و", "L": "ب+ح"}
        variants = {}
        for v in VARIANTS:
            m = metrics(all_trades[v])
            m["label"] = labels[v]
            if v != "A" and ma["win_rate"] is not None and m["win_rate"] is not None:
                m["d_win"] = round(m["win_rate"] - ma["win_rate"], 1)
                m["d_avg"] = round((m["avg_ret"] or 0) - (ma["avg_ret"] or 0), 3)
                m["d_med"] = round((m["med_ret"] or 0) - (ma["med_ret"] or 0), 3)
            variants[v] = m
        out = {
            "t": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "version": 2,
            "span": span,
            "symbols_ok": syms_ok,
            "symbols_no_volume": syms_novol,
            "variants": variants,
            "selectivity": sel,
            "log_tail": LOG[-25:],
        }
        with open("volume_bt.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
        L("==== النتائج (v2) ====")
        for v in VARIANTS:
            L(v + " " + labels[v] + ": " + json.dumps(variants[v], ensure_ascii=False))
    except Exception:
        L("ERROR: " + traceback.format_exc())
    finally:
        with open("lab_log.txt", "w", encoding="utf-8") as fh:
            fh.write(chr(10).join(LOG))

if __name__ == "__main__":
    main()
