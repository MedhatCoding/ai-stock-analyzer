# -*- coding: utf-8 -*-
# volume_lab.py — مختبر الفوليوم: باك تست صادق لعوامل حجم التداول على محرك التوصيات الحالي
# (أ) المحرك الحالي حرفيًا كما في index.html/analyze — الأوزان الافتراضية (SIGW=1)
# (ب) + فلتر: OBV صاعد خلال 20 جلسة
# (ج) + فلتر: ضغط شراء — حجم أيام الصعود >= حجم أيام الهبوط خلال 20 جلسة
# (د) إشارة OBV كوزن +/-1 في الدرجة (بدون فلتر)
# (هـ) فلترا (ب)+(ج) معًا
# القياس بنفس ميزان سجل التوصيات في التطبيق: هدف 3xATR، وقف 2xATR، أفق 14 جلسة، توصية واحدة مفتوحة لكل سهم
import json, math, time, datetime, traceback, urllib.request, http.cookiejar

SYMS = "ABUK ACGC ADIB AMOC ARCC ATQA CIEB CIRA CLHO COMI EAST EFID EFIH EGAL EGAS EGCH EGTS ELKA EMFD ETEL ETRS FAIT FAITA FWRY GBCO GOUR HDBK HRHO ICFC IFAP ISPH JUFO LCSW MASR MBSC MCQE MFPC MNHD MPCI MPCO MTIE OCDI OLFI ORAS ORHD ORWE PHDC POUL RACC RMDA SAUD SKPC SWDY TAQA TMGH".split()
HORIZON = 14
MINI = 60        # نفس شرط التطبيق: أقل من 60 شمعة = لا تحليل
W52 = 250        # التطبيق بيجيب سنة — قمة/قاع 52 أسبوع على آخر 250 شمعة
LOG = []

def L(s):
    print(s)
    LOG.append(str(s))

def make_opener():
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"), ("Accept", "application/json,text/plain,*/*"), ("Accept-Language", "en-US,en;q=0.9")]
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
    # SMA rolling
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
    s20 = sma_arr(20); s50 = sma_arr(50)
    # Bollinger 20/2
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
    # ATR: متوسط آخر 14 مدى حقيقي (زي التطبيق بالظبط)
    tr = [0.0] * n
    for i in range(1, n):
        tr[i] = max(rows[i]["h"] - rows[i]["l"], abs(rows[i]["h"] - rows[i - 1]["c"]), abs(rows[i]["l"] - rows[i - 1]["c"]))
    atr = [None] * n
    for i in range(14, n):
        atr[i] = sum(tr[i - 13:i + 1]) / 14.0
    # OBV تراكمي
    obv = [0.0] * n
    for i in range(1, n):
        d = c[i] - c[i - 1]
        obv[i] = obv[i - 1] + (rows[i]["v"] if d > 0 else (-rows[i]["v"] if d < 0 else 0.0))
    return {"c": c, "line": line, "sig": sig, "rsi": rsi, "s20": s20, "s50": s50, "bu": bu, "bl": bl, "atr": atr, "obv": obv}

def engine_sc(rows, pc, i):
    # نقل حرفي لمنطق analyze في index.html (SIGW = 1)
    c = pc["c"]
    p = c[i]
    s20v = pc["s20"][i]; s50v = pc["s50"][i]
    rv = pc["rsi"][i]; mv = pc["line"][i]; sv = pc["sig"][i]
    bU = pc["bu"][i]; bL = pc["bl"][i]
    w0 = max(0, i - W52 + 1)
    hi52 = max(r["h"] for r in rows[w0:i + 1]); lo52 = min(r["l"] for r in rows[w0:i + 1])
    pos = (p - lo52) / ((hi52 - lo52) or 1.0)
    # حجم أيام الصعود/الهبوط آخر 10 جلسات (حسب جسم الشمعة — زي التطبيق)
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
    # عوامل الفوليوم الجديدة (مقاسة هنا بس — غير داخلة في sc)
    obv20 = pc["obv"][i] - pc["obv"][max(0, i - 20)]
    uv = 0.0; dv = 0.0
    for j in range(max(1, i - 19), i + 1):
        d = c[j] - c[j - 1]
        if d > 0: uv += rows[j]["v"]
        elif d < 0: dv += rows[j]["v"]
    pr20 = (uv / dv) if dv > 0 else None
    return sc, obv20, pr20, atr_of(pc, rows, i), pos

def atr_of(pc, rows, i):
    a = pc["atr"][i]
    return a if (a is not None and a > 0) else rows[i]["c"] * 0.02

def simulate(rows, pc, variant):
    c = pc["c"]
    n = len(rows)
    trades = []
    pos = None
    i = MINI - 1
    while i < n:
        if pos is None:
            if i >= n - 1:
                break
            sc, obv20, pr20, a, poss = engine_sc(rows, pc, i)
            ok = sc >= 1.5
            if variant == "B": ok = ok and obv20 > 0
            elif variant == "C": ok = ok and pr20 is not None and pr20 >= 1.0
            elif variant == "D": ok = (sc + (1.0 if obv20 > 0 else -1.0)) >= 1.5
            elif variant == "E": ok = ok and obv20 > 0 and pr20 is not None and pr20 >= 1.0
            if ok:
                p = c[i]
                pos = {"entry": p, "stop": p - 2 * a, "target": p + 3 * a, "i": i, "obv": obv20, "pr": pr20, "sc": sc}
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
    return {"ret": round((exitp / pos["entry"] - 1.0) * 100.0, 3), "outcome": outcome, "days": j - pos["i"], "obv": pos["obv"], "pr": pos["pr"], "sc": round(pos["sc"], 2)}

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
        all_trades = {v: [] for v in ("A", "B", "C", "D", "E")}
        per_symbol = {}
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
            tr_a = simulate(rows, pc, "A")
            tr_b = simulate(rows, pc, "B")
            tr_c = simulate(rows, pc, "C")
            tr_d = simulate(rows, pc, "D")
            tr_e = simulate(rows, pc, "E")
            for v, tr in (("A", tr_a), ("B", tr_b), ("C", tr_c), ("D", tr_d), ("E", tr_e)):
                for t in tr:
                    t["sym"] = s
                    all_trades[v].append(t)
            ma = metrics(tr_a); mb = metrics(tr_b)
            per_symbol[s] = {"A": [ma["n"], ma["win_rate"]], "B": [mb["n"], mb["win_rate"]]}
            syms_ok.append(s)
            L(s + ": شموع " + str(len(rows)) + " · توصيات أ " + str(ma["n"]) + " (فوز " + str(ma["win_rate"]) + "%) · توصيات ب " + str(mb["n"]) + " (فوز " + str(mb["win_rate"]) + "%)")
            if rows[0]["d"] and (span[0] is None or rows[0]["d"] < span[0]): span[0] = rows[0]["d"]
            if rows[-1]["d"] and (span[1] is None or rows[-1]["d"] > span[1]): span[1] = rows[-1]["d"]
        # تحليل الانتقائية: توصيات المحرك الحالي مقسومة حسب حالة الفوليوم وقت الدخول
        base = [t for t in all_trades["A"] if t["outcome"] != "open"]
        sel = {}
        for name, key in (("obv", "obv"), ("pressure", "pr")):
            gp = [t for t in base if (t[key] > 0 if key == "obv" else (t[key] is not None and t[key] >= 1.0))]
            gf = [t for t in base if not (t[key] > 0 if key == "obv" else (t[key] is not None and t[key] >= 1.0))]
            sel[name] = {"pass": metrics([dict(t, outcome="x") for t in gp] if False else gp), "fail": metrics(gf)}
        out = {
            "t": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "span": span,
            "symbols_ok": syms_ok,
            "symbols_no_volume": syms_novol,
            "variants": {v: metrics(all_trades[v]) for v in ("A", "B", "C", "D", "E")},
            "selectivity": sel,
            "per_symbol": per_symbol,
            "log_tail": LOG[-25:],
        }
        with open("volume_bt.json", "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
        L("==== النتائج ====")
        for v in ("A", "B", "C", "D", "E"):
            L(v + ": " + json.dumps(out["variants"][v], ensure_ascii=False))
        L("انتقائية: " + json.dumps({k: {"pass": m["pass"]["win_rate"], "fail": m["fail"]["win_rate"]} for k, m in sel.items()}, ensure_ascii=False))
    except Exception:
        L("ERROR: " + traceback.format_exc())
    finally:
        with open("lab_log.txt", "w", encoding="utf-8") as fh:
            fh.write(chr(10).join(LOG))

if __name__ == "__main__":
    main()
