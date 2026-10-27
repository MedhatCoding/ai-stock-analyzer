# -*- coding: utf-8 -*-
# 🏦 Fund Lab v2 — باك تست فلاتر السعر على تاريخ NAV لصناديق الاستثمار المصرية (كل الصناديق بلا فلتر)
# المصدر: مباشر (Mubasher Info) — قائمة كل الصناديق في طلب واحد + ملف CSV بتاريخ NAV لكل صندوق.
# المنهجية: نفس منهجية Volume Lab — تقسيم زمني نص/نص (SPLIT=2024-01-01) + مقارنة متغيرات بصراحة.
# إشارة الدخول المرشحة: تقاطع صاعد SMA20 فوق SMA50 (زخم قياسي للصناديق).
# الفلاتر المفحوصة: ستوكاستيك %K(14)<80 · السعر فوق SMA100 · الاتنين معًا.
# الخروج: هدف +5% / وقف -3% / أفق 30 جلسة (الصناديق أبطأ من الأسهم).
# صفر تعديل على المحرك — ده تشخيص بس، النتايج بتتنشر على فرع fundbt.
import json, time, sys, urllib.request, urllib.error
from datetime import date, datetime, timezone

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
LIST_URL = "https://english.mubasher.info/api/1/funds?country=eg&size=1000"
CSV_URL = "https://static.mubasher.info/File.MubasherCharts/File.Mutual_Fund_Charts_Dir/priceChartFund_{fid}.csv"
SPLIT = "2024-01-01"
MAX_FUNDS = 250
TARGET, STOP, HORIZON = 1.05, 0.97, 30
MIN_PTS = 120
KEYWORDS = []  # v2: بلا فلتر — كل الصناديق

def fetch(url, timeout=25, tries=2):
    for a in range(tries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": UA, "Accept": "*/*"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read().decode("utf-8", "replace")
        except Exception as e:
            if a == tries - 1:
                print("FAIL", url[:70], e, flush=True)
                return None
            time.sleep(2 + a * 2)

def sma(vals, n, i):
    if i + 1 < n: return None
    return sum(vals[i - n + 1:i + 1]) / n

def stoch_k(vals, i, n=14):
    if i + 1 < n: return None
    w = vals[i - n + 1:i + 1]
    hh, ll = max(w), min(w)
    return 50.0 if hh == ll else (vals[i] - ll) / (hh - ll) * 100.0

def metrics(trades):
    if not trades:
        return {"n": 0, "win_rate": None, "avg_ret": None, "med_ret": None}
    rs = sorted(t["ret"] for t in trades)
    n = len(rs)
    med = rs[n // 2] if n % 2 else (rs[n // 2 - 1] + rs[n // 2]) / 2
    wins = sum(1 for t in trades if t["status"] == "hit")
    return {"n": n, "win_rate": round(wins / n * 100, 1), "avg_ret": round(sum(rs) / n * 100, 2), "med_ret": round(med * 100, 2)}

def simulate(vals, dates, filt):
    """إشارة: تقاطع SMA20 فوق SMA50. الفلتر: None | 'stoch' | 'sma100' | 'both'. خروج: +5%/-3%/30 جلسة."""
    trades = []
    open_until = -1
    for i in range(105, len(vals) - 1):
        if i <= open_until: continue
        s20p, s50p = sma(vals, 20, i - 1), sma(vals, 50, i - 1)
        s20, s50 = sma(vals, 20, i), sma(vals, 50, i)
        if None in (s20p, s50p, s20, s50): continue
        if not (s20p <= s50p and s20 > s50): continue
        k = stoch_k(vals, i)
        s100 = sma(vals, 100, i)
        if filt == "stoch" and not (k is not None and k < 80): continue
        if filt == "sma100" and not (s100 is not None and vals[i] > s100): continue
        if filt == "both" and not (k is not None and k < 80 and s100 is not None and vals[i] > s100): continue
        e = vals[i]
        status, exit_i = "d30", min(i + HORIZON, len(vals) - 1)
        for j in range(i + 1, exit_i + 1):
            if vals[j] >= e * TARGET: status, exit_i = "hit", j; break
            if vals[j] <= e * STOP: status, exit_i = "stop", j; break
        trades.append({"d": dates[i], "ret": vals[exit_i] / e - 1.0, "status": status})
        open_until = exit_i
    return trades

def main():
    log = []
    raw = fetch(LIST_URL)
    if not raw: sys.exit("قائمة الصناديق فشلت")
    rows = json.loads(raw).get("rows", [])
    log.append("funds listed: %d" % len(rows))
    latest = {}
    cands = []
    for r in rows:
        fid = r.get("fundId"); name = (r.get("name") or "").strip(); price = r.get("price") or 0
        if fid is None or not name: continue
        latest[name] = {"id": fid, "nav": price, "date": (r.get("date") or "").strip()}
        if price:
            cands.append((fid, name))
    log.append("candidates by keywords: %d" % len(cands))
    cands = cands[:MAX_FUNDS]

    funds = []
    for n, (fid, name) in enumerate(cands, 1):
        txt = fetch(CSV_URL.format(fid=fid))
        time.sleep(0.3)
        if not txt: continue
        pts = {}
        for line in txt.splitlines():
            s, _, v = line.strip().partition(",")
            try:
                d = s[:10].replace("/", "-"); val = float(v)
                if val > 0: pts[d] = val
            except ValueError: continue
        pts = dict(sorted(pts.items()))
        if len(pts) >= MIN_PTS:
            funds.append({"id": fid, "name": name, "dates": list(pts), "vals": list(pts.values())})
        if n % 20 == 0: log.append("fetched %d/%d" % (n, len(cands)))
    log.append("funds with history: %d" % len(funds))

    variants = {k: [] for k in ("base", "stoch", "sma100", "both")}
    per_fund = []
    for f in funds:
        vf = {}
        for k in variants:
            vf[k] = simulate(f["vals"], f["dates"], None if k == "base" else k)
            variants[k] += vf[k]
        per_fund.append({"name": f["name"], "base_n": len(vf["base"]),
                         "both_n": len(vf["both"]),
                         "both_win": metrics(vf["both"])["win_rate"]})

    def split(trades, late):
        return [t for t in trades if (t["d"] >= SPLIT) == late]

    out = {"t": datetime.now(timezone.utc).isoformat(), "version": 2, "source": "mubasher",
           "split": SPLIT, "target": "+5%", "stop": "-3%", "horizon": "30 جلسة",
           "funds_ok": len(funds),
           "span": [[min(f["dates"][0] for f in funds), max(f["dates"][-1] for f in funds)] if funds else None][0],
           "variants": {}}
    for k, tr in variants.items():
        out["variants"][k] = {
            "all": metrics(tr),
            "early": metrics(split(tr, False)),
            "late": metrics(split(tr, True)),
        }
    out["per_fund_both"] = sorted(per_fund, key=lambda x: -(x["both_win"] or 0))[:15]
    out["latest_nav"] = latest
    out["log_tail"] = log
    with open("fund_bt.json", "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, indent=1)
    print("\n".join(log))
    print("✅ fund_bt.json جاهز")

if __name__ == "__main__":
    main()
