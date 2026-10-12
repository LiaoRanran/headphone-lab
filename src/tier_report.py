#!/usr/bin/env python3
"""分价位报告：把实测匹配结果 × 已知价格做分档。

价格来源只有两处，且**都标了出处**：
  1) data/prices.json —— 本次核到的少量价格
  2) 技能市场库 market_catalog.json 的 msrp(USD)，按 7.2 粗算 → 标「粗算」
未核到价格的型号**不猜**，单独列入「价格待核」。
"""
import argparse
import json
import os
import re

import engine

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SKILL_CAT = r"C:\Users\ASUS\.codebuddy\skills\headphone-advisor\references\market_catalog.json"

TIERS = [(0, 500, "≤¥500 入门"), (500, 1500, "¥500–1500 中低"),
         (1500, 3000, "¥1500–3000 中端"), (3000, 7000, "¥3000–7000 中高"),
         (7000, 10 ** 9, ">¥7000 高端")]


def norm(s):
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()


def load_used():
    pr = json.load(open(os.path.join(ROOT, "data", "prices.json"), encoding="utf-8"))
    return {norm(k): v for k, v in (pr.get("used_cny") or {}).items()}


def lookup_used(table, name):
    k = norm(name)
    for key, v in table.items():
        if key and (key in k or k in key):
            return v
    return None


def load_prices():
    pr = json.load(open(os.path.join(ROOT, "data", "prices.json"), encoding="utf-8"))
    rate = pr.get("usd_cny", 7.2)
    table = {}
    for k, v in pr.get("overrides_cny", {}).items():
        table[norm(k)] = (v["cny"], v.get("src", ""), v.get("note", ""))
    for k, v in pr.get("overrides_usd", {}).items():
        table[norm(k)] = (round(v["usd"] * rate), "USD %d → ¥ 粗算 ｜ %s" % (v["usd"], v.get("src", "")), "待核" if v.get("pending") else "")
    if os.path.exists(SKILL_CAT):
        cat = json.load(open(SKILL_CAT, encoding="utf-8"))
        for m in cat["models"]:
            key = norm("%s %s" % (m["brand"], m["model"]))
            msrp = (m.get("price_ref") or {}).get("msrp")
            if msrp and key not in table:
                table[key] = (round(msrp * rate), "技能库 msrp $%d → ¥ 粗算" % msrp, "需核")
    return table


def lookup(table, name):
    k = norm(name)
    for key, v in table.items():
        if key and (key in k or k in key):
            return v
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bass", type=float, default=3)
    ap.add_argument("--presence", type=float, default=2)
    ap.add_argument("--treble", type=float, default=1)
    ap.add_argument("--peak-guard", type=float, default=4.5)
    ap.add_argument("--avoid-weight", type=float, default=0.6)
    ap.add_argument("--dislike", action="append", default=["Edition XS"])
    ap.add_argument("--per-tier", type=int, default=4)
    ap.add_argument("--scan", type=int, default=200, help="在前 N 名里做分档")
    a = ap.parse_args()

    rows, meta = engine.rank({"bass": a.bass, "presence": a.presence, "treble": a.treble,
                              "peak_guard": a.peak_guard, "dislike": a.dislike,
                              "avoid_weight": a.avoid_weight})
    prices = load_prices()

    print("画像：低频%+.1f / presence%+.1f / 高频%+.1f ｜ 峰护栏%.1f ｜ 避 %s"
          % (a.bass, a.presence, a.treble, a.peak_guard, a.dislike))
    print("基准：Harman 系 ｜ 样本 %d 型号\n" % meta["n_models"])

    for lo, hi, label in TIERS:
        bucket = []
        for r in rows[:a.scan]:
            p = lookup(prices, r["name"])
            if p and lo <= p[0] < hi:
                bucket.append((r, p))
            if len(bucket) >= a.per_tier:
                break
        print("### %s" % label)
        if not bucket:
            print("   （前 %d 名里没有已核到价格的型号）" % a.scan)
        for r, (cny, src, note) in bucket:
            print("   %-38s 得分%.1f  ¥%d  %s  %s" % (r["name"][:38], r["score"], cny, engine.badge(r), note))
            print("       价源：%s" % src)
        print()

    # 高匹配但无价格
    miss = [r for r in rows[:40] if not lookup(prices, r["name"])]
    print("### 匹配高但**价格待核**（不猜价，需自行查）")
    for r in miss[:12]:
        print("   %-38s 得分%.1f  %s" % (r["name"][:38], r["score"], engine.badge(r)))

    # 二手参考（闲鱼行情帖来源；挂牌价≠成交价）
    used = load_used()
    print("\n### 二手行情参考（**闲鱼无法抓取，来自行情帖；下单前自行核成交价**）")
    if not used:
        print("   （暂无已核到的二手行情）")
    for r in rows[:a.scan]:
        u = lookup_used(used, r["name"])
        if u:
            print("   %-38s 得分%.1f  二手约 ¥%d–%d" % (r["name"][:38], r["score"], u["low"], u["high"]))
            print("       来源：%s ｜ 注意：%s" % (u["src"], u.get("note", "")))


if __name__ == "__main__":
    main()
