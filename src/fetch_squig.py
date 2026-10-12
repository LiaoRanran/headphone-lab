#!/usr/bin/env python3
"""第二数据源：squig.link 各测评人子站（持续更新，覆盖 2024–2026 新品）。

结构： https://<site>.squig.link/data/phone_book.json   → 索引（品牌组 → 型号 → 文件 stem）
        https://<site>.squig.link/data/<stem> L.txt     → 曲线（分声道/分次；graphtool.js 用 fileName+" "+s+n）

用法: python fetch_squig.py [--sites a,b] [--limit N] [--workers 16]
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import tempfile
import threading
import time
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "squig_raw")
TMP = tempfile.gettempdir()

SITES = ["kuulokenurkka", "listener", "vsg", "precog", "jaytiss", "hbb", "kr0mka", "dhrme"]
SUFFIXES = (" L", " R", " L1", " R1", "")

HP_BRANDS = [
    "sennheiser", "hifiman", "beyerdynamic", "focal", "meze", "audeze", "akg",
    "audio-technica", "audio technica", "dan clark", "harmonicdyne", "sivga", "sendy",
    "gold planar", "superlux", "takstar", "abyss", "raal", "mysphere", "kennerton",
    "erzetich", "ollo", "koss", "grado", "stax", "bowers", "denon", "drop", "massdrop",
    "aune", "philips", "shure", "pioneer", "yamaha", "austrian audio", "verum",
    "nectar sound", "modhouse", "zmf", "fostex", "warwick", "es labs",
]


def curl(url, out, to="25"):
    subprocess.run(["curl.exe", "-sL", "--max-time", to, "-o", out, url], capture_output=True)
    return os.path.exists(out) and os.path.getsize(out) > 60


def safe(s):
    return re.sub(r"[^\w\u4e00-\u9fff.+\-()\[\] ]+", "_", s).strip()


_lock = threading.Lock()
_done = [0]


def grab_one(job):
    site, stem, sub = job
    for sfx in SUFFIXES:
        cand = os.path.join(sub, safe(stem + sfx) + ".txt")
        if os.path.exists(cand) and os.path.getsize(cand) > 60:
            return stem + sfx, cand
        url = "https://%s.squig.link/data/%s.txt" % (site, urllib.parse.quote(stem + sfx))
        if curl(url, cand):
            return stem + sfx, cand
    return None, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sites", default=",".join(SITES))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()

    jobs = []
    for site in a.sites.split(","):
        site = site.strip()
        if not site:
            continue
        pb = os.path.join(TMP, "pb_%s.json" % site)
        d = None
        for attempt in range(4):
            subprocess.run(["curl.exe", "-sL", "--max-time", "60", "-o", pb,
                            "https://%s.squig.link/data/phone_book.json" % site],
                           capture_output=True)
            try:
                d = json.load(open(pb, encoding="utf-8-sig", errors="ignore"))
                break
            except Exception:
                time.sleep(1.5)
        if d is None:
            print("!! %s: phone_book 取不到/解析失败" % site)
            continue
        sub = os.path.join(RAW, site)
        os.makedirs(sub, exist_ok=True)
        for g in d:
            brand = g.get("name") or ""
            if not any(k in brand.lower() for k in HP_BRANDS):
                continue
            for ph in g.get("phones", []):
                stems = ph.get("file")
                stems = stems if isinstance(stems, list) else ([stems] if stems else [])
                for stem in stems:
                    if stem:
                        jobs.append((site, stem, sub))
    if a.limit:
        jobs = jobs[:a.limit]
    print("候选文件 %d 个，并发 %d" % (len(jobs), a.workers))

    rows = []
    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for job, (used, path) in zip(jobs, ex.map(grab_one, jobs)):
            site, stem, _ = job
            with _lock:
                _done[0] += 1
                if _done[0] % 40 == 0:
                    print("   ... %d/%d（已成 %d）" % (_done[0], len(jobs), len(rows)))
            if used:
                rows.append({"site": site, "stem": used, "brand_hint": stem,
                             "txt": os.path.relpath(path, ROOT)})

    man = os.path.join(DATA, "squig_manifest.jsonl")
    merged = {}
    if os.path.exists(man):
        for ln in open(man, encoding="utf-8"):
            try:
                x = json.loads(ln)
                merged[(x["site"], x["stem"])] = x
            except Exception:
                pass
    for r in rows:
        merged[(r["site"], r["stem"])] = r
    with open(man, "w", encoding="utf-8") as f:
        for k in sorted(merged):
            f.write(json.dumps(merged[k], ensure_ascii=False) + "\n")
    print("\n本次 %d ｜ 累计 %d → %s" % (len(rows), len(merged), man))
    if rows:
        p = os.path.join(ROOT, rows[0]["txt"])
        print("\n样例 %s：\n%s" % (rows[0]["txt"], open(p, encoding="utf-8", errors="ignore").read()[:220]))


if __name__ == "__main__":
    main()
