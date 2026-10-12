#!/usr/bin/env python3
"""拉取实测频响（AutoEq 镜像）—— 头戴式数据层。

数据源 : AutoEq (github.com/jaakkopasanen/AutoEq)，经 jsDelivr CDN 定点取原始 CSV。
CSV 列 : frequency,raw,smoothed,error,error_smoothed,equalization,...,target
         raw=实测曲线 ; target=该来源使用的目标曲线 ; error=raw-target

用法:
    python fetch_fr.py --sources oratory1990,crinacle,Rtings,Innerfidelity [--limit N] [--workers 12]
    python fetch_fr.py --list-sources
"""
import argparse
import concurrent.futures as cf
import json
import os
import re
import subprocess
import sys
import tempfile
import urllib.parse

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
FR = os.path.join(DATA, "fr")
# 镜像按顺序回退（jsDelivr 在批量抓取后会被限流，故放最后）
MIRRORS = [
    "https://ghproxy.net/https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/",
    "https://gh-proxy.com/https://raw.githubusercontent.com/jaakkopasanen/AutoEq/master/results/",
    "https://cdn.jsdelivr.net/gh/jaakkopasanen/AutoEq@master/results/",
]
JS = MIRRORS[0]      # 索引也用第一个可用镜像
TMP = tempfile.gettempdir()
ENTRY = re.compile(r"^- \[(?P<name>[^\]]+)\]\((?P<path>\./[^\)]+)\) by (?P<src>[^ ]+)(?: on (?P<coupler>.+))?$", re.M)


def curl(url, out):
    for _ in range(2):
        subprocess.run(["curl.exe", "-sL", "--max-time", "60", "-o", out, url], capture_output=True)
        if os.path.exists(out) and os.path.getsize(out) > 200:
            return True
    return False


def load_index():
    os.makedirs(DATA, exist_ok=True)
    cache = os.path.join(DATA, "autoeq_index.md")
    if not os.path.exists(cache) or os.path.getsize(cache) < 10000:
        if not curl(JS + "INDEX.md", cache):
            sys.exit("!! 无法获取 AutoEq INDEX.md（检查网络/curl）")
    with open(cache, encoding="utf-8-sig", errors="ignore") as f:
        return f.read()


def parse(txt):
    out = []
    for m in ENTRY.finditer(txt):
        d = m.groupdict()
        if "over-ear" not in d["path"]:
            continue
        out.append(d)
    return out


def safe(s):
    return re.sub(r"[^\w\u4e00-\u9fff.+\-()\[\] ]+", "_", s).strip()


def grab(e, sub):
    name, path = e["name"], e["path"][2:]                 # 路径已 URL 编码，勿再编
    rel = path + "/" + urllib.parse.quote(name) + ".csv"
    dst = os.path.join(sub, safe(name) + ".csv")
    if os.path.exists(dst) and os.path.getsize(dst) > 200:
        return True
    for i, m in enumerate(MIRRORS):
        tmp = os.path.join(TMP, "hpl_%s.csv" % abs(hash(rel + str(i))))
        if curl(m + rel, tmp):
            os.replace(tmp, dst)
            return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="oratory1990")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--catalog", default=os.path.join(DATA, "fr_manifest.jsonl"))
    a = ap.parse_args()

    entries = parse(load_index())
    if a.list_sources:
        from collections import Counter
        for s, n in Counter(e["src"] for e in entries).most_common():
            print("  %-22s %d" % (s, n))
        return

    want = [s.strip() for s in a.sources.split(",") if s.strip()]
    sel = [e for e in entries if e["src"] in want]
    if a.limit:
        sel = sel[:a.limit]
    print("目标 %d 条（来源 %s），并发 %d" % (len(sel), want, a.workers))

    os.makedirs(FR, exist_ok=True)
    results, done = [], [0]
    lock = __import__("threading").Lock()

    def work(e):
        sub = os.path.join(FR, safe(e["src"]))
        os.makedirs(sub, exist_ok=True)
        ok = grab(e, sub)
        with lock:
            done[0] += 1
            if done[0] % 50 == 0:
                print("   ... %d/%d" % (done[0], len(sel)))
        if ok:
            return {"name": e["name"], "source": e["src"], "coupler": e["coupler"],
                    "csv": os.path.relpath(os.path.join(sub, safe(e["name"]) + ".csv"), ROOT)}
        return None

    with cf.ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, sel):
            if r:
                results.append(r)

    merged = {}
    if os.path.exists(a.catalog):
        with open(a.catalog, encoding="utf-8") as f:
            for ln in f:
                try:
                    d = json.loads(ln)
                    merged[(d["source"], d["name"])] = d
                except Exception:
                    pass
    for r in results:
        merged[(r["source"], r["name"])] = r
    with open(a.catalog, "w", encoding="utf-8") as f:
        for k in sorted(merged):
            f.write(json.dumps(merged[k], ensure_ascii=False) + "\n")
    print("\n本次成功 %d / %d；清单累计 %d → %s" % (len(results), len(sel), len(merged), a.catalog))
    from collections import Counter
    print("累计按来源:", dict(Counter(v["source"] for v in merged.values())))


if __name__ == "__main__":
    main()
