#!/usr/bin/env python3
"""实测匹配引擎 v1.2（可复用模块 + CLI）。

    from engine import rank
    rows, meta = rank({"bass":3, "presence":2, "treble":1, "peak_guard":4.5, "dislike":["Edition XS"]})

基准 devt = 曲线 − 数据源自带目标曲线（oratory1990 ≈ Harman 系）。
CLI:
    python engine.py --bass 3 --presence 2 --treble 1 --peak-guard 4.5 --dislike "Edition XS" --top 20
"""
import argparse
import json
import math
import os
import statistics as st
import sys

_SRC = os.path.dirname(os.path.abspath(__file__))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

ROOT = os.path.dirname(_SRC)
DATA = os.path.join(ROOT, "data")
_CACHE = {}


def load():
    if "d" not in _CACHE:
        with open(os.path.join(DATA, "features.json"), encoding="utf-8") as f:
            _CACHE["d"] = json.load(f)
    return _CACHE["d"]


def group(models):
    g = {}
    for m in models:
        g.setdefault(m["name"], []).append(m)
    out = {}
    for name, ms in g.items():
        n = len(ms[0]["dev"])
        med = [st.median([m["dev"][i] for m in ms]) for i in range(n)]
        medt = [st.median([m["devt"][i] for m in ms]) for i in range(n)]
        spread = (sum(abs(m["dev"][i] - med[i]) for m in ms for i in range(n)) / (len(ms) * n)) \
            if len(ms) > 1 else None
        out[name] = {"curve": med, "curve_t": medt, "sources": [m["source"] for m in ms],
                     "spread": spread, "features_t": ms[0].get("features_t", ms[0]["features"])}
    return out


def base_weight():
    """感知权重：ISO 226:2003 等响曲线 + floor（见 src/psycho.py）。

    为什么不是「直接用等响敏感度」——实测结论（src/weight_sensitivity.py）：
    floor 从 0.05 提到 0.30 时，与历史手调权重的前 10 重合从 4 个升到 7 个、
    Spearman 从 0.918 升到 0.967；而 floor=0.05（纯敏感度）会把
    **游戏耳麦 PC38X 顶到第一名**——因为 20 Hz 与 >10 kHz 的权重被压到接近 0，
    "只有中频好"的耳机于是占便宜。

    根因：**纯音敏感度 ≠ 宽带谱差的显著性**。等响曲线说 20 Hz 在 80 方时
    敏感度仅 ~1e-4，直接当权重等于完全忽略超低频；但音乐少了超低频你
    **听得出来**（声音变薄）。floor 就是"谱差在任何频段都可闻"的下限。

    floor=0.30 的取值依据：与历史手调权重的形状最接近（前 10 重合 7/10），
    而手调值是多年听音验证过的、不是拍的。改这个值前先跑
    `python src/weight_sensitivity.py` 看排名稳定性。
    """
    import psycho
    return psycho.weight_fn(phon=80.0, floor=0.30)


def rank(cfg, w_override=None):
    """cfg: bass/presence/treble/peak_guard/like/dislike/basis/min_sources

    w_override: 直接替换感知权重函数。供敏感性分析用
    （见 src/weight_sensitivity.py）——默认权重是手调的，
    必须能量化它有多要紧，否则排名不可信。
    """
    basis = cfg.get("basis", "target")
    pg = cfg.get("peak_guard", 4.5)
    D = load()
    grid, n = D["grid"], len(D["grid"])
    G = group(D["models"])
    W = w_override if w_override is not None else base_weight()
    music_note = None
    if cfg.get("music"):
        import music_profile as mp
        prof = (mp.load_profiles().get("sourced", {}).get(cfg["music"]) or {}).get("profile")
        if prof:
            W = mp.weight_fn(prof)
            music_note = cfg["music"]
    if cfg.get("music_trend"):
        # 用本项目自测的目录级频谱走向设权重（trackscore 只覆盖电子乐，
        # J-pop / 人声 / 动漫 OST 没有任何公开档案，必须用自测数据）
        import music_trend as mt
        tr = mt.load_trend(cfg["music_trend"])
        if tr:
            W = mt.weight_fn_from_bands(tr["bands"])
            music_note = tr["name"]
    KEY = "curve_t" if basis == "target" else "curve"
    if cfg.get("min_sources", 1) > 1:
        G = {k: v for k, v in G.items() if len(v["sources"]) >= cfg["min_sources"]}
    if cfg.get("wired_only", True):
        import filters
        G = {k: v for k, v in G.items() if filters.is_wired_over_ear(k)}
    if cfg.get("only"):
        # 显式白名单：测量库里小众型号远多于日常能买到的型号，
        # 不做这一步会把「有实测」误当成「买得到/认得出」。
        keys = [k.strip().lower() for k in cfg["only"]]
        G = {k: v for k, v in G.items() if any(x in k.lower() for x in keys)}
    if cfg.get("open_only", False):
        import filters as _f
        G = {k: v for k, v in G.items() if not _f.classify(k) == "closed"}  # 预留：需 type 字段时才严谨

    def match(name, pats):
        return any(p.lower() in name.lower() for p in pats)
    chosen = [G[k][KEY] for k in G if match(k, cfg.get("like", []))]
    avoided = [G[k][KEY] for k in G if match(k, cfg.get("dislike", []))]
    dw = None
    if chosen and avoided:
        cl = [st.mean([c[i] for c in chosen]) for i in range(n)]
        av = [st.mean([a[i] for a in avoided]) for i in range(n)]
        d = [abs(cl[i] - av[i]) for i in range(n)]
        mx = max(d) or 1.0
        dw = [0.5 + 1.5 * (x / mx) for x in d]

    base = [st.mean([c[i] for c in chosen]) for i in range(n)] if chosen else [0.0] * n
    # 「讨厌」样本方向从目标里减掉（无"喜欢"锚点时唯一可用的个性化信号；样本少 → 低置信）
    avoid_dir = None
    if avoided:
        av = [st.mean([a[i] for a in avoided]) for i in range(n)]
        mu = sum(av) / n
        avoid_dir = [x - mu for x in av]
    lam = cfg.get("avoid_weight", 0.6)
    target = []
    for i, f in enumerate(grid):
        d = 0.0
        if f <= 200:
            # 低频目标按频段加权：<40Hz 只给 0.4（开放式物理滚降，且"中等低频"不含超低频要求）
            if f < 40:
                d = cfg.get("bass", 0.0) * 0.4
            elif f < 60:
                d = cfg.get("bass", 0.0) * 0.7
            elif f <= 120:
                d = cfg.get("bass", 0.0) * 1.0
            else:
                d = cfg.get("bass", 0.0) * 0.7
        if 2000 <= f <= 5000:
            d += cfg.get("presence", 0.0)
        if 6000 <= f <= 12000:
            d += cfg.get("treble", 0.0)
        t = base[i] + d
        if avoid_dir:
            t -= lam * avoid_dir[i]
        target.append(t)

    rows = []
    # ---- 积分域：默认 1/12 倍频程网格（旧行为），可切到 Bark 临界频带 ----
    # 为什么要这个开关：网格按恒定比例分布，而人耳按临界频带分析。
    # 实测 50 Hz 处 1/12 倍频带比 Bark 带窄 26 倍 —— 等于 50Hz 上同样的 dB 误差
    # 对 RMS 的贡献是 1kHz 的 26 倍。感知权重部分补偿，但"权重"和"积分域"
    # 是两件事，用一个参数干两件事会互相牵扯。
    bark_edges = None
    band_w: dict[int, float] = {}
    if cfg.get("bark"):
        import perceptual
        bark_edges = perceptual.bark_edges()
        acc: dict[int, list[float]] = {}
        for i, k in enumerate(perceptual.band_index(grid, bark_edges)):
            if k >= 0:
                acc.setdefault(k, []).append(W(grid[i]) * (dw[i] if dw else 1.0))
        band_w = {k: sum(v) / len(v) for k, v in acc.items()}

    for name, g in G.items():
        curve = g[KEY]
        if bark_edges is not None:
            # RMS 与峰护栏**同一个积分域**：都取 Bark 带内均值，
            # 否则等于把两套口径混进一个分数。
            bm = perceptual.bark_band_means(curve, target, grid, bark_edges)
            se = sw = sp = 0.0
            for k, m in bm.items():
                w = band_w.get(k, 1.0)
                sw += w
                se += w * m * m
                if m > 2.0:
                    sp += w * (m - 2.0) ** 2
            rms = math.sqrt(se / sw) if sw > 0 else 0.0
            peak = math.sqrt(sp / sw) if sw > 0 else 0.0
        else:
            sw = se = sp = 0.0
            for i in range(n):
                w = W(grid[i]) * (dw[i] if dw else 1.0)
                dd = curve[i] - target[i]
                sw += w
                se += w * dd * dd
                sp += w * max(0.0, dd - 2.0) ** 2
            rms = math.sqrt(se / sw)
            peak = math.sqrt(sp / sw)
        s = 100 - 7.0 * rms - 6.0 * peak
        pf = max(curve[i] for i in range(n) if 5000 <= grid[i] <= 10000)
        if pf > pg:
            s -= 4.0 * (pf - pg)
        if avoided:
            am = [st.mean([x[i] for x in avoided]) for i in range(n)]
            s -= 0.8 * max(0.0, 3.0 - math.sqrt(sum((curve[i] - am[i]) ** 2 for i in range(n)) / n))
        if chosen:
            cm = [st.mean([x[i] for x in chosen]) for i in range(n)]
            s += 0.6 * max(0.0, 3.0 - math.sqrt(sum((curve[i] - cm[i]) ** 2 for i in range(n)) / n))
        rows.append({"name": name, "score": s, "rms": rms, "pf": pf,
                     "spread": g["spread"], "n_src": len(g["sources"]), "f": g["features_t"],
                     # 曲线与目标一起带出来，供 tier_report 做频率维 bootstrap。
                     # 不让 tier_report 自己重算——那会造出第二套口径，
                     # 分档和得分就对不上了。
                     "_curve": curve, "_target": target})
    rows.sort(key=lambda r: -r["score"])
    meta = {"basis": basis, "like": cfg.get("like", []), "dislike": cfg.get("dislike", []),
            "n_models": len(rows), "n_curves": len(D["models"]), "music": music_note,
            # 感知权重与峰护栏也带出来，bootstrap 必须用同一套参数重算，
            # 否则分档与得分的口径不一致
            "_W": W, "_pg": pg, "_grid": grid}
    return rows, meta


def badge(r):
    if r["spread"] is None:
        return "单源"
    return ("跨源一致 %.1f(%d源)" if r["spread"] <= 1.5 else "跨源分歧 %.1f(%d源)") % (r["spread"], r["n_src"])


def tier_report(rows, meta, n=12, n_boot=200, seed=0):
    """把前 n 名按 bootstrap 置信区间重叠分档，并说清这串数字能承受多重的解读。

    **为什么必须有这个**：得分是一个加权 RMS 偏差，把它当排行榜用等于按噪声排序。
    stats.py 实测：Harman 基准下前 14 名 CI 全部重叠、连 #1 与 #2 都不可区分。
    既然自己的统计这么说，输出却不打印分档，就是在纵容误读。

    分档规则：按 CI 上限降序扫，上限不低于当前档最低下限的并入同档。
    """
    import random
    import stats
    W, pg, grid = meta["_W"], meta["_pg"], meta["_grid"]
    idx_all = list(range(len(grid)))
    rng = random.Random(seed)

    out = []
    for r in rows[:n]:
        curve, target = r["_curve"], r["_target"]
        base = stats._score(curve, target, grid, W, pg, idx_all)
        vals = []
        for _ in range(n_boot):
            sub = rng.choices(idx_all, k=len(idx_all))
            vals.append(stats._score(curve, target, grid, W, pg, sub))
        vals.sort()
        out.append({"name": r["name"], "score": base,
                    "lo": vals[int(0.025 * len(vals))],
                    "hi": vals[int(0.975 * len(vals))]})

    out.sort(key=lambda x: -x["hi"])
    tiers, cur = [], [out[0]]
    for o in out[1:]:
        if o["hi"] >= cur[-1]["lo"] - 1e-9:
            cur.append(o)
        else:
            tiers.append(cur)
            cur = [o]
    tiers.append(cur)

    print("\n── 分档（同档内名次差异不可解读）────────────────────────────")
    for i, t in enumerate(tiers, 1):
        print("  第 %d 档（%2d 款，CI 跨度 %.1f–%.1f）：%s"
              % (i, len(t), min(x["lo"] for x in t), max(x["hi"] for x in t),
                 "、".join(x["name"][:20] for x in t)))
    return tiers


#: 表头。写「候选（非排名）」而不写「排名」：ordinal 编号会让人以为可以按序下单，
#: 而下面的分档会说清同档内根本不可区分。**措辞要与数据的效力相称。**
TABLE_HEADER = ("序", "候选（非排名）", "得分", "RMS", "峰dB", "跨源")


DISCLAIMER = """\
────────────────────────────────────────────────────────────────────
这是**推荐候选集**，不是测量结论，更不是"排名"。

这个分数只衡量一件事：实测频响曲线与某条目标曲线的加权 RMS 偏差。
它**不能**告诉你：声场、结像、瞬态、大音量失真、单元一致性、佩戴、
隔音、做工、驱动难度、保值率——这些要么测不到，要么没测。
相邻名次的分差在统计上几乎必然不显著（下面有分档），
把它当排行榜按序下单，等于按噪声下单。

低频目标"该补多少"取决于你的音量与偏好，**测量推不出来**，
所以本工具只负责告诉你"补了会怎样"，不替你决定补多少。
────────────────────────────────────────────────────────────────────"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basis", choices=["target", "population"], default="target")
    ap.add_argument("--bass", type=float, default=0.0)
    ap.add_argument("--presence", type=float, default=0.0)
    ap.add_argument("--treble", type=float, default=0.0)
    ap.add_argument("--peak-guard", type=float, default=4.5)
    ap.add_argument("--avoid-weight", type=float, default=0.6, help="把「讨厌」样本方向从目标里减掉的强度")
    ap.add_argument("--like", action="append", default=[])
    ap.add_argument("--dislike", action="append", default=[])
    ap.add_argument("--explain", default=None)
    ap.add_argument("--min-sources", type=int, default=1)
    ap.add_argument("--music", default=None,
                    help="用音乐能量分布设定频段权重（electronic_general/deep_house/trance/drum_and_bass/bass_music_dubstep）")
    ap.add_argument("--music-trend", default=None,
                    help="用本项目自测的目录档案设权重（见 data/music_trends.json，如 'BA OST'）")
    ap.add_argument("--bark", action="store_true",
                    help="积分域改用 Bark 临界频带（默认 1/12 倍频程网格）。"
                         "实测 50Hz 处后者比前者窄 26 倍，低频误差被过度代表")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--mainstream", action="store_true",
                    help="只保留国内能买到、常见于讨论的型号（见 data/mainstream_match.json "
                         "的 _whitelist）。默认不开——测量库天然偏向发烧友与小众型号，"
                         "但这个开关是主观的「常见」判断，不是测量结论，所以不默认开")
    ap.add_argument("--no-tiers", dest="tiers", action="store_false",
                    help="不打印 bootstrap 分档（默认打印）")
    ap.add_argument("--boot", type=int, default=200, help="分档的 bootstrap 次数")
    a = ap.parse_args()
    cfg = vars(a)
    if a.mainstream:
        import json
        import os
        p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                         "mainstream_match.json")
        with open(p, encoding="utf-8") as f:
            cfg["only"] = json.load(f)["_whitelist"]

    rows, meta = rank(cfg)
    cn = "相对来源目标曲线(Harman 系)" if meta["basis"] == "target" else "相对人群基线"
    print("基准：%s ｜ 目标 低频%+.1f / presence%+.1f / 高频%+.1f ｜ 峰护栏 %.1f"
          % (cn, a.bass, a.presence, a.treble, a.peak_guard))
    if meta["like"] or meta["dislike"]:
        print("锚点：喜欢 %s ｜ 讨厌 %s（锚点少→置信低）" % (meta["like"] or "—", meta["dislike"] or "—"))
    print("样本 %d 型号 / %d 条曲线\n" % (meta["n_models"], meta["n_curves"]))
    print(DISCLAIMER)
    # 表头写「候选」不写「排名」——见 TABLE_HEADER 注释
    print("\n%-3s %-40s %6s %6s %7s %s" % TABLE_HEADER)
    for i, r in enumerate(rows[:a.top], 1):
        print("%-3d %-40s %6.1f %6.1f %+7.1f %s" % (i, r["name"][:40], r["score"], r["rms"], r["pf"], badge(r)))
    if a.tiers:
        tier_report(rows, meta, n=min(a.top, 12), n_boot=a.boot)

    if a.explain:
        for r in [x for x in rows if a.explain.lower() in x["name"].lower()][:3]:
            f = r["f"]
            print("\n== %s ==  得分 %.1f ｜ %s" % (r["name"], r["score"], badge(r)))
            print("   低频 %+.1f ｜ 下潜 %+.1f ｜ 中频 %+.1f ｜ presence %+.1f ｜ 高频 %+.1f ｜ 空气感 %+.1f"
                  % (f["bass"], f["sub_extension_db"], f["mid"], f["presence"], f["treble"], f["air"]))
            print("   1k-10k 倾角 %+.2f dB/oct ｜ 5-10k 峰 %+.1fdB@%sHz ｜ 粗糙度 %.2f"
                  % (f["tilt_1k_10k"], f.get("peak_5_10k_db", 0), f.get("peak_5_10k_hz", "?"), f["roughness"]))


if __name__ == "__main__":
    main()
