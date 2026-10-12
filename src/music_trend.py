#!/usr/bin/env python3
"""目录级频谱走向：抽样代表曲目 → 全采样率测各频段能量与频谱倾斜。

与 music_profile.py 的区别（也是为什么要重写）：
  1. music_profile 把文件解码成 **22.05kHz 单声道**，奈奎斯特只有 11kHz，
     于是 6–16k 与 16–20k 两段必然测错（高频段恒为空）。
     本脚本用 soundfile 全采样率解码。
  2. music_profile 只测「一首歌」，无法回答「这个目录整体是什么走向」。
     本脚本按目录抽样、跨专辑铺开，给均值与离散度。

口径与 trackscore.ai 的 7 段一致（% 线性功率），便于与已发布档案对照。

输出：
  data/music_trends.json    每目录的频段占比、频谱倾斜、代表曲目明细
  out/music/trends.png      各目录平均频谱叠加图（归一化后比"走向"）
"""
import argparse
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
OUT = os.path.join(ROOT, "out", "music")

# 与 trackscore.ai 对齐的 7 段（Hz）
BANDS = [("sub", 20, 60), ("bass", 60, 250), ("low_mid", 250, 500),
         ("mid", 500, 2000), ("upper_mid", 2000, 6000), ("high", 6000, 16000),
         ("air", 16000, 20000)]
BAND_CN = {"sub": "超低", "bass": "低频", "low_mid": "中低", "mid": "中频",
           "upper_mid": "中高", "high": "高频", "air": "空气"}

# 与 src/fr_features.py 的 BANDS 完全一致（Hz）—— 用于和耳机的 band 偏差对齐。
# trackscore 口径与本表频段边界不同（60–250 vs 60–200、2–6k vs 2–5k），
# **不能混用**：把 trackscore 的占比和耳机的 dB 偏差相加是错的。
BANDS_LAB = [("sub_bass", 20, 60), ("bass", 60, 200), ("low_mid", 200, 500),
             ("mid", 500, 2000), ("presence", 2000, 5000), ("treble", 5000, 10000),
             ("air", 10000, 20000)]

# 默认目录（用户曲库）；路径含 [ ] 与全角字符，故写进脚本而非命令行传参
DEFAULT_ROOTS = [
    ("Ado ベストアドバム", "C:/Users/ASUS/Desktop/Music/[Hi-Res][250319]Adoのベストアドバム[48kHz／24bit][FLAC]"),
    ("超かぐや姫！劇中歌集", "C:/Users/ASUS/Desktop/Music/[Hi-Res][260123]映画『超かぐや姫！』劇中歌集"),
    ("ReoNa - HUMAN", "C:/Users/ASUS/Desktop/Music/[Hires]ReoNa - HUMAN 2023 [96kHz24bit]"),
    ("ELECTROCUTICA", "C:/Users/ASUS/Desktop/Music/ELECTROCUTICA"),
    ("BA OST", "C:/Users/ASUS/Desktop/Music/BA OST"),
    ("ray of hope", "C:/Users/ASUS/Desktop/Music/ray of hope"),
]

AUDIO_EXT = (".flac", ".wav", ".aiff", ".aif", ".mp3", ".m4a", ".aac", ".ogg", ".opus")
SEG_SECONDS = 60.0          # 取中段 60 秒；比全曲快，且避开淡入淡出
NPERSEG = 16384


def list_audio(root):
    out = []
    for dp, _, fns in os.walk(root):
        for fn in fns:
            if fn.lower().endswith(AUDIO_EXT):
                out.append(os.path.join(dp, fn))
    return sorted(out)


def pick_representative(files, n, spread_across_dirs=True):
    """跨子目录铺开抽样：先按目录分组各取一条，再回填到 n 条。

    跨专辑抽样比「同一专辑取 n 首」更能代表一个目录的整体走向。
    """
    if not files:
        return []
    if n >= len(files):
        return files
    if spread_across_dirs:
        groups = {}
        for p in files:
            groups.setdefault(os.path.dirname(p), []).append(p)
        keys = sorted(groups)
        picked, i = [], 0
        while len(picked) < n:
            g = groups[keys[i % len(keys)]]
            if g:
                picked.append(g.pop(len(g) // 2))      # 取每组中位那首，避免开场曲偏差
            i += 1
            if i > 500:
                break
        # 若目录数少于 n，剩下的名额按顺序补
        rest = [p for p in files if p not in picked]
        picked += rest[:max(0, n - len(picked))]
        return sorted(set(picked))
    step = max(1, len(files) // n)
    return files[::step][:n]


def curve_of(path, nperseg=NPERSEG, bands=None):
    """全曲分块 Welch PSD → (f, dB, 每频段线性功率占比, fs)。

    两处按审计结论修正（`src/audit.py`）：
    1. **不再先平均 L/R 再算谱**。单声道化的交叉项 2·Re{L·R*} 会让左右相关的
       成分叠加、不相关的抵消——实测在电子乐上把超低段高估最多 5.82 个百分点、
       中频低估 4.09，且 4 首方向一致。改为**各自算功率谱再平均**（标准做法）。
    2. **不再只取中段 60 秒**。实测中段 vs 全曲最大差 8.56 个百分点（低频）。
       改为分块读**全曲**并累计，保留 intro/副歌的结构差异。
    """
    info = sf.info(path)
    fs, total = info.samplerate, info.frames
    acc, f = None, None
    pos, chunk = 0, int(30 * fs)          # 30s 一块，内存安全；块边界只影响 1/1750 帧
    while pos < total:
        n = min(chunk, total - pos)
        if n < nperseg * 2:
            break
        x, _ = sf.read(path, start=pos, frames=n, dtype="float32", always_2d=True)
        pos += n
        kw = dict(fs=fs, nperseg=nperseg, window="hann",
                  noverlap=nperseg // 2, scaling="spectrum")
        if x.shape[1] >= 2:                       # 各自算功率谱再平均，避开交叉项
            f, PL = signal.welch(x[:, 0], **kw)
            _, PR = signal.welch(x[:, 1], **kw)
            P = (PL + PR) / 2.0
        else:
            f, P = signal.welch(x[:, 0], **kw)
        acc = P if acc is None else acc + P
    if acc is None:
        raise ValueError("无可分析内容")
    db = 10 * np.log10(acc + 1e-20)
    tot = float(acc[(f >= 20) & (f <= 20000)].sum()) or 1.0
    # 注意别用 `bands` 同时当「传入的频段表」和「输出字典」——
    # 早先写成 bands={} 后再 for ... in (bands or BANDS)，循环体直接变空。
    band_def = bands if bands is not None else BANDS
    out = {}
    for name, lo, hi in band_def:
        msk = (f >= lo) & (f < hi)
        out[name] = round(float(acc[msk].sum() / tot * 100), 2)
    return f, db, out, fs


def tilt_db_per_oct(f, db):
    """100Hz–10kHz 的最小二乘倾斜（dB/octave）。

    正=能量随频率上升，负=典型的"低频主导"。
    这是判断「音乐本身缺不缺高频」最直接的单一数字。
    """
    m = (f >= 100) & (f <= 10000)
    lf = np.log2(f[m])
    y = db[m]
    if lf.size < 8:
        return None
    k = np.polyfit(lf, y, 1)[0]
    return round(float(k), 2)


def analyze_folder(name, root, n_pick):
    files = list_audio(root)
    if not files:
        return None
    picked = pick_representative(files, n_pick)
    rows, curves = [], []
    for p in picked:
        try:
            f, db, bands, fs = curve_of(p)
        except Exception as e:                                      # noqa: BLE001
            rows.append({"file": os.path.basename(p), "error": str(e)[:100]})
            continue
        t = tilt_db_per_oct(f, db)
        rows.append({
            "file": os.path.relpath(p, root),
            "sr": fs,
            "bands_pct": bands,
            "tilt_db_per_oct": t,
            "low_share": round(bands["sub"] + bands["bass"], 2),
            "presence_share": round(bands["upper_mid"], 2),
            "air_share": round(bands["high"] + bands["air"], 2),
        })
        # 曲线重采样到公共 1/12 倍频程网格以便叠加
        grid = np.geomspace(30, 20000, 121)
        curves.append(np.interp(np.log2(grid), np.log2(f[f > 0]), db[f > 0]))
        print("    ok  %-52s sr=%d tilt=%s" % (os.path.basename(p)[:52], fs, t), flush=True)

    ok = [r for r in rows if "bands_pct" in r]
    if not ok:
        return None
    agg = {}
    for b, _, _ in BANDS:
        v = np.array([r["bands_pct"][b] for r in ok])
        agg[b] = {"mean": round(float(v.mean()), 2), "std": round(float(v.std()), 2)}
    tilts = [r["tilt_db_per_oct"] for r in ok if r["tilt_db_per_oct"] is not None]
    S = np.array(curves)
    # 归一化：减去各自 200–2k 均值，只比"走向"不比绝对电平
    grid = np.geomspace(30, 20000, 121)
    ref = (grid >= 200) & (grid <= 2000)
    norm = S - S[:, ref].mean(axis=1, keepdims=True)
    return {
        "name": name, "root": root, "n_files": len(files), "n_analyzed": len(ok),
        "bands": agg,
        "low_share": round(float(np.mean([r["low_share"] for r in ok])), 2),
        "presence_share": round(float(np.mean([r["presence_share"] for r in ok])), 2),
        "air_share": round(float(np.mean([r["air_share"] for r in ok])), 2),
        "tilt_db_per_oct": round(float(np.mean(tilts)), 2) if tilts else None,
        "tilt_std": round(float(np.std(tilts)), 2) if tilts else None,
        "tracks": rows,
        # 归一化后的平均曲线一并存盘：换绘图样式时可 --replot，不必重新解码 24 个 FLAC
        "curve_grid_hz": [round(float(g), 1) for g in grid],
        "curve_mean_db": [round(float(v), 2) for v in np.mean(norm, axis=0)],
    }


def load_trend(name):
    """按目录名（模糊匹配）读回一条自测档案，给引擎当频段权重来源。"""
    path = os.path.join(DATA, "music_trends.json")
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    key = (name or "").strip().lower()
    for f_ in d.get("folders", []):
        if key in f_["name"].lower() or f_["name"].lower() in key:
            return f_
    return None


def weight_fn_from_bands(bands):
    """把测量得到的频段占比变成权重曲线。

    复用 music_profile 的实现（alpha 压缩 + floor 抬底），不另造一套口径——
    音乐 50–70% 能量压在低频，不平滑会把权重全喂给低频。
    """
    import music_profile as mp
    flat = {k: (v["mean"] if isinstance(v, dict) else v) for k, v in bands.items()}
    return mp.weight_fn(flat)


def plot(trends, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.font_manager as fm
    import matplotlib.pyplot as plt
    for cand in (r"C:\Windows\Fonts\msyh.ttc", r"C:\Windows\Fonts\simhei.ttf"):
        if os.path.exists(cand):
            try:
                fm.fontManager.addfont(cand)
                plt.rcParams["font.sans-serif"] = [fm.FontProperties(fname=cand).get_name()]
                break
            except Exception:                                       # noqa: BLE001
                pass
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 6.4),
                                       gridspec_kw={"width_ratios": [1.5, 1.35, 0.95]})
    fig.patch.set_facecolor("#14161c")
    grid = np.array(trends[0]["curve_grid_hz"])
    colors = ["#5ee0c0", "#7aa2ff", "#ffb454", "#ff6f91", "#b48cff", "#8ad46a", "#e8e8e8"]

    # ---- (a) 平均频谱走向。y 轴收紧到数据实际跨度，否则 4/5 是空白 ----
    for i, t in enumerate(trends):
        ax.semilogx(grid, t["curve_mean_db"], lw=2.2, color=colors[i % len(colors)],
                    label="%s（%+.1f dB/oct）" % (t["name"], t["tilt_db_per_oct"]))
    ax.set_xlabel("频率 (Hz)", color="#c8cede", fontsize=10.5)
    ax.set_ylabel("相对电平 (dB，按 200–2k 归一化)", color="#c8cede", fontsize=10.5)
    ax.set_title("① 平均频谱走向（6 个目录几乎平行 → 是同一种平衡）",
                 color="#e8ebf2", fontsize=12.5, pad=10)
    ax.set_xlim(30, 20000)
    ax.set_ylim(-16, 7)
    ax.legend(fontsize=9, facecolor="#1a1d25", labelcolor="#c8cede",
              edgecolor="#2c313c", loc="lower left")

    # ---- (b) 分频段柱状：比曲线叠加更能看出"哪一段差多少" ----
    names = [b for b, _, _ in BANDS]
    x = np.arange(len(names))
    bw = 0.13
    for i, t in enumerate(trends):
        vals = [t["bands"][b]["mean"] for b in names]
        ax2.bar(x + (i - len(trends) / 2 + .5) * bw, vals, bw * .92,
                color=colors[i % len(colors)], label=t["name"])
    ax2.set_xticks(x)
    ax2.set_xticklabels(["%s\n%s" % (BAND_CN[b], ("%g–%g" % (lo, hi / 1000 if hi >= 1000 else hi)))
                         for b, lo, hi in BANDS], fontsize=8.2)
    ax2.set_ylabel("占 20–20kHz 总功率 %", color="#c8cede", fontsize=10.5)
    ax2.set_title("② 频段能量分布（20–60Hz 与 16–20kHz 是两个极端）",
                  color="#e8ebf2", fontsize=12.5, pad=10)
    ax2.annotate("ray of hope\n超低段 43.7%", xy=(0.02, 43.7), xycoords=("axes fraction", "data"),
                 xytext=(0.10, 50), textcoords=("axes fraction", "data"),
                 color="#ff8fa8", fontsize=9, ha="center",
                 arrowprops=dict(arrowstyle="->", color="#ff8fa8", lw=1.1))

    # ---- (c) 低频多 ↔ 中高频少 ----
    ax3.axvspan(50, 100, color="#ff6f91", alpha=.07)
    ax3.axhspan(0, 8, color="#ff6f91", alpha=.07)
    for i, t in enumerate(trends):
        ax3.plot([t["low_share"]], [t["presence_share"]], "o", ms=13,
                 color=colors[i % len(colors)], zorder=5)
        ax3.annotate(t["name"], (t["low_share"], t["presence_share"]),
                     textcoords="offset points", xytext=(0, 12), fontsize=8.8,
                     ha="center", color=colors[i % len(colors)], zorder=6)
    ax3.set_xlabel("低频能量占比 %（20–250Hz）", color="#c8cede", fontsize=10.5)
    ax3.set_ylabel("中高频能量占比 %（2–6kHz）", color="#c8cede", fontsize=10.5)
    ax3.set_title("③ 越靠右下＝越需要耳机「别补低频」", color="#e8ebf2", fontsize=12.5, pad=10)
    ax3.set_xlim(35, 82)
    ax3.set_ylim(-2, 20)
    ax3.text(76, 1.5, "低频主导角\n（再抬低频→糊中频）", color="#ff8fa8",
             fontsize=9, ha="center", va="bottom")

    for a in (ax, ax2, ax3):
        a.set_facecolor("#1a1d25")
        a.grid(True, which="both", color="#2c313c", lw=.6, alpha=.8)
        for s in a.spines.values():
            s.set_color("#2c313c")
        a.tick_params(colors="#8b93a5")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=fig.get_facecolor())
    print("图已存:", path, flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-folder", type=int, default=12,
                    help="每目录抽样曲目数。审计（src/audit.py B3）显示 n=4 时 ReoNa 的 "
                         "95%% 置信区间达 ±25.9 个百分点，跨目录比较会被误差棒淹没，故默认提到 12")
    ap.add_argument("--out", default=os.path.join(DATA, "music_trends.json"))
    ap.add_argument("--chart", default=os.path.join(OUT, "trends.png"))
    ap.add_argument("--roots-file", default=None,
                    help="UTF-8 文本，每行 '名称\\t路径'（覆盖内置目录表）")
    ap.add_argument("--replot", action="store_true",
                    help="只用已存的 data/music_trends.json 重绘图表，不解码音频")
    a = ap.parse_args()

    if a.replot:
        with open(a.out, encoding="utf-8") as f:
            trends = json.load(f)["folders"]
        if not trends:
            print("data/music_trends.json 里没有目录数据")
            return 1
        plot(trends, a.chart)
        return 0

    roots = DEFAULT_ROOTS
    if a.roots_file:
        roots = []
        with open(a.roots_file, encoding="utf-8") as fh:
            for ln in fh:
                ln = ln.rstrip("\n")
                if ln and not ln.startswith("#") and "\t" in ln:
                    nm, pth = ln.split("\t", 1)
                    roots.append((nm.strip(), pth.strip()))

    trends, all_rows = [], []
    for nm, root in roots:
        print("\n=== %s ===" % nm, flush=True)
        if not os.path.isdir(root):
            print("    !! 目录不存在:", root, flush=True)
            continue
        t = analyze_folder(nm, root, a.per_folder)
        if t:
            trends.append(t)
            all_rows.append(t)

    if not trends:
        print("没有可用结果")
        return 1

    payload = {
        "_metric": "% 线性功率（占据 20–20000Hz 总功率），口径同 trackscore.ai",
        "_method": "soundfile 全采样率解码 + Welch PSD(16384, 中段 60s)；跨子目录抽样",
        "_caveat": "音乐能量分布只描述「素材里已经有什么」，不等于「耳机好不好」，"
                   "也不能直接当耳机评分权重——它决定的是耳机「需不需要补这一段」。",
        "folders": [{k: v for k, v in t.items() if not k.startswith("_")} for t in trends],
    }
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("\n已写入", a.out, flush=True)

    plot(trends, a.chart)

    print("\n%-24s %6s %6s %6s %6s %6s %6s %6s | %7s %7s" % (
        "目录", *[BAND_CN[b] for b, _, _ in BANDS], "低频%", "倾斜"))
    for t in trends:
        print("%-24s %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f | %7.1f %7.1f" % (
            t["name"][:24], *[t["bands"][b]["mean"] for b, _, _ in BANDS],
            t["low_share"], t["tilt_db_per_oct"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
