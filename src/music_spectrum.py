#!/usr/bin/env python3
"""本地曲库频谱分析：验证「Hi-Res 真伪」与「有损溯源」。

判据（不靠标签，靠内容）：
  · 容器采样率 fs、位深、时长          —— 元数据层（可能骗人）
  · 长时平均谱的**真实高频截止点** cutoff —— 内容层（骗不了人）
  · cutoff / (fs/2) 比值              —— 是否被升频填充
  · cutoff 绝对位置                    —— 是否有损转码（16.5k≈192k、20k≈320k/AAC、22k≈CD）

用法: python src/music_spectrum.py --root "<目录>" [--root ...] [--out <目录>] [--deep-per-album 3]
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import soundfile as sf
from scipy import signal

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

for _f in ("Microsoft YaHei", "Meiryo", "Yu Gothic", "SimHei", "MS Gothic", "Noto Sans CJK JP"):
    try:
        if any(_f.lower() in x.name.lower() for x in matplotlib.font_manager.fontManager.ttflist):
            plt.rcParams["font.sans-serif"] = [_f]
            break
    except Exception:
        pass
plt.rcParams["axes.unicode_minus"] = False

EXTS = (".flac", ".wav", ".mp3", ".m4a", ".aif", ".aiff", ".ogg")
THRESH_DB = 75.0          # 相对峰值谱，低于 75dB 视为"无内容"
BRICK_DB_PER_OCT = 40.0   # 判定"砖墙"的陡度


def lta(path, seconds=45.0, nperseg=8192):
    """长时平均谱：取中段 seconds 秒，Welch 平均。返回 (f, psd_db, fs)。"""
    info = sf.info(path)
    fs = info.samplerate
    total = info.frames
    start = int(total * 0.25)
    n = int(min(total - start - 1, seconds * fs))
    if n < nperseg * 4:
        start, n = 0, min(total - 1, max(nperseg * 4, int(seconds * fs)))
    x, _ = sf.read(path, start=start, frames=n, dtype="float32", always_2d=True)
    m = x.mean(axis=1)
    m = m - m.mean()
    if not np.any(np.abs(m) > 1e-6):
        return None, None, fs
    f, p = signal.welch(m, fs=fs, nperseg=nperseg, noverlap=nperseg // 2,
                        window="hann", scaling="density")
    db = 10 * np.log10(p + 1e-20)
    return f, db, fs


def smooth_log(f, db):
    """在 log-f 轴上做 ~1/6 倍频程平滑。"""
    out = db.copy()
    ratio = 2 ** (1 / 6)
    for i, fc in enumerate(f):
        if fc <= 0:
            continue
        m = (f >= fc / ratio) & (f <= fc * ratio)
        if m.sum() >= 2:
            out[i] = db[m].mean()
    return out


def cutoff_of(f, db):
    """真实高频截止点：从高往低找连续 3 个 1/12oct 带均高于阈值的最高点。"""
    if f is None:
        return None, None
    s = smooth_log(f, db)
    thr = s.max() - THRESH_DB
    step = 2 ** (1 / 12)
    lo = max(float(f[0]), 20.0)   # 注意 f[0] 是 0Hz 直流分量，不能拿它当循环下界
    cur, run, cut = f[-1], 0, None
    while cur > lo:
        j = int(np.argmin(np.abs(f - cur)))
        if s[j] > thr:
            run += 1
        else:
            run = 0
        if run >= 3:
            cut = cur      # 自高向低扫，第一个连续满足点即真实截止
            break
        cur /= step
    if cut is None:
        return None, thr
    # 砖墙陡度：cut 之上 1 octave 内掉了多少 dB
    j0 = int(np.argmin(np.abs(f - cut)))
    j1 = int(np.argmin(np.abs(f - min(cut * 2, f[-1]))))
    slope = (s[j0] - s[j1]) if j1 > j0 else 0.0
    return cut, slope


def classify(fs, cutoff, slope):
    nyq = fs / 2
    if cutoff is None:
        return "异常（无可测内容）", "—"
    ratio = cutoff / nyq
    k = cutoff / 1000
    if fs <= 45000:                       # 44.1kHz 容器
        if k >= 21.5:
            how = "满带宽（CD 级完整）"
        elif k >= 19.0:
            how = "轻微限制（≈320kbps 有损，或母带本身带限）"
        elif k >= 16.0:
            how = "**有损溯源**（≈192–256kbps 转码）"
        else:
            how = "**强有损溯源**（≤128kbps）"
    elif fs <= 50000:                     # 48kHz 容器
        if k >= 23.0:
            how = "真 48kHz 带宽"
        elif k >= 21.0:
            how = "**疑似 44.1kHz 升频**：容器 48kHz，内容止于 %.1fkHz（≈22.05k 为 44.1k 上限）" % k
        elif k >= 19.0:
            how = "带宽偏窄（%.1fkHz）——48k 容器里更像有损/带限母带" % k
        elif k >= 16.0:
            how = "**有损溯源**（≈192–256kbps 转码）"
        else:
            how = "**强有损溯源**（≤128kbps）"
    else:                                 # >48k，宣称 Hi-Res
        if k <= 24.5:
            how = "**伪 Hi-Res**：容器 %.1fkHz，内容止于 %.1fkHz → 实为 44.1/48kHz 升频" % (fs / 1000, k)
        elif ratio >= 0.85:
            how = "真 Hi-Res：内容延伸到 %.1fkHz（占奈奎斯特 %.0f%%）" % (k, ratio * 100)
        else:
            how = "半 Hi-Res：内容 %.1fkHz，容器 %.1fkHz（上端为空）" % (k, fs / 1000)
    if slope and slope < -BRICK_DB_PER_OCT and k < 22:
        how += "｜边界陡降 %.0fdB/oct（典型砖墙滤波）" % slope
    return how, ratio


def plot_album(name, tracks, outdir, spec_track=None):
    os.makedirs(outdir, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_（）()" else "_" for c in name)[:60]
    if tracks:
        plt.figure(figsize=(10, 5))
        for t in tracks:
            if t.get("f") is None:
                continue
            plt.semilogx(t["f"], t["db"], lw=0.8, label="%s（%.1fk 截止）" % (t["label"][:34], t["cutoff_k"]))
        plt.xlim(20, 48000)
        plt.ylim(-110, 5)
        plt.grid(which="both", alpha=0.25)
        plt.xlabel("频率 Hz")
        plt.ylabel("相对电平 dB")
        plt.title("长时平均谱 · " + name[:48])
        plt.legend(fontsize=7, loc="lower left")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "lta_" + safe + ".png"), dpi=125)
        plt.close()
    if spec_track:
        try:
            info = sf.info(spec_track)
            x, _ = sf.read(spec_track, start=int(info.frames * 0.3),
                           frames=int(info.samplerate * 12), dtype="float32", always_2d=True)
            m = x.mean(axis=1)
            fr, tt, sx = signal.spectrogram(m, fs=info.samplerate, nperseg=4096, noverlap=3072)
            plt.figure(figsize=(10, 5))
            plt.pcolormesh(tt, fr, 10 * np.log10(sx + 1e-20), shading="auto", cmap="magma",
                           vmin=-120, vmax=-20)
            plt.yscale("log")
            plt.ylim(20, info.samplerate / 2)
            plt.xlabel("时间 s")
            plt.ylabel("频率 Hz")
            plt.title("频谱图 · " + os.path.basename(spec_track)[:52])
            plt.colorbar(label="dB")
            plt.tight_layout()
            plt.savefig(os.path.join(outdir, "spec_" + safe + ".png"), dpi=125)
            plt.close()
        except Exception as e:
            print("   (频谱图失败 %s: %s)" % (os.path.basename(spec_track), e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--deep-per-album", type=int, default=3)
    a = ap.parse_args()
    out = a.out or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "out", "music")
    os.makedirs(out, exist_ok=True)

    files = []
    for root in a.root:
        for dp, _, fns in os.walk(root):
            for fn in fns:
                if fn.lower().endswith(EXTS):
                    files.append(os.path.join(dp, fn))
    files.sort()
    print("扫描到音频文件 %d 个（含非 FLAC）" % len(files))

    albums = defaultdict(list)
    for p in files:
        albums[os.path.dirname(p)].append(p)

    report, deep_rows = [], []
    for alb in sorted(albums):
        lst = albums[alb]
        name = os.path.basename(alb)
        infos = []
        for p in lst:
            try:
                i = sf.info(p)
                infos.append((p, i.samplerate, i.subtype, i.channels, i.frames / i.samplerate))
            except Exception as e:
                print("  !! 读不了 %s: %s" % (os.path.basename(p), e))
        if not infos:
            continue
        rates = sorted({x[1] for x in infos})
        subs = sorted({x[2] for x in infos})
        durs = sum(x[4] for x in infos)
        print("\n[%s]  %d 曲 ｜ %s Hz ｜ %s ｜ 共 %.1f 分钟" % (
            name[:56], len(infos), "/".join(str(r) for r in rates), ",".join(subs), durs / 60))

        picks = lst[:max(1, a.deep_per_album)] if a.deep_per_album else []
        tr, labels = [], []
        for p in picks:
            f, db, fs = lta(p)
            cut, slope = cutoff_of(f, db)
            how, ratio = classify(fs, cut, slope)
            kk = (cut / 1000) if cut else float("nan")
            print("    %-42s %5.1fk  %s" % (os.path.basename(p)[:42], kk, how))
            tr.append({"f": f, "db": db, "cutoff_k": kk, "label": os.path.basename(p)})
            labels.append((p, fs, cut, how))
            deep_rows.append({"album": name, "file": os.path.basename(p), "path": p,
                              "fs": fs, "cutoff_hz": cut, "ratio": ratio, "verdict": how})
        plot_album(name, tr, os.path.join(out, "plots"), spec_track=picks[0] if picks else None)
        report.append({"album": name, "dir": alb, "n": len(infos), "rates": rates, "subtypes": subs,
                       "minutes": round(durs / 60, 1), "tracks": deep_rows[-len(picks):] if picks else []})

    with open(os.path.join(out, "music_spectrum.json"), "w", encoding="utf-8") as f:
        json.dump({"albums": report, "tracks": deep_rows}, f, ensure_ascii=False, indent=1)
    print("\n结果: %s" % os.path.join(out, "music_spectrum.json"))
    print("图表: %s" % os.path.join(out, "plots"))


if __name__ == "__main__":
    main()
