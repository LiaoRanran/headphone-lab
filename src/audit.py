#!/usr/bin/env python3
"""审计：数据库与方法论的可疑之处（用户质疑「数据库和测试方法还是有问题」）。

不预设结论，逐项跑可验证的检查：

A. 数据库
  A1 来源构成 —— 不同来源用不同人工耳/补偿/目标曲线，能否直接混在一起取中位数？
  A2 squig 覆盖率 —— 采了多少条 vs 实际进库多少条（历史上「合并重跑」这步可能没做）
  A3 参考曲线一致性 —— `devt` = 曲线 − 来源自带目标曲线。若各来源目标曲线不同，
     `devt` 就不在同一参考系上，「相对 Harman」这个说法只对部分来源成立。
     检验：同型号多来源时，devt 的跨源离散是否显著大于 dev 的跨源离散。
  A4 归一化区间

B. 音乐测量方法
  B1 单声道化处理 —— 先平均 L/R 再做 FFT，会把左右不相关的成分抵消掉
     （电子乐/宽立体声尤其明显）。对照：先各自算功率谱再平均。
  B2 只取 30–90% 段 —— 副歌与 intro 的平衡可以差很多，固定取中段是否引入系统偏差
  B3 每目录只抽 4 首 —— 用已有逐曲数据算标准误，看「目录画像」有多稳
"""
import json
import os
import sys

import numpy as np
import soundfile as sf
from scipy import signal

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")

BANDS = [("sub_bass", 20, 60), ("bass", 60, 200), ("low_mid", 200, 500),
         ("mid", 500, 2000), ("presence", 2000, 5000), ("treble", 5000, 10000),
         ("air", 10000, 20000)]


def hr(t):
    print("\n" + "=" * 78)
    print(t)
    print("=" * 78)


# ------------------------------------------------------------------ A 数据库
def audit_db():
    import engine
    import fr_features as ff
    D = engine.load()
    ms = D["models"]
    G = engine.group(ms)

    hr("A1 来源构成")
    import collections
    c = collections.Counter(m["source"] for m in ms)
    for k, v in c.most_common():
        print("  %-16s %4d 条" % (k, v))
    print("\n  → %d 个来源。每个来源有**自己的人工耳、自己的补偿曲线、自己的目标曲线**。" % len(c))
    print("    把它们的中位数当成「一个客观值」用，前提是它们的参考系一致——见 A3。")

    hr("A2 squig 覆盖率")
    squig_srcs = [s for s in c if s not in ("Innerfidelity", "Rtings", "oratory1990")]
    in_db = sum(c[s] for s in squig_srcs)
    squig_dir = os.path.join(DATA, "squig_csv")
    n_csv = 0
    for dp, _, fns in os.walk(squig_dir):
        n_csv += sum(1 for f in fns if f.lower().endswith(".csv"))
    man = os.path.join(DATA, "squig_csv_manifest.jsonl")
    n_man = 0
    if os.path.exists(man):
        n_man = sum(1 for _ in open(man, encoding="utf-8"))
    print("  已转成标准 CSV 的文件数 : %d" % n_csv)
    print("  CSV 清单 manifest 行数 : %d" % n_man)
    print("  其中进入 features.json  : %d 条（来源 %s）" % (in_db, squig_srcs))
    gap = n_csv - in_db
    print("  → 缺口 %d 条。**「合并重跑 features.json」这一步历史上没有完成。**" % gap if gap > 0
          else "  → 无缺口")

    hr("A3 参考曲线一致性（关键）")
    print("  dev  = 曲线 − 数据集参考（%s）" % D.get("reference"))
    print("  devt = 曲线 − **来源自带目标曲线**")
    print("  若各来源目标曲线不同，devt 就不在同一参考系上。\n")
    rows = []
    for name, g in G.items():
        if len(g["sources"]) < 2:
            continue
        # 该型号各来源曲线的 devt 与 dev 的跨源离散
        ms_n = [m for m in ms if m["name"] == name]
        if len(ms_n) < 2:
            continue
        n = len(ms_n[0]["devt"])
        sp_devt = sum(abs(a["devt"][i] - b["devt"][i])
                      for a in ms_n for b in ms_n for i in range(n)) / (len(ms_n) ** 2 * n)
        sp_dev = sum(abs(a["dev"][i] - b["dev"][i])
                     for a in ms_n for b in ms_n for i in range(n)) / (len(ms_n) ** 2 * n)
        rows.append((name, len(ms_n), sp_dev, sp_devt))
    if not rows:
        print("  没有多来源型号，无法检验")
        return
    rows.sort(key=lambda r: -(r[3] - r[2]))
    d_dev = np.mean([r[2] for r in rows])
    d_devt = np.mean([r[3] for r in rows])
    print("  多来源型号 %d 个" % len(rows))
    print("  平均跨源离散：dev = %.3f dB ｜ devt = %.3f dB  → 差 %+.3f dB"
          % (d_dev, d_devt, d_devt - d_dev))
    print("\n  devt 离散最大的 12 个（若这些远大于 dev，说明参考系不一致）：")
    print("  %-38s %4s %8s %8s %8s" % ("型号", "源数", "dev", "devt", "差"))
    for name, k, sd, st in rows[:12]:
        print("  %-38s %4d %8.2f %8.2f %+8.2f" % (name[:38], k, sd, st, st - sd))

    hr("A4 归一化区间")
    print("  NORM_LO..NORM_HI = %g..%g Hz" % (ff.NORM_LO, ff.NORM_HI))
    print("  → 所有曲线先按此区间均值归零再比较。若某来源的测量在此区间有凹陷，")
    print("    会把整条曲线的相对形状带偏。")


# ------------------------------------------------------------------ B 测量
def _band_share(P, f):
    tot = float(P[(f >= 20) & (f <= 20000)].sum()) or 1.0
    return {k: float(P[(f >= lo) & (f < hi)].sum() / tot * 100) for k, lo, hi in BANDS}


def audit_method():
    hr("B1 单声道化：先平均 L/R vs 先算功率谱再平均")
    tracks = []
    for root, names in [
        (os.path.join(os.environ["USERPROFILE"], "Desktop", "Music", "ELECTROCUTICA"), "ELECTROCUTICA"),
        (os.path.join(os.environ["USERPROFILE"], "Desktop", "Music", "ray of hope"), "ray of hope"),
        (os.path.join(os.environ["USERPROFILE"], "Desktop", "Music",
                      "[Hi-Res][250319]Adoのベストアドバム[48kHz／24bit][FLAC]"), "Ado"),
    ]:
        if not os.path.isdir(root):
            continue
        # 每个子目录取一首、铺开（此前 break 写在 if 外面，只取到根目录一层，
        # 而 Ado/ELECTROCUTICA 的 flac 都在子目录里 → 实际只测到 1 首）
        for dp, _, fns in sorted(os.walk(root)):
            fl = sorted(f for f in fns if f.lower().endswith(".flac"))
            if fl:
                tracks.append((names, os.path.join(dp, fl[len(fl) // 2])))
            if len([t for t in tracks if t[0] == names]) >= 2:
                break

    print("  Δ = 「先平均成单声道」−「各自算功率谱再平均」，单位百分点")
    print("  单声道化的交叉项 2·Re{L·R*} 会让左右相关的成分叠加、不相关的抵消，")
    print("  电子乐/宽立体声受影响最大。\n")
    print("  %-13s %-30s %7s %7s %7s %7s %7s" % (
        "目录", "曲目", "超低", "低频", "中低", "中频", "中高"))
    worst = 0.0
    for label, p in tracks[:6]:
        info = sf.info(p)
        fs, total = info.samplerate, info.frames
        st_ = int(total * 0.3)
        n = int(min(60 * fs, total - st_ - 1))
        x, _ = sf.read(p, start=st_, frames=n, dtype="float32", always_2d=True)
        if x.shape[1] < 2:
            print("  %-13s %-30s  （非立体声，跳过）" % (label[:13], os.path.basename(p)[:30]))
            continue
        L, R = x[:, 0], x[:, 1]
        f1, P1 = signal.welch((L + R) / 2.0, fs=fs, nperseg=16384, window="hann",
                              noverlap=8192, scaling="spectrum")
        a = _band_share(P1, f1)
        f2, PL = signal.welch(L, fs=fs, nperseg=16384, window="hann",
                              noverlap=8192, scaling="spectrum")
        _, PR = signal.welch(R, fs=fs, nperseg=16384, window="hann",
                             noverlap=8192, scaling="spectrum")
        b = _band_share((PL + PR) / 2.0, f2)
        ds = [a[k] - b[k] for k, _, _ in BANDS]
        worst = max(worst, max(abs(d) for d in ds))
        print("  %-13s %-30s %+7.2f %+7.2f %+7.2f %+7.2f %+7.2f"
              % (label[:13], os.path.basename(p)[:30], *ds[:5]))
    print("\n  最大偏差 %.2f 个百分点。" % worst)

    hr("B2 只取 30–90% 段 vs 全曲")
    for label, p in tracks[:4]:
        info = sf.info(p)
        fs, total = info.samplerate, info.frames
        if total < fs * 90:
            continue
        seg_n = int(60 * fs)
        st_ = int(total * 0.3)
        x, _ = sf.read(p, start=st_, frames=seg_n, dtype="float32", always_2d=True)
        f1, P1 = signal.welch(x.mean(axis=1), fs=fs, nperseg=16384, window="hann",
                              noverlap=8192, scaling="spectrum")
        a = _band_share(P1, f1)
        # 全曲：分块累计（内存安全）
        acc = np.zeros_like(P1)
        f2 = f1
        hop = int(120 * fs)
        pos = 0
        while pos < total:
            xx, _ = sf.read(p, start=pos, frames=min(hop, total - pos),
                            dtype="float32", always_2d=True)
            if xx.shape[0] > 16384:
                _, P = signal.welch(xx.mean(axis=1), fs=fs, nperseg=16384,
                                    window="hann", noverlap=8192, scaling="spectrum")
                acc += P
            pos += hop
        b = _band_share(acc, f2)
        print("  %-14s %s" % (label[:14], os.path.basename(p)[:46]))
        print("      %-10s %8.2f %8.2f %8.2f %8.2f %8.2f" % ("段/全曲", *[a[k] for k, _, _ in BANDS[:5]]))
        print("      %-10s %8.2f %8.2f %8.2f %8.2f %8.2f"
              % ("", *[b[k] for k, _, _ in BANDS[:5]]))
        print("      %-10s %+8.2f %+8.2f %+8.2f %+8.2f %+8.2f   ← 中段 − 全曲"
              % ("Δ", *[a[k] - b[k] for k, _, _ in BANDS[:5]]))

    hr("B3 每目录只抽 4 首 → 标准误")
    with open(os.path.join(DATA, "music_trends.json"), encoding="utf-8") as f:
        trends = json.load(f)["folders"]
    print("  %-24s %6s %8s %10s %10s" % ("目录", "曲目", "均值%", "标准差", "标准误"))
    for t in trends:
        lows = [tr["low_share"] for tr in t["tracks"] if tr.get("low_share") is not None]
        if len(lows) < 2:
            continue
        mu, sd = float(np.mean(lows)), float(np.std(lows, ddof=1))
        se = sd / np.sqrt(len(lows))
        print("  %-24s %6d %8.1f %10.1f %10.1f  → 95%% 区间 ±%.1f"
              % (t["name"][:24], len(lows), mu, sd, se, 1.96 * se))


def main():
    print("审计中……（B 部分要解码音频，稍慢）", flush=True)
    audit_db()
    audit_method()


if __name__ == "__main__":
    sys.exit(main())
