#!/usr/bin/env python3
"""音乐能量层：把「听什么」变成频段权重，替代我手拍的 w(f)。

两条腿：
  A. 已发布的量化档案（有来源）—— 目前只找到**电子乐**，见 data/music_profiles.json
  B. **自己实测**：把音频放进 data/audio/，本脚本用 STFT 算各频段线性功率占比
     （numpy/scipy 已具备；wav 直读，mp3/m4a 走 ffmpeg）

⚠ 已知边界：trackscore.ai 明确声明 "Hip-hop, pop, and acoustic genres are outside the
scoring scope" —— **J-pop / 人声 / 动漫 OST 没有任何公开量化档案**。所以这两类必须靠 B。
"""
import json
import math
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
AUDIO = os.path.join(DATA, "audio")
PROFILES = os.path.join(DATA, "music_profiles.json")

# 与 trackscore 口径对齐的 7 段（Hz）
BANDS = [("sub", 20, 60), ("bass", 60, 250), ("low_mid", 250, 500),
         ("mid", 500, 2000), ("upper_mid", 2000, 6000), ("high", 6000, 16000),
         ("air", 16000, 20000)]

# 1/12 倍频程网格（与 fr_features 一致）
N = 121
FLO, FHI = 20.0, 20000.0
GRID = [FLO * (FHI / FLO) ** (i / (N - 1)) for i in range(N)]

SOURCED = {
    "electronic_general": {
        "profile": {"sub": 27, "bass": 54, "low_mid": 6, "mid": 5, "upper_mid": 4, "high": 4, "air": 0},
        "src": "trackscore.ai/blog/frequency-balance-electronic-music（2026-03 发布 / 2026-09 更新）",
        "metric": "% linear-power（占总频谱能量的线性功率百分比）",
    },
    "deep_house": {"profile": {"sub": 32, "bass": 52}, "src": "trackscore.ai（同上）", "metric": "同上"},
    "trance": {"profile": {"sub": 35, "bass": 49}, "src": "trackscore.ai（同上）", "metric": "同上"},
    "drum_and_bass": {"profile": {"sub": 31, "bass": 50}, "src": "trackscore.ai（同上）", "metric": "同上"},
    "bass_music_dubstep": {"profile": {"sub": 45, "bass": 31}, "src": "trackscore.ai（同上）", "metric": "同上"},
}


def load_profiles():
    if os.path.exists(PROFILES):
        with open(PROFILES, encoding="utf-8") as f:
            return json.load(f)
    return {"sourced": SOURCED, "measured": {}}


def _decode(path, tmp_wav):
    """非 wav 用 ffmpeg 转 16bit 单声道 22.05k。"""
    try:
        r = subprocess.run(["ffmpeg", "-y", "-v", "error", "-i", path,
                            "-ac", "1", "-ar", "22050", tmp_wav], capture_output=True)
        return r.returncode == 0 and os.path.exists(tmp_wav)
    except FileNotFoundError:
        return False


def analyze(path):
    """STFT -> 各频段线性功率占比（%）。"""
    import numpy as np
    from scipy.io import wavfile
    src = path
    if not path.lower().endswith(".wav"):
        tmp = os.path.join(os.environ.get("TEMP", "."), "_mp_decode.wav")
        if not _decode(path, tmp):
            return {"error": "需要 ffmpeg 解码非 wav（或改放 wav 文件）"}
        src = tmp
    sr, x = wavfile.read(src)
    if x.ndim > 1:
        x = x.mean(axis=1)
    x = np.asarray(x, dtype=np.float64)
    if x.size < sr:                                  # <1 秒，跳过
        return {"error": "音频过短"}
    win = 8192
    hop = win // 2
    n_frames = max(1, (len(x) - win) // hop)
    acc = np.zeros(win // 2 + 1)
    w = np.hanning(win)
    for i in range(n_frames):
        seg = x[i * hop: i * hop + win] * w
        spec = np.abs(np.fft.rfft(seg)) ** 2
        acc += spec
    freqs = np.fft.rfftfreq(win, 1.0 / sr)
    total = acc.sum() or 1.0
    out = {}
    for name, lo, hi in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        out[name] = round(float(acc[m].sum() / total * 100), 2)
    return out


def weights_from_profile(prof, alpha=0.5, floor=0.12):
    """把频段能量占比变成 n=121 点的权重曲线。

    w(f) = (能占比^alpha) 归一化，再抬一个下限 floor。
    alpha<1 用于压缩动态范围：电子乐 81% 能量在 sub+bass，若不平滑会淹没中高频。
    """
    band_pct = {}
    for name, lo, hi in BANDS:
        band_pct[(lo, hi)] = max(float(prof.get(name, 0.0)), 0.0)
    mx = max(band_pct.values()) or 1.0
    raw = {k: (v / mx) ** alpha for k, v in band_pct.items()}
    lo_v = min(raw.values())
    hi_v = max(raw.values()) or 1.0
    out = []
    for f in GRID:
        v = None
        for (lo, hi), val in raw.items():
            if lo <= f < hi:
                v = val
                break
        if v is None:
            v = lo_v
        norm = (v - lo_v) / (hi_v - lo_v + 1e-9)     # 0..1
        out.append(floor + (1.0 - floor) * norm)      # 抬下限
    return out


def weight_fn(prof):
    w = weights_from_profile(prof)
    idx = {round(g, 3): w[i] for i, g in enumerate(GRID)}

    def f(hz):
        k = min(idx, key=lambda g: abs(g - hz))
        return idx[k]
    return f


def main():
    P = load_profiles()
    if len(sys.argv) > 1 and sys.argv[1] == "scan":
        os.makedirs(AUDIO, exist_ok=True)
        files = [x for x in os.listdir(AUDIO) if x.lower().endswith((".wav", ".mp3", ".m4a", ".flac", ".aac", ".ogg"))]
        if not files:
            print("data/audio/ 里没有音频。\n把 2–3 首代表性曲目（或 30 秒片段）放进去，再跑：python src/music_profile.py scan")
            return
        measured = {}
        for fn in files:
            r = analyze(os.path.join(AUDIO, fn))
            print("  %-40s %s" % (fn[:40], r))
            if "error" not in r:
                measured[fn] = r
        if measured:
            P.setdefault("measured", {}).update(measured)
            with open(PROFILES, "w", encoding="utf-8") as f:
                json.dump(P, f, ensure_ascii=False, indent=2)
            print("\n已写入 %s" % PROFILES)
        return
    print("已发布的量化档案：")
    for k, v in P.get("sourced", {}).items():
        print("  %-22s %s" % (k, v["profile"]))
        print("      来源: %s" % v["src"])
    print("\n自测档案：", list(P.get("measured", {}).keys()) or "（空 —— 请把音频放入 data/audio/ 后跑 scan）")


if __name__ == "__main__":
    main()
