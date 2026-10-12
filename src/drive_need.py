#!/usr/bin/env python3
"""L2 推导：由 L1 规格（阻抗/灵敏度）算「推到 110dB 峰值需要多少功率/电压/电流」。

为什么算 110dB：现代压缩流行的音乐平均电平高，峰值到 110dB SPL 是常见的
「开大一点」场景。要留 10dB 余量就按 120dB 看（表里同时给）。

口径规则（必须守住，否则跨厂商比较是错的）：
  各厂标法不同 —— 森海用 dB/V，DCA/Focal 用 dB/mW。换算：
      dB/V = dB/mW − 10·log10(Z) + 30
  比较前必须统一，否则会得出「森海比 DCA 好推」这类**看似合理但错误**的结论。
"""
import math

# (型号, 阻抗Ω, 灵敏度数值, 基准('dbv'|'dbmw'|None), 重量g, 价格, 来源级, 备注)
CAND = [
    ("Sennheiser HD 280 Pro", 64, 113.0, None, 285, None, "S0",
     "官网只写 '113 dB'，**未标基准**；两种解释都算一遍"),
    ("Focal Hadenys", 26, 100.0, "dbmw", 294, None, "S1",
     "官网只给 26Ω/294g，**灵敏度官网未标**（100dB/mW 出自 pconline，S1/S3）"),
    ("Austrian Audio Hi-X60", 25, 110.0, "dbv", 320, 499, "S0",
     "官网 '110 dBspl/V'"),
    ("Dan Clark Audio E3", 27, 90.0, "dbmw", 455, 2299.99, "S0",
     "官网 '~90 dB/mW'"),
    ("Sennheiser HD 560S", 120, 110.0, "dbv", 280, 199.95, "S0",
     "官网 110 dB/V（已核）"),
    ("FiiO FT1 Pro", 20, 95.0, "dbmw", 374, 199, "S0",
     "官网 20Ω/95dB/mW = 112dB/V"),
    ("Dan Clark Audio EXPANSE", 23, 86.5, "dbmw", None, 1999, "S3",
     "**无官方规格**；23Ω/86-87dB/mW 出自 Headfonics 评论（S3）"),
    ("Philips Fidelio X2HR", 30, 100.0, "dbmw", 380, 150, "S3", "未本轮核实"),
    ("Sony MDR-7506", 63, 106.0, "dbmw", 230, 100, "S3", "未本轮核实"),
    # ---- 本轮「低频有要求」的主流候选 ----
    ("AKG K371", 32, 114.0, "dbv", 255, 149, "S0",
     "AKG 手册：32Ω / 114 dB SPL/V @1kHz（旧档 provisional，本轮按手册升 S0）"),
    ("Focal Azurys", 26, 100.0, "dbmw", 306, 549, "S3",
     "**官网未标灵敏度**；26Ω/100dB/mW 出自 SoundStage 评论（S3）。重量/封闭式有官网"),
    ("Meze 105 AER", 42, 112.0, "dbmw", 336, 399, "S0", "官网 42Ω/112 dB/mW（已核）"),
    ("Sennheiser HD 600", 300, 97.0, "dbv", 260, 299, "S0", "官网 300Ω/97 dB/V"),
    ("Sennheiser HD 650", 300, 103.0, "dbv", 260, 399, "S0", "官网 300Ω/103 dB/V"),
    ("HIFIMAN Sundara", 37, 94.0, "dbmw", 372, 299, "S3", "catalog provisional"),
    ("HIFIMAN Edition XS", 18, 92.0, "dbmw", 350, 499, "S3", "catalog provisional"),
]


def to_dbv(z, val, unit):
    if unit == "dbv":
        return val
    if unit == "dbmw":
        return val - 10 * math.log10(z) + 30
    return None


def need(z, dbv, target_spl):
    """推到 target_spl 所需 功率(mW) / 电压(Vrms) / 电流(mA)。"""
    dbmw = dbv + 10 * math.log10(z) - 30
    p_mw = 10 ** ((target_spl - dbmw) / 10)
    v = math.sqrt(p_mw / 1000.0 * z)
    i_ma = v / z * 1000
    return dbmw, p_mw, v, i_ma


print("%-26s %6s %9s %8s | %s" % ("型号", "阻抗", "等效dB/mW", "重量", "推到 110 / 120 dB SPL"))
print("%-26s %6s %9s %8s | %-34s | %s" % ("", "Ω", "", "g", "110dB: 功率/电压/电流", "120dB: 功率/电压"))
print("-" * 132)
for name, z, val, unit, wt, price, tier, note in CAND:
    dbv = to_dbv(z, val, unit)
    if dbv is None:
        print("%-26s %6d  —（基准未标，两种解释）" % (name, z))
        for guess, lab in ((val, "若为 dB/V"), (val - 10 * math.log10(z) + 30, "若为 dB/mW→dB/V")):
            d2, p, v, i = need(z, guess, 110)
            _, p2, v2, _ = need(z, guess, 120)
            print("      %-22s dB/V=%.1f → 110dB 需 %6.2f mW / %.2f V / %5.1f mA ｜ 120dB 需 %6.1f mW / %.2f V"
                  % (lab, guess, p, v, i, p2, v2))
        continue
    dbmw, p, v, i = need(z, dbv, 110)
    _, p2, v2, _ = need(z, dbv, 120)
    print("%-26s %6d %9.1f %8s | %6.2f mW / %.2f V / %5.1f mA | %6.1f mW / %.2f V   [%s]"
          % (name, z, dbmw, wt or "?", p, v, i, p2, v2, tier))

print("""
判定门槛（工程经验，非标准）：
  功率 < 10 mW   → 手机/普通 dongle 够
  功率 10–50 mW  → 需像样的 dongle 或便携耳放
  功率 > 50 mW   → 需桌面耳放；且注意**电流**是否够（低阻平板常是电流瓶颈）
  电压 > 2 Vrms  → 多数手机直推会先削顶
""")
