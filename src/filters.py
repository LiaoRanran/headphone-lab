#!/usr/bin/env python3
"""型号属性过滤：无线/降噪/游戏耳机 与 压耳式 的识别。

AutoEq 的型号名不带属性标签，必须靠规则+人工覆盖来分流。
规则保守（宁可标注 unknown 让人复核，也不误杀）。
"""
import re

# 无线 / 蓝牙 / 降噪 / 游戏无线 —— 一旦命中即判 wireless
WIRELESS = [
    r"wireless", r"bluetooth", r"\bbt\b", r"\banc\b", r"noise cancell?ing", r"noise-cancell?ing",
    r"quietcomfort", r"quietcontrol", r"quiet control",
    r"airpods", r"maxwell", r"bathys", r"accentum", r"momentum (2|3|4)\b", r"momentum wireless",
    r"wh-", r"wf-", r"wi-", r"wh1000", r"1000x", r"h900n", r"mdr-zx", r"mdr-xb", r"mdr-1000",
    r"hd 4\.", r"hd 350", r"hd 450", r"px7", r"px8", r"pxc", r"pi7", r"pi5",
    r"beats (studio|solo|fit)", r"studio pro", r"galaxy", r"buds", r"\btws\b",
    r"jbl (tune|live|club|reflect)", r"soundtrue", r"soundlink", r"bose (qc|700|ultra)",
    r"n700nc", r"stealth", r"astro", r"a50\b", r"lightspeed", r"g733", r"g935", r"g435",
    r"turtle beach", r"arctis", r"hyperx cloud", r"razer", r"corsair",
    r"tune \d", r"live \d", r"elite \d", r"solo \d",
    r"sonoflow", r"ultrafocus", r"monster inspiration", r"soundcore", r"anker", r"edifier",
    r"final audio (u[0-9]|ze[0-9])", r"hyperx", r"logitech", r"steelseries", r"bose ",
]
ON_EAR = [r"on-ear", r"on ear", r"supra-aural", r"porta pro", r"ksc75", r"koss"]
# 人工覆盖：真值 > 规则（name 小写包含即生效）
OVERRIDE = {
    "psb m4u 8": "wireless",             # M4U 8 带蓝牙（M4U 2 才是有线 ANC）
    "psb m4u 2": "wired",
    "beats pro": "wired",                # Beats Pro 是有线
    "bowers & wilkins p7": "wired",      # P7 有线（Px7/Px8 才无线）
    "sony mdr-z7": "wired",              # MDR-Z7/Z7M2 有线
    "fine audio": "wired",
}


def classify(name):
    n = name.lower()
    for k, v in OVERRIDE.items():
        if k in n:
            return v
    if any(re.search(p, n) for p in WIRELESS):
        return "wireless"
    if any(re.search(p, n) for p in ON_EAR):
        return "on-ear"
    return "wired"


def is_wired_over_ear(name):
    return classify(name) == "wired"
