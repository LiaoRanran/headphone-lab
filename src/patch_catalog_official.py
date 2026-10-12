#!/usr/bin/env python3
"""把官方页（S0）抄录到的规格写回市场档案，并把 spec_status 升为 verified。

原则：只写**官网原文**；官网没给的（如 HD 620S 本体重量）保持 pending。
"""
import json
import os

CAT = os.path.join(os.path.expanduser("~"), ".codebuddy", "skills", "headphone-advisor",
                   "references", "market_catalog.json")
SENN = "https://us.sennheiser-hearing.com/products/"

UP = {
    "sennheiser_hd600": dict(
        spec={"impedance_ohm": 300, "sens_db_V": 97, "weight_g": 260, "fr": "12Hz-40.5kHz",
              "thd": "小于0.1% (1kHz,1V)"},
        url=SENN + "hd-600", usd_now=349.95, usd_msrp=499.95,
        note="S0 官网抄录；灵敏度基准=1V（dB/V）"),
    "sennheiser_hd650": dict(
        spec={"impedance_ohm": 300, "sens_db_V": 103, "weight_g": 260, "fr": "10Hz-41kHz",
              "thd": "小于0.05% (1kHz,1V)"},
        url=SENN + "hd-650", usd_now=379.95, usd_msrp=579.95,
        note="**勘误**：旧档灵敏度写 97 dB/V，官网实为 **103 dB/V**；官网同页特性栏写 12Hz、规格表写 10Hz（并列）"),
    "sennheiser_hd660s2": dict(
        spec={"impedance_ohm": 300, "sens_db_V": 104, "weight_g": 260, "fr": "8Hz-41.5kHz",
              "thd": "小于0.04% (1kHz,100dB)"},
        url=SENN + "hd-660s2", usd_now=479.95, usd_msrp=679.95,
        note="S0 官网；104 dB/V ≈ 98.8 dB/mW（推算，官网未给 dB/mW）"),
    "sennheiser_hd599": dict(
        spec={"impedance_ohm": 50, "sens_db_V": 106, "weight_g": 250, "fr": "12Hz-38.5kHz",
              "thd": "小于0.1% (1kHz,100dB)"},
        url=SENN + "hd-599", usd_now=239.95, usd_msrp=239.95, note="S0 官网抄录；38mm 单元"),
    "sennheiser_hd560s": dict(
        spec={"impedance_ohm": 120, "sens_db_V": 110, "weight_g": 280, "fr": "6Hz-38kHz",
              "thd": "小于0.05% (1kHz,90dB)"},
        url=SENN + "hd-560s", usd_now=199.95, usd_msrp=279.95,
        note="**勘误**：旧档重量写 240g，官网实为 **280g**；阻抗项规格表未列，120Ω 出自官网正文（声学团队 Kurt Mick 引述）"),
    "sennheiser_hd620s": dict(
        spec={"impedance_ohm": 150, "sens_db_V": 110, "weight_g": None, "fr": "6Hz-30kHz",
              "thd": "小于0.05% (1kHz,90dB)"},
        url=SENN + "hd-620-s", usd_now=399.95, usd_msrp=399.95,
        note="**补空**：旧档阻抗为 null，官网实为 **150Ω**；本体重量官网**未给**（只给包装 670g），故仍留 pending"),
    "sennheiser_hd800s": dict(
        spec={"impedance_ohm": 300, "sens_db_V": 102, "weight_g": 330, "fr": "4Hz-51kHz",
              "thd": "小于0.02% (1kHz,1Vrms)"},
        url=SENN + "hd-800-s", usd_now=1799.95, usd_msrp=1999.95,
        note="**来源升级**：旧档引自百科 S4，现为官网 S0；56mm 环形振膜"),
    "fiio_ft1pro": dict(
        spec={"impedance_ohm": 20, "sens_db_V": 112, "sens_db_mW": 95, "weight_g": 374,
              "fr": "7Hz-40kHz"},
        url="https://www.fiio.com/ft1pro", usd_now=None, usd_msrp=None,
        note="**冲突裁决**：官网参数页明写 **20Ω**（旧档 20Ω vs 32Ω 两说→定为 20Ω）；官网同时给 112 dB/Vrms = 95 dB/mW；重量官网约 374g（旧档 350g）。官网未标价"),
    "meze_105aer": dict(
        spec={"impedance_ohm": 42, "sens_db_mW": 112, "weight_g": 336, "fr": "5Hz-30kHz",
              "thd": "小于0.1% (1kHz)"},
        url="https://mezeaudio.com/products/105-aer", usd_now=299.0, usd_msrp=399.0,
        note="**补空**：旧档 spec 为空；50mm 动圈、生物纤维素球顶、开放"),
    "beyerdynamic_dt770": dict(
        spec={"impedance_ohm": 250, "weight_g": None, "fr": "5Hz-35kHz",
              "sens_note": "80Ω 版本官网标 96 dB **未注明基准**"},
        url="https://north-america.beyerdynamic.com/dt-770-pro.html", usd_now=159.99, usd_msrp=199.99,
        note="**口径警告**：官网 96 dB **未标基准**（既非明确 dB/V 也非 dB/mW），不可直接与他牌比；官网重量 375g 为 **80Ω 版含线缆**，250Ω 版另有 spec sheet（PDF）待取，故本体重量留 pending"),
}


def main():
    d = json.load(open(CAT, encoding="utf-8"))
    changed = []
    for m in d["models"]:
        u = UP.get(m["id"])
        if not u:
            continue
        old = json.dumps(m.get("spec", {}), ensure_ascii=False)
        m["spec"] = u["spec"]
        m["spec_source"] = {"url": u["url"], "grade": "S0", "note": u["note"]}
        m["spec_status"] = "verified"
        m["verify"] = "spec已核(S0官网)"
        if u["usd_now"] is not None:
            m.setdefault("price_ref", {})
            m["price_ref"]["usd_listed"] = u["usd_now"]
            if u["usd_msrp"]:
                m["price_ref"]["usd_msrp"] = u["usd_msrp"]
            m["price_ref"]["as_of"] = "2026-10-11"
            m["price_ref"]["src"] = u["url"]
        changed.append((m["id"], old, json.dumps(m["spec"], ensure_ascii=False)))
    d["updated"] = "2026-10-11"
    d["official_spec_pass"] = ("2026-10-11 首轮官方页抄录：10 只升为 S0 verified；"
                              "勘误 2 处（HD650 灵敏度、HD560S 重量）、补空 2 处（HD620S 阻抗、105 AER 全套）、"
                              "裁决冲突 1 处（FT1 Pro = 20Ω）")
    json.dump(d, open(CAT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("已更新 %d 只 → %s\n" % (len(changed), CAT))
    for cid, o, n in changed:
        print("  %-24s\n     旧: %s\n     新: %s" % (cid, o, n))


if __name__ == "__main__":
    main()
