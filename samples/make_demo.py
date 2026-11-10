"""데모용 추출 결과 생성: 화기작업 + 밀폐공간, 일부러 몇 군데 누락."""
import json, sys
sys.path.insert(0, ".")
import form_spec as F

fields, checks, triples, hazards = [], [], [], []
for fid, *_ in F.HEADER_FIELDS + F.BOTTOM_FIELDS:
    fields.append({"id": fid, "state": "filled"})
for sec in F.SUPPLEMENTARY.values():
    for fid, _ in sec["fields"] + sec.get("fields_on_complete", []):
        fields.append({"id": fid, "state": "empty"})
on = {"type_hot", "att_risk_assessment", "att_msds", "precheck_ok", "supp_confined",
      "k_hazard", "k_device", "k_method", "hzsec_hot", "hzsec_confined"}
ids = [c for c, _ in F.HEADER_CHECKS + F.BOTTOM_CHECKS + F.CONFINED_KNOWLEDGE]
ids += [f"supp_{k}" for k in F.SUPPLEMENTARY] + [f"hzsec_{k}" for k in F.HAZARD_SECTIONS]
for c in ids:
    checks.append({"id": c, "state": "checked" if c in on else "unchecked"})
needed = {"s_edu", "s_zone", "s_flammable", "s_spark_shield", "s_fire_watch", "s_extinguisher",
          "s_gas_test", "s_ppe", "s_ventilation", "s_confined_watch", "c_comm", "c_rescue"}
all_triples = [s for s, _ in F.SAFETY_ITEMS] + [c for sec in F.SUPPLEMENTARY.values() for c, _ in sec["checks"]]
for t in all_triples:
    n = t in needed
    triples.append({"id": t, "needed": "checked" if n else "unchecked",
                    "confirmed": "checked" if n and t != "s_spark_shield" else "unchecked",
                    "extended": "unchecked"})
for key, (_, items) in F.HAZARD_SECTIONS.items():
    for i in range(len(items)):
        hazards.append({"id": F.hazard_item_id(key, i), "mark": "good", "remedy": "empty"})
def setf(i, s):
    for x in fields:
        if x["id"] == i: x["state"] = s
def seth(i, m):
    for x in hazards:
        if x["id"] == i: x["mark"] = m
# 의도적 누락
setf("firewatch_sign", "empty"); setf("approver_sign", "empty")
for i in ("c_time", "c_checker_sign", "c_watcher_name", "c_watcher_sign"): setf(i, "filled")
setf("c_watcher_sign", "empty")
for i in ("during_check", "after16_sign", "done_vendor_time", "done_vendor_sign", "done_site_time",
          "done_site_sign", "ext_time", "ext_sign", "ext_reason"): setf(i, "empty")
setf("work_summary", "unclear")
seth("hz_hot_5", "none"); seth("hz_hot_6", "fix"); seth("hz_confined_6", "none")
data = {"pages_seen": ["page1", "page2"], "fields": fields, "checks": checks, "triples": triples,
        "gas": [{"column": 1, "time": "filled", "measurer": "filled", "o2": "20.8", "co": "35",
                 "h2s": "0", "ch4": "0", "co2": "0.04"}],
        "hazards": hazards, "apply_end_time": "17:00",
        "photo_quality_note": "작업개요 칸 일부에 빛 반사가 있음"}
json.dump(data, open("samples/demo_extraction.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
