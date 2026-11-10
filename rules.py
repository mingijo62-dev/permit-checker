"""추출 결과(JSON) → 누락/확인필요/주의 목록. AI 없이 규칙만으로 계산합니다.

판정 수준
- missing : 누락 (반드시 보완)
- danger  : 위험 (가스 기준 초과, 복원 불량 등)
- unclear : 사진으로 확인 불가 (재촬영 또는 육안 확인)
- warn    : 주의 (작업 종류에 비해 빠진 듯한 항목, 표시 불일치 등)
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass

import form_spec as F

STAGES = {
    "start": "작업 시작 전 (승인 완료 시점)",
    "during": "작업 중",
    "complete": "작업 완료 후",
}

# 작업 종류별로 '필요' 표시가 있어야 자연스러운 안전조치
EXPECTED_SAFETY = {
    "hot": ["s_flammable", "s_spark_shield", "s_fire_watch", "s_extinguisher", "s_gas_test", "s_ppe"],
    "confined": ["s_gas_test", "s_ventilation", "s_confined_watch", "s_ppe"],
    "radiation": ["s_rad_ppe", "s_rad_leak", "s_rad_exam", "s_rad_dosimeter"],
    "heavy": ["s_equip_route", "s_signalman"],
    "height": ["s_ppe"],
}
WORK_LABEL = {"hot": "화기작업", "confined": "밀폐공간 작업", "radiation": "방사선 작업",
              "heavy": "중장비 작업", "height": "고소 작업"}


@dataclass
class Finding:
    level: str
    section: str
    item: str
    message: str


class Checker:
    def __init__(self, data: dict, stage: str):
        self.stage = stage
        self.out: list[Finding] = []
        self.pages = set(data.get("pages_seen") or [])
        self.fields = {x["id"]: x["state"] for x in data.get("fields", [])}
        self.checks = {x["id"]: x["state"] for x in data.get("checks", [])}
        self.triples = {x["id"]: x for x in data.get("triples", [])}
        self.hazards = {x["id"]: x for x in data.get("hazards", [])}
        self.gas = data.get("gas", [])
        self.end_time = (data.get("apply_end_time") or "").strip()
        self.labels = self._labels()

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _labels() -> dict[str, str]:
        d: dict[str, str] = {}
        for fid, label, *_ in F.HEADER_FIELDS + F.BOTTOM_FIELDS:
            d[fid] = label
        for cid, label in F.HEADER_CHECKS + F.BOTTOM_CHECKS + F.SAFETY_ITEMS + F.CONFINED_KNOWLEDGE:
            d[cid] = label
        for sec in F.SUPPLEMENTARY.values():
            for i, label in sec["checks"] + sec["fields"] + sec.get("fields_on_complete", []):
                d[i] = label
        return d

    def add(self, level, section, item, message):
        self.out.append(Finding(level, section, item, message))

    def f(self, i):
        return self.fields.get(i, "unclear")

    def c(self, i):
        return self.checks.get(i, "unclear")

    def on(self, i):
        return self.c(i) == "checked"

    def need_field(self, i, section, label=None):
        label = label or self.labels.get(i, i)
        s = self.f(i)
        if s == "empty":
            self.add("missing", section, label, "기재 누락")
        elif s == "unclear":
            self.add("unclear", section, label, "사진으로 확인 불가")

    def need_check(self, i, section, msg="체크 누락"):
        s = self.c(i)
        if s == "unchecked":
            self.add("missing", section, self.labels.get(i, i), msg)
        elif s == "unclear":
            self.add("unclear", section, self.labels.get(i, i), "사진으로 확인 불가")

    def triple(self, i):
        t = self.triples.get(i) or {}
        return (t.get("needed", "unclear"), t.get("confirmed", "unclear"), t.get("extended", "unclear"))

    def check_triple(self, i, section, extension: bool):
        label = self.labels.get(i, i)
        need, conf, ext = self.triple(i)
        if need == "checked":
            if conf == "unchecked":
                self.add("missing", section, label, "'필요(□)' 표시했으나 '확인(○)' 누락")
            elif conf == "unclear":
                self.add("unclear", section, label, "확인(○) 표시 판독 불가")
            if extension and ext == "unchecked":
                self.add("missing", section, label, "연장작업 승인 시 '확인(◇)' 누락")
        elif need == "unchecked" and conf == "checked":
            self.add("warn", section, label, "확인(○)만 있고 필요(□) 표시 없음")
        elif need == "unclear":
            self.add("unclear", section, label, "필요(□) 표시 판독 불가")

    # ---- main ------------------------------------------------------------
    def run(self) -> list[Finding]:
        if "page1" in self.pages:
            self.page1()
        else:
            self.add("missing", "사진", "1쪽(작업허가서 본문)", "1쪽 사진이 없거나 인식되지 않음")
        if "page2" in self.pages:
            self.page2()
        else:
            self.add("missing", "사진", "2쪽(잠재위험요인 체크리스트)", "2쪽 사진이 없거나 인식되지 않음")
        return self.out

    @property
    def hot(self):
        return self.on("type_hot")

    def supp_on(self, key):
        return self.on(f"supp_{key}")

    @property
    def extension(self):
        return (
            self.on("ext_approved") or self.on("ext_rejected")
            or self.f("ext_time") == "filled" or self.f("ext_sign") == "filled"
        )

    def page1(self):
        S = "기본정보"
        types = [self.c(x) for x in ("type_hot", "type_general", "type_danger")]
        if "checked" not in types:
            if "unclear" in types:
                self.add("unclear", S, "작업종류(화기/일반/위험)", "사진으로 확인 불가")
            else:
                self.add("missing", S, "작업종류(화기/일반/위험)", "작업 종류 □ 미표시")

        for fid, label, group in F.HEADER_FIELDS:
            if fid.startswith("firewatch_") and not self.hot:
                continue
            self.need_field(fid, group, label)

        # 첨부서류
        S = "첨부서류"
        self.need_check("att_risk_assessment", S, "위험성평가 첨부 □ 미표시")
        if self.supp_on("power_off"):
            for a in ("att_single_line", "att_blueprint"):
                if not self.on(a):
                    self.add("warn", S, self.labels[a], "정전작업인데 첨부 표시 없음")
        if (self.supp_on("height") or self.supp_on("heavy")) and not self.on("att_work_plan"):
            self.add("warn", S, self.labels["att_work_plan"], "고소/중장비 작업인데 첨부 표시 없음")
        if self.hot and not self.on("att_msds"):
            self.add("warn", S, self.labels["att_msds"], "화기작업인데 MSDS 첨부 표시 없음(가연성/화학물질 취급 시 필요)")
        if self.supp_on("radiation") and not self.on("att_misc"):
            self.add("warn", S, self.labels["att_misc"], "방사선 작업인데 관련 서류 첨부 표시 없음")

        # 사전안전확인 결과
        S = "사전안전확인"
        ok, fix = self.c("precheck_ok"), self.c("precheck_fix")
        if ok != "checked" and fix != "checked":
            if "unclear" in (ok, fix):
                self.add("unclear", S, "결과(양호/보완)", "사진으로 확인 불가")
            else:
                self.add("missing", S, "결과(양호/보완)", "결과 □ 미표시")
        elif fix == "checked":
            self.add("warn", S, "결과: 보완", "보완 내용이 조치되었는지 확인 필요")

        # 안전조치 확인표
        S = "안전조치 확인"
        ext = self.extension and self.on("ext_approved")
        any_needed = False
        for sid, _ in F.SAFETY_ITEMS:
            self.check_triple(sid, S, ext)
            if self.triple(sid)[0] == "checked":
                any_needed = True
        if not any_needed:
            self.add("missing", S, "안전조치 전체", "필요(□) 표시된 안전조치가 하나도 없음")

        active = {"hot": self.hot, "confined": self.supp_on("confined"),
                  "radiation": self.supp_on("radiation"), "heavy": self.supp_on("heavy"),
                  "height": self.supp_on("height")}
        for work, on in active.items():
            if not on:
                continue
            for sid in EXPECTED_SAFETY[work]:
                if self.triple(sid)[0] == "unchecked":
                    self.add("warn", S, self.labels[sid], f"{WORK_LABEL[work]}인데 '필요' 표시 없음")

        # 보충작업허가
        for key, sec in F.SUPPLEMENTARY.items():
            S = f"보충작업허가·{sec['label']}"
            if self.supp_on(key):
                for fid, label in sec["fields"]:
                    self.need_field(fid, S, label)
                if self.stage == "complete":
                    for fid, label in sec.get("fields_on_complete", []):
                        self.need_field(fid, S, label)
                for cid, label in sec["checks"]:
                    self.check_triple(cid, S, ext)
                    if self.triple(cid)[0] == "unchecked":
                        self.add("warn", S, label, "필요(□) 표시 없음 — 해당 없는지 확인")
            else:
                filled = [fid for fid, _ in sec["fields"] if self.f(fid) == "filled"]
                if filled and self.c(f"supp_{key}") == "unchecked":
                    self.add("missing", S, f"{sec['label']} □", "내용은 기재되었으나 보충작업 □ 미표시")

        # 가스농도 측정
        need_gas = self.hot or self.supp_on("confined") or self.triple("s_gas_test")[0] == "checked"
        if need_gas:
            self.gas_check()
        if self.supp_on("confined"):
            for cid, _ in F.CONFINED_KNOWLEDGE:
                self.need_check(cid, "밀폐공간 숙지여부 판정", "숙지여부 □ 미표시")

        # 하단
        S = "승인/완료"
        for fid, label, when in F.BOTTOM_FIELDS:
            if when == "start" or (when == "during" and self.stage in ("during", "complete")):
                self.need_field(fid, S, label)
            elif when == "complete" and self.stage == "complete":
                self.need_field(fid, S, label)
        if self.after_16():
            self.need_field("after16_sign", S)
        if self.stage == "complete":
            for who in ("vendor", "site"):
                ok, bad = self.c(f"done_{who}_ok"), self.c(f"done_{who}_bad")
                name = "업체감독자" if who == "vendor" else "현업감독자"
                if ok != "checked" and bad != "checked":
                    lvl = "unclear" if "unclear" in (ok, bad) else "missing"
                    self.add(lvl, S, f"{name} 복원(조치)상태", "양호/불량 □ 미표시" if lvl == "missing" else "사진으로 확인 불가")
                elif bad == "checked":
                    self.add("danger", S, f"{name} 복원(조치)상태", "'불량' 표시 — 복원 조치 필요")
        if self.extension:
            S = "작업허가연장승인"
            self.need_field("ext_time", S)
            self.need_field("ext_sign", S)
            if not (self.on("ext_approved") or self.on("ext_rejected")):
                self.add("missing", S, "승인/불승인", "승인 여부 □ 미표시")
            if self.on("ext_rejected"):
                self.need_field("ext_reason", S)

    def after_16(self) -> bool:
        m = re.match(r"^(\d{1,2}):?(\d{2})$", self.end_time)
        if not m:
            return False
        h, mi = int(m.group(1)), int(m.group(2))
        return h * 60 + mi > 16 * 60

    def gas_check(self):
        S = "가스농도 측정"
        cols = [g for g in self.gas if any(
            [g.get("time") == "filled", g.get("measurer") == "filled"]
            + [str(g.get(k, "")).strip() for k in F.GAS_LIMITS]
        )]
        if not cols:
            self.add("missing", S, "측정 기록", "화기/밀폐 작업인데 가스농도 측정값이 없음")
            return
        for g in cols:
            col = f"{g.get('column', '?')}차 측정"
            for k in ("time", "measurer"):
                s = g.get(k, "unclear")
                name = "측정시간" if k == "time" else "측정자"
                if s == "empty":
                    self.add("missing", S, f"{col} {name}", "기재 누락")
                elif s == "unclear":
                    self.add("unclear", S, f"{col} {name}", "사진으로 확인 불가")
            for k, (label, std, ok) in F.GAS_LIMITS.items():
                raw = str(g.get(k, "")).strip()
                if not raw:
                    self.add("missing", S, f"{col} {label}", "측정값 누락")
                    continue
                num = re.sub(r"[^0-9.\-]", "", raw)
                try:
                    v = float(num)
                except ValueError:
                    self.add("unclear", S, f"{col} {label}", f"값 판독 불가('{raw}')")
                    continue
                if not ok(v):
                    self.add("danger", S, f"{col} {label}", f"측정값 {raw} — 기준({std}) 벗어남, 작업 중지 검토")

    def page2(self):
        expected = set()
        if self.hot:
            expected.add("hot")
        if self.on("type_general"):
            expected.add("general")
        for supp, hz in F.SUPP_TO_HAZARD.items():
            if self.supp_on(supp):
                expected.add(hz)

        for key, (label, items) in F.HAZARD_SECTIONS.items():
            S = f"잠재위험요인·{label}"
            box = self.c(f"hzsec_{key}")
            if box != "checked" and key not in expected:
                continue
            if box == "unchecked" and key in expected:
                self.add("missing", S, f"구분 '{label}' □", "1쪽에서 해당 작업으로 표시했으나 2쪽 구분 □ 미표시")
            for i, text in enumerate(items):
                hz = self.hazards.get(F.hazard_item_id(key, i)) or {}
                mark, remedy = hz.get("mark", "unclear"), hz.get("remedy", "unclear")
                item = f"{i + 1}. {text}"
                if mark == "none":
                    self.add("missing", S, item, "양호/무관/보완 중 표시 없음")
                elif mark == "multiple":
                    self.add("warn", S, item, "2개 이상 표시됨 — 하나만 선택")
                elif mark == "unclear":
                    self.add("unclear", S, item, "사진으로 확인 불가")
                elif mark == "fix":
                    if remedy == "empty":
                        self.add("missing", S, item, "'보완' 표시했으나 보완내역 미기재")
                    elif remedy == "unclear":
                        self.add("unclear", S, item, "보완내역 판독 불가")


LEVEL_ORDER = {"danger": 0, "missing": 1, "unclear": 2, "warn": 3}


def evaluate(data: dict, stage: str = "start") -> dict:
    if stage not in STAGES:
        stage = "start"
    findings = sorted(Checker(data, stage).run(), key=lambda x: LEVEL_ORDER[x.level])
    counts = {k: sum(1 for x in findings if x.level == k) for k in LEVEL_ORDER}
    passed = counts["missing"] == 0 and counts["danger"] == 0 and counts["unclear"] == 0
    return {
        "stage": stage,
        "stage_label": STAGES[stage],
        "passed": passed,
        "counts": counts,
        "findings": [asdict(x) for x in findings],
        "photo_quality_note": data.get("photo_quality_note", ""),
    }
