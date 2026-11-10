"""사진 → 칸별 기재 상태(JSON) 추출. Claude API 비전 + 구조화 출력 사용.

AI는 '판단'하지 않고 '읽기'만 합니다. 누락 판단은 rules.py 담당.
"""
from __future__ import annotations

import base64
import io
import json
import os

import anthropic
from PIL import Image, ImageOps

import form_spec as F

MODEL = os.environ.get("PERMIT_MODEL", "claude-opus-5-5")
MAX_EDGE = 2400  # 긴 변 기준 리사이즈(px). 너무 작으면 손글씨 판독률이 떨어짐.

_client: anthropic.Anthropic | None = None


def client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


# ---------------------------------------------------------------------------
# 프롬프트 구성
# ---------------------------------------------------------------------------
def _id_list() -> str:
    lines: list[str] = []
    lines.append("## fields (기재/서명 칸) → state: filled | empty | unclear")
    for fid, label, _ in F.HEADER_FIELDS:
        lines.append(f"- {fid}: {label}")
    for sec in F.SUPPLEMENTARY.values():
        for fid, label in sec["fields"] + sec.get("fields_on_complete", []):
            lines.append(f"- {fid}: [{sec['label']}] {label}")
    for fid, label, _ in F.BOTTOM_FIELDS:
        lines.append(f"- {fid}: {label}")

    lines.append("\n## checks (단일 체크박스) → state: checked | unchecked | unclear")
    for cid, label in F.HEADER_CHECKS + F.BOTTOM_CHECKS:
        lines.append(f"- {cid}: {label}")
    for key, sec in F.SUPPLEMENTARY.items():
        lines.append(f"- supp_{key}: 보충작업허가 '{sec['label']}' 행 맨 앞 □")
    for cid, label in F.CONFINED_KNOWLEDGE:
        lines.append(f"- {cid}: 밀폐공간 숙지여부 판정 □ {label}")
    for key, (label, _) in F.HAZARD_SECTIONS.items():
        lines.append(f"- hzsec_{key}: 2쪽 구분란 '{label}' □")

    lines.append(
        "\n## triples (□ 필요 / ○ 확인 / ◇ 연장확인 3칸 항목) → needed, confirmed, extended 각각 checked | unchecked | unclear"
    )
    for sid, label in F.SAFETY_ITEMS:
        lines.append(f"- {sid}: [안전조치 확인] {label}")
    for sec in F.SUPPLEMENTARY.values():
        for cid, label in sec["checks"]:
            lines.append(f"- {cid}: [{sec['label']}] {label}")

    lines.append(
        "\n## hazards (2쪽 잠재위험요인) → mark: good(양호) | na(무관) | fix(보완) | none(아무것도 표시 안 됨) | multiple(2개 이상 표시) | unclear, remedy: 보완내역 칸 filled | empty | unclear"
    )
    for key, (label, items) in F.HAZARD_SECTIONS.items():
        for i, text in enumerate(items):
            lines.append(f"- {F.hazard_item_id(key, i)}: [{label}] {i + 1}. {text}")
    return "\n".join(lines)


SYSTEM_PROMPT = f"""당신은 산업현장 '화기/일반/위험 작업허가서' 사진을 판독하는 검사원입니다.
사진에서 각 칸의 기재 상태를 있는 그대로 읽어 JSON으로 보고합니다. 누락 여부를 판단하거나 추측해서 채우지 마세요.

판독 규칙:
- filled: 글자, 숫자, 서명, 도장, 'N/A'나 '-' 같은 의도적 기재가 있으면 filled.
- empty: 칸이 사진에 분명히 보이는데 비어 있으면 empty.
- unclear: 사진이 흐리거나, 잘리거나, 빛 반사/그림자로 확인이 어렵거나, 해당 쪽이 사진에 없으면 unclear. 확신이 없으면 unclear를 고르세요.
- 체크박스는 ☑, ✓, V, O, ●, 빗금, 동그라미 등 어떤 표시든 있으면 checked.
- '(인)' 칸은 서명이나 도장이 있어야 filled. 인쇄된 '(인)' 글자만 있으면 empty.
- 안전조치 확인표와 보충작업허가 □○◇는 각각 따로 읽으세요: needed=□, confirmed=○, extended=◇.
- 가스농도 측정표는 측정 열(최대 3개)마다 측정시간/측정자 기재 여부와 O2, CO, H2S, CH4, CO2 값을 숫자 문자열(단위 제외)로 적고, 빈칸이면 ""로 적으세요.
- apply_end_time: 작업허가신청 시간의 종료 시각을 "HH:MM" 형식으로, 읽을 수 없으면 "".
- pages_seen: 사진에 실제로 보이는 쪽(page1 = 작업허가서 본문, page2 = 잠재위험요인 체크리스트).
- 아래 목록의 모든 id를 빠짐없이 한 번씩 보고하세요.

{_id_list()}
"""

STATE3 = {"type": "string", "enum": ["filled", "empty", "unclear"]}
CHK3 = {"type": "string", "enum": ["checked", "unchecked", "unclear"]}

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "pages_seen": {"type": "array", "items": {"type": "string", "enum": ["page1", "page2"]}},
        "fields": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "state": STATE3},
                "required": ["id", "state"],
                "additionalProperties": False,
            },
        },
        "checks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {"id": {"type": "string"}, "state": CHK3},
                "required": ["id", "state"],
                "additionalProperties": False,
            },
        },
        "triples": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "needed": CHK3,
                    "confirmed": CHK3,
                    "extended": CHK3,
                },
                "required": ["id", "needed", "confirmed", "extended"],
                "additionalProperties": False,
            },
        },
        "gas": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "column": {"type": "integer"},
                    "time": STATE3,
                    "measurer": STATE3,
                    "o2": {"type": "string"},
                    "co": {"type": "string"},
                    "h2s": {"type": "string"},
                    "ch4": {"type": "string"},
                    "co2": {"type": "string"},
                },
                "required": ["column", "time", "measurer", "o2", "co", "h2s", "ch4", "co2"],
                "additionalProperties": False,
            },
        },
        "hazards": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "mark": {
                        "type": "string",
                        "enum": ["good", "na", "fix", "none", "multiple", "unclear"],
                    },
                    "remedy": STATE3,
                },
                "required": ["id", "mark", "remedy"],
                "additionalProperties": False,
            },
        },
        "apply_end_time": {"type": "string"},
        "photo_quality_note": {"type": "string"},
    },
    "required": [
        "pages_seen", "fields", "checks", "triples", "gas", "hazards",
        "apply_end_time", "photo_quality_note",
    ],
    "additionalProperties": False,
}


# ---------------------------------------------------------------------------
# 이미지 전처리
# ---------------------------------------------------------------------------
def prepare_image(raw: bytes) -> str:
    """휴대폰 사진 회전 보정 + 리사이즈 + JPEG 변환 후 base64 반환."""
    im = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    scale = MAX_EDGE / max(w, h)
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return base64.standard_b64encode(buf.getvalue()).decode("ascii")


# ---------------------------------------------------------------------------
# 호출
# ---------------------------------------------------------------------------
class ExtractionError(RuntimeError):
    pass


def extract(images: list[bytes]) -> dict:
    content: list[dict] = []
    for i, raw in enumerate(images, 1):
        content.append({"type": "text", "text": f"사진 {i}"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": prepare_image(raw)},
        })
    content.append({"type": "text", "text": "위 사진의 작업허가서를 판독해 JSON으로 보고하세요."})

    with client().beta.messages.stream(
        model=MODEL,
        max_tokens=64000,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",  # 안전 분류기가 거절하면 서버가 다른 모델로 자동 재시도
        thinking={"type": "adaptive"},
        output_config={
            "effort": "high",  # 손글씨/체크 판독 정확도 우선
            "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA},
        },
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": content}],
    ) as stream:
        msg = stream.get_final_message()

    if msg.stop_reason == "refusal":
        raise ExtractionError("AI가 이 이미지 판독을 거절했습니다. 다른 사진으로 다시 시도해 주세요.")
    if msg.stop_reason == "max_tokens":
        raise ExtractionError("판독 결과가 너무 길어 중단되었습니다.")
    text = next((b.text for b in msg.content if b.type == "text"), None)
    if not text:
        raise ExtractionError("AI 응답에 결과가 없습니다.")
    return json.loads(text)
