"""사진 → 칸별 기재 상태(JSON) 추출. Google Gemini API(무료 사용량) 비전 + JSON 출력 사용.

AI는 '판단'하지 않고 '읽기'만 합니다. 누락 판단은 rules.py 담당.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageOps

import form_spec as F

# 기본은 최신 Flash 별칭. 이 이름을 못 찾으면 키로 쓸 수 있는 Flash 모델을 자동으로 골라 씁니다.
# 특정 모델로 고정하려면 GEMINI_MODEL 환경변수 설정(예: gemini-2.5-flash).
MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-latest")
MAX_EDGE = 2400  # 긴 변 기준 리사이즈(px). 너무 작으면 손글씨 판독률이 떨어짐.

log = logging.getLogger("permit")
_client: genai.Client | None = None
_model: str = MODEL
_model_resolved = False


def client() -> genai.Client:
    global _client
    if _client is None:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise MissingKeyError()
        # GEMINI_API_KEY 환경변수를 자동으로 읽음. v1beta 에서만 쓸 수 있는 별칭/JSON 스키마 기능 때문에 v1beta 사용.
        _client = genai.Client(http_options=types.HttpOptions(api_version="v1beta"))
    return _client


_EXCLUDE = ("lite", "image", "tts", "audio", "live", "embed", "robotics", "computer", "veo", "imagen")


def pick_flash_model(models) -> str | None:
    """사용 가능한 모델 목록에서 이미지 판독용 Flash 모델 하나를 고름(정식판 우선, 그중 최신 버전)."""
    best, best_key = None, None
    for m in models:
        name = (m.name or "").removeprefix("models/")
        actions = m.supported_actions or []
        if "flash" not in name or any(x in name for x in _EXCLUDE):
            continue
        if actions and "generateContent" not in actions:
            continue
        ver = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        key = (
            "preview" not in name and "exp" not in name,  # 정식판 우선(무료 한도/안정성)
            float(ver.group(1)) if ver else 0.0,  # 그다음 최신 버전
            name.endswith("latest") or name.count("-") <= 2,  # 날짜가 붙은 고정판보다 기본 이름 우선
        )
        if best_key is None or key > best_key:
            best, best_key = name, key
    return best


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
def prepare_image(raw: bytes) -> bytes:
    """휴대폰 사진 회전 보정 + 리사이즈 + JPEG 변환."""
    im = Image.open(io.BytesIO(raw))
    im = ImageOps.exif_transpose(im).convert("RGB")
    w, h = im.size
    scale = MAX_EDGE / max(w, h)
    if scale < 1:
        im = im.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# 호출
# ---------------------------------------------------------------------------
class ExtractionError(RuntimeError):
    """사용자에게 그대로 보여줄 메시지와 HTTP 상태코드를 담는 오류."""

    def __init__(self, message: str, status: int = 422):
        super().__init__(message)
        self.status = status


class MissingKeyError(ExtractionError):
    def __init__(self):
        super().__init__("서버에 GEMINI_API_KEY가 설정되지 않았습니다. 관리자에게 문의하세요.", 500)


def _strip_additional_props(schema):
    """Gemini JSON 스키마 호환을 위해 additionalProperties 제거."""
    if isinstance(schema, dict):
        return {k: _strip_additional_props(v) for k, v in schema.items() if k != "additionalProperties"}
    if isinstance(schema, list):
        return [_strip_additional_props(v) for v in schema]
    return schema


GEMINI_SCHEMA = _strip_additional_props(OUTPUT_SCHEMA)


def extract(images: list[bytes]) -> dict:
    contents: list = []
    for i, raw in enumerate(images, 1):
        contents.append(f"사진 {i}")
        contents.append(types.Part.from_bytes(
            data=prepare_image(raw),
            mime_type="image/jpeg",
        ))
    contents.append("위 사진의 작업허가서를 판독해 JSON으로 보고하세요.")

    config = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=GEMINI_SCHEMA,
        temperature=0,
        media_resolution="MEDIA_RESOLUTION_HIGH",  # 작은 체크박스/손글씨 판독용
    )
    global _model, _model_resolved
    try:
        try:
            resp = client().models.generate_content(model=_model, contents=contents, config=config)
        except genai_errors.ClientError as e:
            if e.code != 404 or _model_resolved:
                raise
            # 모델 이름을 못 찾음 → 이 키로 쓸 수 있는 Flash 모델을 찾아 한 번 재시도
            _model_resolved = True
            found = pick_flash_model(client().models.list())
            if not found:
                raise
            log.warning("model %s not found, using %s", _model, found)
            _model = found
            resp = client().models.generate_content(model=_model, contents=contents, config=config)
    except genai_errors.ClientError as e:
        msg = str(e)
        if e.code == 429:
            raise ExtractionError("무료 사용량 한도에 도달했습니다. 1분 뒤 다시 시도하거나, 오늘 한도가 찼다면 내일 다시 시도해 주세요.", 429)
        if e.code in (400, 401, 403) and ("API_KEY" in msg or "API key" in msg or e.code != 400):
            raise ExtractionError("서버의 GEMINI_API_KEY가 올바르지 않습니다. 관리자에게 문의하세요.", 500)
        if e.code == 404:
            raise ExtractionError(f"AI 모델({_model})을 찾을 수 없습니다. GEMINI_MODEL 환경변수를 확인하세요.", 500)
        raise ExtractionError(f"AI 판독 요청 오류({e.code}): {e.message or msg}", 502)
    except genai_errors.ServerError as e:
        raise ExtractionError(f"AI 판독 서버 오류({e.code}). 잠시 후 다시 시도해 주세요.", 502)

    cand = (resp.candidates or [None])[0]
    reason = getattr(getattr(cand, "finish_reason", None), "name", "") if cand else ""
    if reason == "MAX_TOKENS":
        raise ExtractionError("판독 결과가 너무 길어 중단되었습니다. 다시 시도해 주세요.")
    if not resp.text:
        if reason in ("SAFETY", "PROHIBITED_CONTENT", "BLOCKLIST", "RECITATION"):
            raise ExtractionError("AI가 이 이미지 판독을 거절했습니다. 다른 사진으로 다시 시도해 주세요.")
        raise ExtractionError("AI 응답에 결과가 없습니다. 다시 시도해 주세요.")
    try:
        return json.loads(resp.text)
    except json.JSONDecodeError:
        raise ExtractionError("AI 응답을 해석하지 못했습니다. 다시 시도해 주세요.")
