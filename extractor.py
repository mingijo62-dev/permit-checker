"""사진 → 칸별 기재 상태(JSON) 추출. Google Gemini API(무료 사용량) 비전 + JSON 출력 사용.

AI는 '판단'하지 않고 '읽기'만 합니다. 누락 판단은 rules.py 담당.
"""
from __future__ import annotations

import io
import json
import logging
import os
import re
import time

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from PIL import Image, ImageOps

import form_spec as F

# 시도할 모델 순서. 앞 모델이 혼잡(503)·한도초과(429)·없음(404)·시간초과면 다음 모델로 넘어갑니다.
# 바꾸려면 GEMINI_MODEL 환경변수에 쉼표로 구분해 설정(예: "gemini-2.5-flash,gemini-2.0-flash").
MODELS = [m.strip() for m in os.environ.get(
    "GEMINI_MODEL", "gemini-2.5-flash,gemini-flash-latest,gemini-2.0-flash,gemini-2.5-flash-lite"
).split(",") if m.strip()]
# Vercel 함수 제한시간(300초) 안에 끝나도록: 모델 1회 시도 최대 60초, 전체 최대 230초
REQUEST_TIMEOUT_MS = 60_000
TOTAL_DEADLINE_S = 230
MAX_EDGE = 2400  # 긴 변 기준 리사이즈(px). 너무 작으면 손글씨 판독률이 떨어짐.

log = logging.getLogger("permit")
_client: genai.Client | None = None
_last_ok: str | None = None     # 마지막으로 성공한 모델(다음 요청에서 먼저 시도)
_missing: set[str] = set()      # 이 키로는 없는 모델(404)


def client() -> genai.Client:
    global _client
    if _client is None:
        if not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
            raise MissingKeyError()
        # GEMINI_API_KEY 환경변수를 자동으로 읽음. v1beta 에서만 쓸 수 있는 별칭/JSON 스키마 기능 때문에 v1beta 사용.
        _client = genai.Client(http_options=types.HttpOptions(
            api_version="v1beta",
            timeout=REQUEST_TIMEOUT_MS,
            retry_options=types.HttpRetryOptions(attempts=1),  # 같은 모델 재시도 대신 다른 모델로 넘어감
        ))
    return _client


_EXCLUDE = ("image", "tts", "audio", "live", "embed", "robotics", "computer", "veo", "imagen", "thinking")


def rank_flash_models(models) -> list[str]:
    """이 키로 쓸 수 있는 모델 중 이미지 판독용 Flash 모델을 좋은 순서로 정렬.
    일반 Flash 먼저(정확도), Lite 는 뒤(혼잡할 때 대안). 각각 정식판 우선, 그다음 최신 버전."""
    ranked = []
    for m in models:
        name = (m.name or "").removeprefix("models/")
        actions = m.supported_actions or []
        if "gemini" not in name or "flash" not in name or any(x in name for x in _EXCLUDE):
            continue
        if actions and "generateContent" not in actions:
            continue
        ver = re.search(r"gemini-(\d+(?:\.\d+)?)", name)
        key = (
            "lite" not in name,
            "preview" not in name and "exp" not in name,
            float(ver.group(1)) if ver else 0.0,
            name.endswith("latest") or not re.search(r"-\d{2,}", name.split("flash", 1)[-1]),  # 날짜 고정판보다 기본 이름
        )
        ranked.append((key, name))
    ranked.sort(reverse=True)
    return [n for _, n in ranked]


def pick_flash_model(models) -> str | None:
    r = rank_flash_models(models)
    return r[0] if r else None


_available: list[str] | None = None


def available_models() -> list[str]:
    """처음 한 번 모델 목록을 조회해 시도 순서를 정함(실패하면 기본 목록 사용). 서버가 켜져 있는 동안 재사용."""
    global _available
    if _available is None:
        env = os.environ.get("GEMINI_MODEL")
        if env:
            _available = MODELS
        else:
            try:
                found = rank_flash_models(client().models.list())
                # 기본 목록 중 실제로 있는 것을 우선, 그 외 발견된 모델을 뒤에
                pref = [m for m in MODELS if m in found or m.endswith("latest")]
                _available = pref + [m for m in found if m not in pref] or MODELS
                log.info("available models: %s", _available)
            except Exception:
                log.exception("model list failed; using defaults")
                _available = MODELS
    return _available


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


def _config(model: str) -> types.GenerateContentConfig:
    thinking = None
    if "2.5" in model:
        # 2.5 Flash 는 기본으로 오래 '생각'함 → 끄면 훨씬 빨라짐(칸 판독은 생각이 거의 필요 없음)
        thinking = types.ThinkingConfig(thinking_budget=0)
    elif re.search(r"gemini-[3-9]", model):
        thinking = types.ThinkingConfig(thinking_level="low")
    return types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT,
        response_mime_type="application/json",
        response_json_schema=GEMINI_SCHEMA,
        temperature=0,
        media_resolution="MEDIA_RESOLUTION_HIGH",  # 작은 체크박스/손글씨 판독용
        thinking_config=thinking,
    )


def extract(images: list[bytes]) -> dict:
    contents: list = []
    for i, raw in enumerate(images, 1):
        contents.append(f"사진 {i}")
        contents.append(types.Part.from_bytes(
            data=prepare_image(raw),
            mime_type="image/jpeg",
        ))
    contents.append("위 사진의 작업허가서를 판독해 JSON으로 보고하세요.")

    global _last_ok
    started = time.monotonic()
    chain = available_models()
    queue = ([_last_ok] if _last_ok else []) + [m for m in chain if m != _last_ok]
    tried: list[str] = []
    results: list[tuple[str, str]] = []  # (모델, 결과코드)
    resp = None
    while queue and len(tried) < 5:
        model = queue.pop(0)
        if model in _missing or model in tried:
            continue
        if time.monotonic() - started > TOTAL_DEADLINE_S - 30:
            break
        tried.append(model)
        for attempt in (1, 2):  # 일시적 혼잡(503)이면 같은 모델로 한 번 더
            t0 = time.monotonic()
            try:
                resp = client().models.generate_content(model=model, contents=contents, config=_config(model))
                log.info("gemini ok in %.0fs (model=%s, attempt=%d, before=%s)", time.monotonic() - t0, model, attempt, results)
                _last_ok = model
                break
            except genai_errors.ClientError as e:
                msg = str(e)
                if e.code == 404:
                    _missing.add(model)
                    results.append((model, "404"))
                elif e.code == 429:
                    results.append((model, "429"))
                    log.warning("model %s quota exceeded: %s", model, (e.message or msg)[:300])
                elif e.code in (400, 401, 403) and ("API_KEY" in msg or "API key" in msg or e.code != 400):
                    raise ExtractionError("서버의 GEMINI_API_KEY가 올바르지 않습니다. 관리자에게 문의하세요.", 500)
                else:
                    log.warning("model %s client error %s: %s", model, e.code, msg[:300])
                    results.append((model, str(e.code)))
                break
            except genai_errors.ServerError as e:
                results.append((model, str(e.code)))
                log.warning("model %s server error %s after %.0fs", model, e.code, time.monotonic() - t0)
                if e.code == 503 and attempt == 1 and time.monotonic() - started < TOTAL_DEADLINE_S - 60:
                    time.sleep(1.5)
                    continue
                break
            except Exception as e:
                if "Timeout" not in type(e).__name__:
                    raise
                results.append((model, "시간초과"))
                log.warning("model %s timeout after %.0fs", model, time.monotonic() - t0)
                break
        if resp is not None:
            break

    if resp is None:
        log.warning("all models failed: %s", results)
        codes = [c for _, c in results]
        detail = ", ".join(f"{m}:{c}" for m, c in results) or "시도한 모델 없음"
        if codes and all(c in ("429", "404") for c in codes) and "429" in codes:
            raise ExtractionError(
                "Gemini 무료 사용량 한도를 넘었습니다. 분당 한도라면 1분 뒤, 일일 한도라면 내일(한국시간 오후 4~5시 초기화) 다시 시도해 주세요.\n"
                f"[상세] {detail}", 429)
        if codes and all(c == "404" for c in codes):
            raise ExtractionError(f"이 API 키로 쓸 수 있는 Gemini 모델을 찾지 못했습니다.\n[상세] {detail}", 500)
        raise ExtractionError(f"Google AI 무료 서버가 혼잡합니다. 1~2분 뒤 다시 시도해 주세요.\n[상세] {detail}", 503)

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
