"""작업허가서 누락 점검 웹 서버.

로컬 실행:  uvicorn app:app --host 0.0.0.0 --port 8000
Vercel:     api/index.py 가 이 app 을 그대로 사용합니다.

환경변수
- ANTHROPIC_API_KEY : Claude API 키 (필수)
- ACCESS_CODE       : 설정하면 이 코드를 입력한 사람만 판독 가능 (공개 URL 비용 보호용, 권장)
- PERMIT_MODEL      : 기본 claude-opus-5-5
- PERMIT_DEBUG      : 1 이면 AI 판독 원본을 응답에 포함
"""
from __future__ import annotations

import hmac
import json
import logging
import os
from pathlib import Path

import anthropic
from fastapi import FastAPI, File, Form, Header, HTTPException, UploadFile
from fastapi.responses import FileResponse

import extractor
import rules

BASE = Path(__file__).parent
MAX_FILES = 6
MAX_BYTES = 4 * 1024 * 1024  # Vercel 요청 본문 한도(4.5MB) 때문에 화면에서 미리 줄여서 보냄

log = logging.getLogger("permit")
app = FastAPI(title="작업허가서 누락 점검")


def _check_access(code: str | None):
    expected = os.environ.get("ACCESS_CODE", "")
    if expected and not hmac.compare_digest((code or "").encode(), expected.encode()):
        raise HTTPException(401, "접근 코드가 올바르지 않습니다.")


@app.get("/")
def index():
    # 로컬 실행용. Vercel 에서는 public/index.html 이 CDN 으로 바로 제공됩니다.
    return FileResponse(BASE / "public" / "index.html")


@app.get("/api/config")
def config():
    return {"access_code_required": bool(os.environ.get("ACCESS_CODE"))}


@app.post("/api/check")
async def check(
    files: list[UploadFile] = File(...),
    stage: str = Form("start"),
    x_access_code: str | None = Header(default=None),
):
    _check_access(x_access_code)
    if not files:
        raise HTTPException(400, "사진을 올려주세요.")
    if len(files) > MAX_FILES:
        raise HTTPException(400, f"사진은 최대 {MAX_FILES}장까지 올릴 수 있습니다.")
    images = []
    for f in files:
        raw = await f.read()
        if len(raw) > MAX_BYTES:
            raise HTTPException(413, f"{f.filename}: 사진 용량이 너무 큽니다.")
        images.append(raw)

    try:
        data = extractor.extract(images)
    except extractor.ExtractionError as e:
        raise HTTPException(422, str(e))
    except anthropic.AuthenticationError:
        raise HTTPException(500, "서버의 ANTHROPIC_API_KEY가 올바르지 않습니다. 관리자에게 문의하세요.")
    except anthropic.RateLimitError:
        raise HTTPException(429, "요청이 많습니다. 잠시 후 다시 시도해 주세요.")
    except anthropic.APIStatusError as e:
        log.exception("Claude API error")
        raise HTTPException(502, f"AI 판독 서버 오류({e.status_code}). 잠시 후 다시 시도해 주세요.")
    except anthropic.APIConnectionError:
        raise HTTPException(502, "AI 판독 서버에 연결할 수 없습니다.")
    except Exception as e:  # 이미지 파일 손상, 키 미설정 등
        log.exception("check failed")
        if "authentication" in str(e).lower():
            raise HTTPException(500, "서버에 ANTHROPIC_API_KEY가 설정되지 않았습니다. 관리자에게 문의하세요.")
        raise HTTPException(400, f"사진을 처리하지 못했습니다: {e}")

    result = rules.evaluate(data, stage)
    if os.environ.get("PERMIT_DEBUG"):
        result["raw"] = data
    return result


@app.post("/api/demo")
def demo(stage: str = Form("start")):
    """API 키 없이 화면을 확인하기 위한 예시 결과."""
    data = json.loads((BASE / "samples" / "demo_extraction.json").read_text(encoding="utf-8"))
    return rules.evaluate(data, stage)
