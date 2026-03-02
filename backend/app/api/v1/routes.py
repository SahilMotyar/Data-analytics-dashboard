from __future__ import annotations

import asyncio
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import APIRouter, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel

from app.core.config import settings
from app.core.storage import read_json, write_json
from app.services.analysis_service import (
    analysis_path,
    get_analysis,
    metadata_path,
    set_column_type_override,
    upload_dir,
)
from app.services.export_service import create_pdf_report, create_share_link
from app.services.ai_service import generate_chat_answer
from app.workers.tasks import run_analysis_pipeline


router = APIRouter(prefix="/api/v1")


def _rate_limit_path() -> Path:
    return settings.storage_dir / "rate_limit.json"


def _apply_upload_rate_limit(client_key: str) -> None:
    store = read_json(_rate_limit_path(), {})
    day_key = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    client_day_key = f"{client_key}:{day_key}"
    count = int(store.get(client_day_key, 0))

    if count >= settings.upload_daily_limit:
        raise HTTPException(status_code=429, detail="Upload daily limit reached (50/day)")

    store[client_day_key] = count + 1
    write_json(_rate_limit_path(), store)


class StartAnalysisRequest(BaseModel):
    active_sheet: str | None = None


class OverrideTypeRequest(BaseModel):
    new_type: str


class ChatRequest(BaseModel):
    message: str
    history: list[dict[str, str]] = []


@router.post("/uploads")
async def upload_file(request: Request, file: UploadFile = File(...)) -> dict[str, Any]:
    client_key = request.client.host if request.client else "anonymous"
    _apply_upload_rate_limit(client_key)

    suffix = Path(file.filename).suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise HTTPException(status_code=400, detail="Only CSV/XLS/XLSX files are supported")

    contents = await file.read()
    size_mb = len(contents) / (1024 * 1024)
    if size_mb > settings.max_upload_mb:
        raise HTTPException(status_code=413, detail="Max file size is 50MB")

    upload_id = str(uuid.uuid4())
    target_dir = upload_dir(upload_id) / "original"
    target_dir.mkdir(parents=True, exist_ok=True)

    file_path = target_dir / file.filename
    with file_path.open("wb") as output:
        output.write(contents)

    sheet_names = []
    active_sheet = None
    if suffix in {".xlsx", ".xls"}:
        excel = pd.ExcelFile(file_path)
        sheet_names = excel.sheet_names
        active_sheet = sheet_names[0] if sheet_names else None

    warning = "Large file (>100K rows) may take longer to process"
    row_warning = None
    try:
        if suffix == ".csv":
            sample_rows = sum(1 for _ in file_path.open("r", encoding="utf-8", errors="ignore")) - 1
        else:
            sample_rows = int(pd.read_excel(file_path, sheet_name=active_sheet).shape[0])
        if sample_rows > 100_000:
            row_warning = warning
    except Exception:
        sample_rows = None

    metadata = {
        "upload_id": upload_id,
        "file_name": file.filename,
        "file_hash": hashlib.sha256(contents).hexdigest(),
        "file_size_mb": round(size_mb, 2),
        "status": "uploaded",
        "analysis_status": "not_started",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sheet_names": sheet_names,
        "active_sheet": active_sheet,
        "row_count_estimate": sample_rows,
        "warning": row_warning,
    }
    write_json(metadata_path(upload_id), metadata)

    return {"upload_id": upload_id, "metadata": metadata}


@router.get("/uploads/{upload_id}")
def get_upload(upload_id: str) -> dict[str, Any]:
    meta = read_json(metadata_path(upload_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")
    return meta


@router.post("/analysis/{upload_id}/start")
def start_analysis(upload_id: str, payload: StartAnalysisRequest) -> dict[str, Any]:
    meta = read_json(metadata_path(upload_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")

    if payload.active_sheet:
        meta["active_sheet"] = payload.active_sheet
    meta["analysis_status"] = "queued"
    write_json(metadata_path(upload_id), meta)

    job = run_analysis_pipeline.delay(upload_id)
    return {"upload_id": upload_id, "job_id": job.id, "status": "queued"}


@router.get("/analysis/{upload_id}/status")
def analysis_status(upload_id: str) -> dict[str, Any]:
    meta = read_json(metadata_path(upload_id), None)
    if not meta:
        raise HTTPException(status_code=404, detail="Upload not found")
    return {"upload_id": upload_id, "status": meta.get("analysis_status", "not_started"), "error": meta.get("error")}


@router.get("/analysis/{upload_id}/summary")
def analysis_summary(upload_id: str) -> dict[str, Any]:
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis.get("summary", {})


@router.get("/analysis/{upload_id}/key-findings")
def key_findings(upload_id: str) -> dict[str, Any]:
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis.get("key_findings", {})


@router.get("/analysis/{upload_id}/columns")
def analysis_columns(upload_id: str) -> list[dict[str, Any]]:
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis.get("columns", [])


@router.get("/analysis/{upload_id}/columns/{col}/stats")
def column_stats(upload_id: str, col: str) -> dict[str, Any]:
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    stats = analysis.get("column_stats", {}).get(col)
    if not stats:
        raise HTTPException(status_code=404, detail="Column not found")
    return stats


@router.patch("/analysis/{upload_id}/columns/{col}/type")
def override_column_type(upload_id: str, col: str, payload: OverrideTypeRequest) -> dict[str, Any]:
    try:
        stats = set_column_type_override(upload_id, col, payload.new_type)
        return {"column": col, "new_type": payload.new_type, "stats": stats}
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@router.post("/analysis/{upload_id}/export/pdf")
def export_pdf(upload_id: str) -> dict[str, Any]:
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    pdf_url = create_pdf_report(upload_id, analysis)
    return {"pdf_url": pdf_url}


@router.get("/analysis/{upload_id}/export/cleaned-csv")
def export_cleaned_csv(upload_id: str):
    cleaned_path = settings.outputs_dir / upload_id / "cleaned.csv"
    if not cleaned_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned CSV not available")
    return FileResponse(cleaned_path, media_type="text/csv", filename="cleaned.csv")


@router.get("/analysis/{upload_id}/export/excel")
def export_excel(upload_id: str):
    cleaned_path = settings.outputs_dir / upload_id / "cleaned.csv"
    excel_path = settings.outputs_dir / upload_id / "cleaned.xlsx"
    if not cleaned_path.exists():
        raise HTTPException(status_code=404, detail="Cleaned CSV not available")

    df = pd.read_csv(cleaned_path)
    df.to_excel(excel_path, index=False)
    return FileResponse(excel_path, media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", filename="cleaned.xlsx")


@router.get("/analysis/{upload_id}/export/stats-json")
def export_stats_json(upload_id: str):
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    stats_path = settings.outputs_dir / upload_id / "stats.json"
    write_json(stats_path, analysis)
    return FileResponse(stats_path, media_type="application/json", filename="stats.json")


@router.post("/analysis/{upload_id}/share")
def share_analysis(upload_id: str) -> dict[str, Any]:
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return create_share_link(upload_id)


@router.post("/analysis/{upload_id}/chat/stream")
async def chat_stream(upload_id: str, payload: ChatRequest):
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    meta = read_json(metadata_path(upload_id), {})

    answer = generate_chat_answer(analysis, meta, payload.message, payload.history)

    async def event_generator():
        for token in answer.split(" "):
            yield f"data: {json.dumps({'token': token + ' '})}\n\n"
            await asyncio.sleep(0.01)
        yield f"data: {json.dumps({'done': True})}\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@router.get("/shared/{token}")
def get_shared(token: str) -> dict[str, Any]:
    from app.services.export_service import serializer

    try:
        payload = serializer.loads(token, max_age=7 * 24 * 3600)
    except Exception as exc:
        raise HTTPException(status_code=400, detail="Invalid/expired share token") from exc

    upload_id = payload["upload_id"]
    analysis = get_analysis(upload_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return {"upload_id": upload_id, "analysis": analysis}
