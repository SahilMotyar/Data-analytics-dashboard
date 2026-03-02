from __future__ import annotations

from app.core.storage import read_json, write_json
from app.services.ai_service import generate_ai_summaries
from app.services.analysis_service import analysis_path, compute_analysis, metadata_path
from app.workers.celery_app import celery_app


@celery_app.task(name="run_analysis_pipeline")
def run_analysis_pipeline(upload_id: str) -> dict:
    meta = read_json(metadata_path(upload_id), {})
    meta["analysis_status"] = "running"
    write_json(metadata_path(upload_id), meta)

    try:
        analysis = compute_analysis(upload_id, sheet_name=meta.get("active_sheet"))
        analysis = generate_ai_summaries(analysis, meta)
        write_json(analysis_path(upload_id), analysis)

        meta["analysis_status"] = "completed"
        write_json(metadata_path(upload_id), meta)
        return {"status": "completed"}
    except Exception as exc:
        meta["analysis_status"] = "failed"
        meta["error"] = str(exc)
        write_json(metadata_path(upload_id), meta)
        raise
